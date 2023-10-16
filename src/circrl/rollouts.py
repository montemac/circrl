"""Functions and classes for running rollouts and saving associated data."""

import os
from typing import Optional, Tuple, Dict, List, Union, Callable, Any
from dataclasses import dataclass
import warnings
import datetime
import json
import pickle
import blosc
import numpy as np
import torch as t
from torch import nn
import xarray as xr
import gym
import gym.vector
from tqdm.auto import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from einops import rearrange

os.environ["SDL_VIDEODRIVER"] = "dummy"


@dataclass
class RlStepSeq:
    """Dataclass for storing a sequence of RL steps.  Each attribute is
    an xarray DataArray (or dict thereof), with the step dimension being
    the first dimension of each array."""

    renders: Optional[xr.DataArray]
    obs: xr.DataArray
    actions: xr.DataArray
    model_states: Optional[xr.DataArray]
    rewards: xr.DataArray
    dones: xr.DataArray
    custom: Dict[str, xr.DataArray]


def run_rollout(
    model_or_func: Union[nn.Module, Callable],
    env: gym.vector.VectorEnv,
    deterministic: bool = False,
    seed: int = 1,
    max_episodes: Optional[int] = 1,
    max_steps: Optional[int] = None,
    show_pbar: bool = True,
    return_renders: bool = True,
    custom_data_funcs: Optional[dict] = None,
) -> Tuple[RlStepSeq, List[float], int]:
    """Run episodes, using provided policy in provided environment.
    Save and return episode data as an RlStepSeq object."""
    assert (
        max_episodes is not None or max_steps is not None
    ), "Must provide max episodes or steps."
    if custom_data_funcs is None:
        custom_data_funcs = {}
    # Confirm that the environment is vectorized (required)
    assert hasattr(env, "num_envs"), "Environment must be vectorized"
    # This function only works for a single environment
    assert env.num_envs == 1, "Only single environment is supported."  # type: ignore
    # Determine what function to call for getting actions
    if isinstance(model_or_func, nn.Module):
        if hasattr(model_or_func, "predict"):
            predict_func = model_or_func.predict
        else:
            predict_func = model_or_func.forward
    else:
        predict_func = model_or_func
    # Init all the seeds!
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.deterministic = True
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    try:
        env.seed(seed)  # type: ignore
    except (TypeError, AttributeError):
        warnings.warn("Seed setting failed.")
    # Reset the environment to get the initial observation
    obs = env.reset()
    # Init lists to store all the data since we don't know the number of steps yet
    renders_all = []
    obs_all = []
    actions_all = []
    model_states_all = []
    rewards_all = []
    dones_all = []
    custom_data_all = {nm: [] for nm in custom_data_funcs.keys()}
    step_cnt = 0
    is_complete = False
    episode_returns: List[float] = []
    with tqdm(total=max_steps, disable=not show_pbar) as pbar:
        while not is_complete:
            # Convert the observation from dict form if needed
            if isinstance(obs, dict):
                obs = obs["rgb"]
            # Get the latest frame
            render = env.render()
            # Get action and model states
            with t.no_grad():
                action, model_states, *extra = predict_func(
                    obs, deterministic=deterministic
                )
            # Vectorize model states if it's None for consistent interface
            model_states = [None] if model_states is None else model_states
            # Call any provided custom data functions (these are called before the env step,
            # so we have env, current render/obs, agent action/state, but no reward/done/obs_next)
            for nm, func in custom_data_funcs.items():
                custom_data_all[nm].append(
                    func(
                        env=env,
                        render=render,
                        obs=obs[0],
                        action=action[0],
                        model_states=model_states[0],
                        predict_extra=extra,
                    )
                )
            # Step environment
            obs_next, rewards, dones, infos = env.step(action)  # type: ignore
            # Store the obs, action, model_state, reward, done
            if return_renders:
                renders_all.append(render)
            obs_all.append(obs[0])
            actions_all.append(action[0])
            model_states_all.append(model_states[0])
            rewards_all.append(rewards[0])
            dones_all.append(dones[0])
            step_cnt += env.num_envs
            # Detect whether episode has terminated
            # TODO: done sometimes set when losing a life in Atari games? Use something else here?
            if dones[0]:
                episode_returns.append(infos[0].get("episode", {"r": 0})["r"])
                # Terminate after max episodes if provided
                if (
                    max_episodes is not None
                    and len(episode_returns) >= max_episodes
                ):
                    is_complete = True
            # Terminate after max steps if provided
            if max_steps is not None and step_cnt >= max_steps:
                is_complete = True
            # Move to new observation for next loop
            obs = obs_next
            # Tick the progress bar
            pbar.update(1)
    # Wrap every up and return
    env.close()
    # Make the sequence data object
    # Renders
    coords = {"step": np.arange(len(renders_all))}
    if return_renders:
        renders_da = xr.DataArray(
            rearrange(renders_all, "step h w rgb -> step h w rgb"),
            dims=("step", "h", "w", "rgb"),
            coords=coords,
        )
    else:
        renders_da = None
    # Observations
    obs_da = xr.DataArray(
        rearrange(obs_all, "step ... -> step ..."),
        dims=tuple(
            ["step"]
            + [f"obd{ii}" for ii, sz in enumerate(obs_all[0].shape) if sz > 1]
        ),
        coords=coords,
    )
    # Actions
    actions_da = xr.DataArray(
        np.array(actions_all), dims=("step"), coords=coords
    )
    # Model states
    if model_states_all[0] is not None:
        model_states_da = xr.DataArray(
            rearrange(model_states_all, "step ms -> step ms"),
            dims=("step", "model_state"),
            coords=coords,
        )
    else:
        model_states_da = None
    # Rewards
    rewards_da = xr.DataArray(
        np.array(rewards_all), dims=("step"), coords=coords
    )
    # Dones
    dones_da = xr.DataArray(np.array(dones_all), dims=("step"), coords=coords)
    # Custom data
    custom_data_das = {}
    for nm, custom_data in custom_data_all.items():
        if isinstance(custom_data[0], np.ndarray):
            custom_data_das[nm] = xr.DataArray(
                rearrange(custom_data, "step ... -> step ..."),
                dims=tuple(
                    ["step"]
                    + [
                        f"{nm}_d{ii}"
                        for ii, sz in enumerate(custom_data[0].shape)
                    ]
                ),
                coords=coords,
            )
        else:
            custom_data_das[nm] = xr.DataArray(
                np.array(custom_data), dims=("step"), coords=coords
            )
    # Final combined data object
    seq_data = RlStepSeq(
        renders_da,
        obs_da,
        actions_da,
        model_states_da,
        rewards_da,
        dones_da,
        custom_data_das,
    )
    return seq_data, episode_returns, step_cnt


def split_seq(seq: RlStepSeq, start_inds: List[int]) -> List[RlStepSeq]:
    """Split sequence object into subsequences, with start_inds defining the start indices,
    of each sequence.  Sequences will span from start_inds[i] to start_inds[i+1]-1 inclusive.
    There will be len(start_inds)-1 subsequences returned."""
    new_seqs = []
    for start_ind, end_ind in zip(start_inds[:-1], start_inds[1:]):
        args = [
            aa.sel(step=slice(start_ind, end_ind)) if aa is not None else aa
            for aa in [
                seq.renders,
                seq.obs,
                seq.actions,
                seq.model_states,
                seq.rewards,
                seq.dones,
            ]
        ]
        args.append(
            {
                nm: aa.sel(step=slice(start_ind, end_ind))
                for nm, aa in seq.custom.items()
            }
        )
        new_seq = RlStepSeq(*args)
        new_seqs.append(new_seq)
    return new_seqs


def make_video_from_renders(
    renders: xr.DataArray, fps: float = 30.0
) -> Tuple[str, float]:
    """Make a video from a sequence of renders.  Returns the filename and the fps."""
    vid_fn = "temp_seq_video.mp4"
    clip = ImageSequenceClip([aa.to_numpy() for aa in renders], fps=fps)
    clip.write_videofile(vid_fn, logger=None)
    return vid_fn, fps


class TempVideoFileFromSeq:
    """Context manager for creating a temporary video file from a sequence of renders."""

    def __init__(self, seq: RlStepSeq, fps=30.0):
        self.seq = seq
        self.fps = fps
        self.vid_fn = ""

    def __enter__(self):
        # Create the temp file from the renders, return the filename
        if self.seq.renders is None:
            raise ValueError("Sequence does not contain renders.")
        else:
            self.vid_fn, _ = make_video_from_renders(
                self.seq.renders, fps=self.fps
            )

    def __exit__(self, *args, **kwargs):
        # Delete the temp file if it exists
        if os.path.isfile(self.vid_fn):
            os.remove(self.vid_fn)


def make_dataset(
    predict_func: Callable,
    desc: str,
    path: str,
    num_episodes: int,
    env_setup_func: Callable,
    initial_seed: int = 0,
    seq_mod_func: Optional[Callable] = None,
    run_rollout_kwargs: Optional[Dict[str, Any]] = None,
):
    """Run a number of episodes and save the data to a dataset folder."""
    # Set defaults
    if run_rollout_kwargs is None:
        run_rollout_kwargs = {}
    # Create a timestamped output folder
    utc = datetime.datetime.utcnow()
    dr = utc.strftime(os.path.join(path, "%Y%m%dT%H%M%S"))
    os.mkdir(dr)
    # Create a dataset metadata file
    metadata = {
        "desc": desc,
        "timestamp": utc.strftime("%Y%m%dT%H%M%S"),
        "num_episodes": num_episodes,
        "run_rollout_kwargs": run_rollout_kwargs,
    }
    with open(os.path.join(dr, "metadata.json"), "w", encoding="utf-8") as fl:
        json.dump(metadata, fl, indent=2, default=str)
    # Loop through episodes
    ep_cnt = 0
    seed = initial_seed
    for ep_cnt in tqdm(range(num_episodes)):
        # Create env
        venv, episode_metadata = env_setup_func(seed=seed)
        # Run rollout
        # run_rollout_kwargs['show_pbar'] = False
        seq, episode_returns, step_cnt = run_rollout(
            predict_func, venv, seed=seed, **run_rollout_kwargs
        )
        # Prepare episode data object
        episode_data = {
            "episode_metadata": episode_metadata,
            "seq": seq if seq_mod_func is None else seq_mod_func(seq),
            "episode_returns": episode_returns,
            "step_cnt": step_cnt,
            "ep_cnt": ep_cnt,
        }
        # Save episode data
        pickled_data = pickle.dumps(
            episode_data, protocol=pickle.HIGHEST_PROTOCOL
        )
        compressed_pickle = blosc.compress(pickled_data)
        with open(os.path.join(dr, f"{ep_cnt}.dat"), "wb") as fl:
            fl.write(compressed_pickle)
        # Increment seed
        seed += 1


def load_saved_rollout(fn):
    """Load a saved rollout from a file."""
    with open(fn, "rb") as f:
        compressed_pickle = f.read()
    depressed_pickle = blosc.decompress(compressed_pickle)
    return pickle.loads(depressed_pickle)


# %%
# Basic tests

# if __name__ == "__main__":
#     # Dummy predict function
#     def predict(obs, deterministic=False):
#         return np.array([0]), None

#     # Test with a normal gym env
#     # env = gym.make('CartPole-v1')
#     # run_rollout(predict, env, max_steps=50)

#     # Test an atari env
#     # from stable_baselines3.common.vec_env import VecFrameStack
#     # from stable_baselines3.common.env_util import make_atari_env
#     # env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)
#     # env = VecFrameStack(env, n_stack=4)
#     # seq, episode_returns, step_cnt = run_rollout(predict, env, max_steps=50)

#     # Test with a procgen gym3 env
#     import procgen

#     env = procgen.ProcgenEnv(
#         num_envs=1, env_name="maze", num_levels=1, start_level=0
#     )
#     seq, episode_returns, step_cnt = run_rollout(predict, env, max_steps=50)
