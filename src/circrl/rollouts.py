# %%
import os
import typing
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import xarray as xr
import gym
from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from einops import rearrange

os.environ["SDL_VIDEODRIVER"] = "dummy"

@dataclass
class RlStepSeq():
    renders:      xr.DataArray    # The RGB or grayscale outputs of the environment rendering at each sequence point
    obs:          xr.DataArray  
    actions:      xr.DataArray  
    model_states: typing.Optional[xr.DataArray]
    rewards:      xr.DataArray  
    dones:        xr.DataArray  

def run_rollout(model_or_func, env: gym.Env, deterministic: bool=False, 
        seed: int=1, max_episodes: typing.Optional[int]=1,
        max_steps: typing.Optional[int]=None) -> typing.Tuple[RlStepSeq, float, int]:
    '''Run episodes, using provided policy in provided environment.
    Save and return episode data as an RlStepSeq object.'''
    assert max_episodes is not None or max_steps is not None, "Must provide max episodes or steps."
    # Confirm that the environment is vectorized (required)
    assert hasattr(env, 'num_envs'), "Environment must be vectorized"
    # This function only works for a single environment
    assert env.num_envs == 1, "Only single environment is supported."
    # Determine what function to call for getting actions
    if isinstance(model_or_func, nn.Module):
        if hasattr(model_or_func, 'predict'):
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
        env.seed(seed)
    except (TypeError, AttributeError):
        warnings.warn('Seed setting failed.')
    # Reset the environment to get the initial observation
    obs = env.reset()
    # Init lists to store all the data since we don't know the number of steps yet
    renders_all = []
    obs_all = []
    actions_all = []
    model_states_all = []
    rewards_all = []
    dones_all = []
    step_cnt = 0
    is_complete = False
    episode_returns = []
    with tqdm(total=max_steps) as pbar:
        while not is_complete:
            # Convert the observation from dict form if needed
            if isinstance(obs, dict):
                obs = obs['rgb']
            # Store the latest frame
            renders_all.append(env.render(mode='rgb_array'))
            # Get action and model states
            with t.no_grad():
                action, model_states = predict_func(obs, deterministic=deterministic)
            # Step environment
            obs_next, rewards, dones, infos = env.step(action)
            # Store the obs, action, model_state, reward, done
            obs_all.append(obs)
            actions_all.append(action)
            model_states_all.append(model_states)
            rewards_all.append(rewards)
            dones_all.append(dones)
            step_cnt += env.num_envs
            # Detect whether episode has terminated
            # TODO: done sometimes set when losing a life in Atari games? Use something else here?
            if dones[0]:
                episode_returns.append(infos[0].get('episode', {'r':0})['r'])
                # Terminate after max episodes if provided
                if max_episodes is not None and len(episode_returns) >= max_episodes:
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
    # # TEMP: debug, hackily show frame index as a pixel inversion!
    # for ii, render in enumerate(renders_all):
    #     ww = ii % render.shape[1]
    #     render[10,ww,:] = 255 - render[10,ww,:]
    coords = {'step': np.arange(len(renders_all))}
    renders_da = xr.DataArray(rearrange(renders_all, 'step h w rgb -> step h w rgb'), 
        dims=('step', 'h', 'w', 'rgb'), coords=coords)
    # Observations
    obs_da = xr.DataArray(np.squeeze(rearrange(obs_all, 'step ... -> step ...')), 
        dims=tuple(['step']+['obd{}'.format(ii) for ii, sz in 
                enumerate(obs_all[0].shape) if sz > 1]), coords=coords)
    # Actions
    actions_da = xr.DataArray(rearrange(actions_all, 'step a -> step a'), 
        dims=('step', 'action'), coords=coords)
    # Model states
    if model_states_all[0] is not None:
        model_states_da = xr.DataArray(rearrange(model_states_all, 'step ms -> step ms'), 
            dims=('step', 'model_state'), coords=coords)
    else:
        model_states_da = None
    # Rewards
    rewards_da = xr.DataArray(np.squeeze(np.array(rewards_all)), dims=('step'), coords=coords)
    # Dones
    dones_da = xr.DataArray(np.squeeze(np.array(dones_all)), dims=('step'), coords=coords)
    # Final combined data object
    seq_data = RlStepSeq(renders_da, obs_da, actions_da, model_states_da, rewards_da, dones_da)
    return seq_data, episode_returns, step_cnt

def split_seq(seq: RlStepSeq, start_inds):
    '''Split sequence object into subsequences, with start_inds defining the start indices,
    of each sequence.  Sequences will span from start_inds[i] to start_inds[i+1]-1 inclusive.
    There will be len(start_inds)-1 subsequences returned.'''
    new_seqs = []
    for start_ind, end_ind in zip(start_inds[:-1], start_inds[1:]):
        args = [aa.sel(step=slice(start_ind, end_ind)) if aa is not None else aa for aa in 
            [seq.renders, seq.obs, seq.actions, seq.model_states, seq.rewards, seq.dones]]
        new_seq = RlStepSeq(*args)
        new_seqs.append(new_seq)
    return new_seqs

def make_video_from_renders(renders: xr.DataArray, fps=30.):
    vid_fn = "temp_seq_video.mp4"
    clip = ImageSequenceClip([aa.to_numpy() for aa in renders], 
        fps=fps)
    clip.write_videofile(vid_fn, logger=None)
    return vid_fn, fps

class TempVideoFileFromSeq:
    def __init__(self, seq: RlStepSeq, fps=30.):
        self.seq = seq
        self.fps = fps
    def __enter__(self):
        # Create the temp file from the renders, return the filename
        self.vid_fn, _ = make_video_from_renders(seq.renders, fps=self.fps)
    def __exit__(self, *args, **kwargs):
        # Delete the temp file if it exists
        if os.path.isfile(self.vid_fn):
            os.remove(self.vid_fn)

# %%
# Basic tests

if __name__ == "__main__":
    # Dummy predict function
    def predict(obs, deterministic=False):
        return np.array([0]), None

    # Test with a normal gym env
    # env = gym.make('CartPole-v1')
    # run_rollout(predict, env, max_steps=50)

    # Test an atari env
    # from stable_baselines3.common.vec_env import VecFrameStack
    # from stable_baselines3.common.env_util import make_atari_env
    # env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)
    # env = VecFrameStack(env, n_stack=4)
    # seq, episode_returns, step_cnt = run_rollout(predict, env, max_steps=50)

    # Test with a procgen gym3 env
    import procgen
    env = procgen.ProcgenEnv(num_envs=1, env_name='maze', num_levels=1, start_level=0)
    seq, episode_returns, step_cnt = run_rollout(predict, env, max_steps=50)