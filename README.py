# %%
# Imports
import numpy as np
import pandas as pd
import torch as t
from einops import rearrange
from IPython.display import display, Video

from huggingface_hub import hf_hub_download

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3 import PPO

import plotly.express as px

import circrl.hooks as chk
import circrl.rollouts as cro
import circrl.probing as cpr

# Boilerplate to make sure we can reload modules
try:
    # pylint: disable=import-outside-toplevel
    from IPython import get_ipython  # type: ignore

    # pylint: enable=import-outside-toplevel

    get_ipython().run_line_magic("reload_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except AttributeError:
    pass


# Define an environment setup function
def env_setup(seed):
    env = make_atari_env(
        "PongNoFrameskip-v4",
        n_envs=1,
        seed=seed,
        # wrapper_kwargs={"action_repeat_probability": 0.25},
    )
    env = VecFrameStack(env, n_stack=4)
    return env, {"seed": seed}


# Download model, and load it for prediction
model_path = hf_hub_download(
    repo_id="sb3/ppo-PongNoFrameskip-v4", filename="ppo-PongNoFrameskip-v4.zip"
)
env = env_setup(0)[0]
dummy_ppo = PPO(ActorCriticCnnPolicy, env, 0.0)
# Custom object overrides are required for sb3 cross-version compatibility
custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
    "observation_space": dummy_ppo.observation_space,
    "action_space": dummy_ppo.action_space,
}
model = PPO.load(model_path, custom_objects=custom_objects)

# %%
# Run a single episode using CircRL
seq, episode_return, step_cnt = cro.run_rollout(
    model.predict,
    env,
    max_episodes=1,
)

# Display video
vid_fn, fps = cro.make_video_from_renders(seq.renders, fps=30.0)
display(Video(vid_fn, embed=True, width=300))


# %%
# Run many rollouts using CircRL
cro.make_dataset(
    model.predict,
    "Testing",
    "datasets",
    10,
    env_setup,
    initial_seed=0,
    run_rollout_kwargs=dict(show_pbar=False),
)

# %%
# Run a rollout while caching
LAYER = "features_extractor.cnn.1"
cache_list = []


def cache_predict(*args, **kwargs):
    """Custom predict func that caches activations from an intermediate
    convolutional layer."""
    with chk.HookManager(model.policy, cache=[LAYER]) as cache_result:
        action = model.predict(*args, **kwargs)
        cache_list.append(cache_result[LAYER])
    return action


# Run a single episode using CircRL, using the custom predict function
# that caches activations from an intermediate convolutional layer.
seq, episode_return, step_cnt = cro.run_rollout(
    cache_predict,
    env,
    max_episodes=1,
)

# Concatenate the cached activations so that batch dimension contains
# all the chached timesteps
activs = t.cat(cache_list, dim=0)
print(activs.shape)

# %%
# Demonstrate probing by finding conv channels that seem to track ball position

# First, find the ball in each render using a simple conv filter
BALL_THRESH = 1500
kernel = rearrange(
    t.tensor(
        [
            [-1, -1, -1, -1],
            [-1, 1, 1, -1],
            [-1, 1, 1, -1],
            [-1, 1, 1, -1],
            [-1, 1, 1, -1],
            [-1, -1, -1, -1],
        ]
    ).to(t.float),
    "i j -> 1 1 i j",
)
is_ball = (
    t.nn.functional.conv2d(
        rearrange(
            t.tensor(seq.renders[:, :, :, 2].values), "t i j -> t 1 i j"
        ).to(t.float),
        kernel,
        padding="same",
    )[:, 0, :, :]
    > BALL_THRESH
)
is_ball_nz = is_ball.nonzero()
ball_ij = t.full((len(seq.renders), 2), np.nan)
ball_ij[is_ball_nz[:, 0], :] = is_ball_nz[:, 1:].to(t.float)

# Convert position in render coords to position in coords of the conv
# layer


# %%
