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
ACTIVS_LAYER = "features_extractor.cnn.1"
LOGITS_LAYER = "action_net"
activs_list = []
logits_list = []


def cache_predict(*args, **kwargs):
    """Custom predict func that caches activations from an intermediate
    convolutional layer."""
    with chk.HookManager(
        model.policy, cache=[ACTIVS_LAYER, LOGITS_LAYER]
    ) as cache_result:
        action = model.predict(*args, **kwargs)
        activs_list.append(cache_result[ACTIVS_LAYER])
        logits_list.append(cache_result[LOGITS_LAYER])
    return action


# Run a single episode using CircRL, using the custom predict function
# that caches activations from an intermediate convolutional layer.
# Create a new environment to ensure deterministic results
env = env_setup(0)[0]
seq, episode_return, step_cnt = cro.run_rollout(
    cache_predict,
    env,
    max_episodes=1,
)

# Concatenate the cached activations so that batch dimension contains
# all the chached timesteps
activs = t.cat(activs_list, dim=0)
logits = t.cat(logits_list, dim=0)
print(activs.shape)


# %%
# Demonstrate probing by finding conv pixels that seem to predict a high
# probability of moving the paddle up

# Get action meanings from the environment
action_strs = env.envs[0].unwrapped.get_action_meanings()

# Get the action indices that correspond to moving the paddle up, which
# are RIGHT and RIGHTFIRE
up_action_idxs = [action_strs.index("RIGHT"), action_strs.index("RIGHTFIRE")]

# Convert the previously cached logits to probabilities
probs = t.nn.functional.softmax(logits, dim=-1)

# Calculate the sum of the probabilities of moving the paddle up
up_probs = probs[:, up_action_idxs].sum(dim=-1)

# Our probe objective variable is the a thresholded version
is_strong_up = (up_probs > 0.9).cpu().numpy()

# Now use CircRL to probe!
# We use a sparse probe to see if we can find a small set of conv pixels
# that predict strong paddle-up actions
SPARSE_NUMS = [1, 5, 10, 20]
probe_results = []
for sparse_num in SPARSE_NUMS:
    result = cpr.linear_probe(
        activs.cpu().numpy(),
        is_strong_up,
        sparse_method="f_test",
        sparse_num=sparse_num,
        random_state=0,
        C=1,
    )
    probe_results.append(
        {"sparse_num": sparse_num, "test_score": result["test_score"]}
    )
probe_results = pd.DataFrame(probe_results)

baseline = is_strong_up.mean()
baseline = max(baseline, 1 - baseline)

fig = px.line(
    probe_results,
    x="sparse_num",
    y="test_score",
    title="Probe test accuracy vs number of pixels probed",
    labels={"test_score": "Test accuracy", "sparse_num": "Number of pixels"},
    render_mode="svg",
)

# Add a labelled horizontal line for the baseline accuracy
fig.add_hline(
    y=baseline,
    line_dash="dash",
    line_color="black",
    annotation_text="Baseline",
)

fig.show()

# %%
# Demonstrate custom hook functions mean-ablating the top-K pixels
# found by the sparse probing to see how this affects the model's
# behavior

top_inds = t.tensor(result["sparse_inds"].copy()).to(activs.device)
mean_activs_flat = t.tensor(result["x"].mean(axis=0)).to(activs.device)


def hook_func(input, output):
    """Custom hook function to mean-ablate certain pixels from a conv
    layer."""
    # pylint: disable=unused-argument
    # Flatten the output, patch the specific indices with the mean
    # value, then return to the original shape
    output_flat = rearrange(output, "b c h w -> b (c h w)")
    output_flat[:, top_inds] = mean_activs_flat
    output = rearrange(
        output_flat,
        "b (c h w) -> b c h w",
        c=output.shape[1],
        h=output.shape[2],
    )
    return output


env = env_setup(0)[0]
with chk.HookManager(model.policy, hook={ACTIVS_LAYER: hook_func}) as _:
    # Run a single episode using CircRL
    seq, episode_return, step_cnt = cro.run_rollout(
        model.predict,
        env,
        max_episodes=1,
    )

# Display video
vid_fn, fps = cro.make_video_from_renders(seq.renders, fps=30.0)
display(Video(vid_fn, embed=True, width=300))
