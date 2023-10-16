import numpy as np
import pytest

from circrl.rollouts import run_rollout, RlStepSeq


@pytest.fixture
def env():
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack

    env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=42)
    env = VecFrameStack(env, n_stack=4)
    return env


@pytest.fixture
def predict_func():
    def predict(obs, deterministic):
        return [0], None

    return predict


@pytest.fixture
def custom_data_funcs():
    def custom_data(env, render, obs, action, model_states, predict_extra):
        return np.random.rand(3)

    return {"custom_data": custom_data}


def test_generate_rollouts(env, predict_func, custom_data_funcs):
    seq_data, episode_returns, step_cnt = run_rollout(
        predict_func,
        env,
        custom_data_funcs=custom_data_funcs,
        max_steps=100,
        max_episodes=2,
        show_pbar=False,
        seed=42,
    )

    assert isinstance(seq_data, RlStepSeq)
    assert isinstance(episode_returns, list)
    assert isinstance(step_cnt, int)
