"""
Test of the fetch environment
"""

import gym
from gym_rearrangement.core.flat_env import FlatGoalEnv
from gym_rearrangement.core.frame_stack import FrameStack
# import gym_rearrangement
from gym_rearrangement.core.image_env import ImageEnv

# Initialize the "rearrangement" environment
env = gym.make("FetchPickAndPlaceDense-v2")
print(env.observation_space.spaces)
env = ImageEnv(env, reward_type='wrapped_env', img_size=128)
env = FrameStack(env, n_frames=4)
print(env.observation_space.spaces)
env = FlatGoalEnv(env, obs_keys=['img_obs'])

print(env.observation_space.shape)

obs = env.reset()
for i in range(500):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.cv_render()  # render on screen with opencv
    if done:
        env.reset()
