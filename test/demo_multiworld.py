"""
Test env and functions defined in multiworld repo
"""

# Optional fetch env id:
# - FetchReach-v1
# - FetchSlide-v1
# - FetchPush-v1
# - FetchPickAndPlace-v1

import gym
import multiworld.envs.mujoco
env = gym.make('Image48SawyerReachXYEnv-v1')


obs = env.reset()
for _ in range(2000):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    if done:
        env.reset()