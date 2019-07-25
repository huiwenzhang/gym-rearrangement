"""
Test of the fetch environment
"""

import gym
import gym_rearrangement
from pprint import pprint

env = gym.make("FetchRearrangement1Dense-v1")

obs = env.reset()
print(obs['observation'].shape)
for _ in range(2000):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    print(rew)
    print('*' * 60)
    pprint(obs)
    env.render()
    if done:
        env.reset()
