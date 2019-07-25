"""
Test of the fetch environment
"""

import gym

env = gym.make("FetchPickAndPlaceDense-v2")

obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    print('reward: ', rew)
    env.render()
    if done:
        env.reset()
