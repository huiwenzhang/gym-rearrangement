"""
Test of the maze environment
"""

import gym
import gym_rearrangement

# Initialize the "maze" environment
env = gym.make("maze-random-20x20-plus-v0")


obs = env.reset()
for _ in range(1500):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    if done:
        env.reset()