"""
Test of the fetch environment
"""

# Optional fetch env id:
# - FetchReach-v1
# - FetchSlide-v1
# - FetchPush-v1
# - FetchPickAndPlace-v1

import gym
import gym_rearrangement

# Initialize the "maze" environment
env = gym.make("FetchRearrangement4Dense-v1")

obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    print(rew)
    env.render()
    if done:
        env.reset()
