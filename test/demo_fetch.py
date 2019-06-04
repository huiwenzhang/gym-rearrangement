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

env = gym.make("FetchPickAndPlaceDense-v2")

obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    print('reward: ', rew)
    env.render()
    if done:
        env.reset()
