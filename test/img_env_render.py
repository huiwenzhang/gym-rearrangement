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
from gym_rearrangement.core.image_env import ImageEnv

# Initialize the "rearrangement" environment
env = gym.make("FetchRearrangement3-v1")
env = ImageEnv(env, reward_type='img_distance', img_size=128)

obs = env.reset()
print(obs)
for i in range(500):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.cv_render()  # render on screen with opencv
    if done:
        env.reset()
