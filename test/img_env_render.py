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
from gym_rearrangement.envs.robotics.cameras_setup import *
import cv2

# Initialize the "maze" environment
env = gym.make("Rearrangement3-v1")
# env = ImageEnv(env, reward_type='img_distance', save_img=True, init_camera=init_sawyer_camera_v1)
env = ImageEnv(env, reward_type='img_distance', img_size=128)

obs = env.reset()
print(obs)
for i in range(500):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    env.render()
    if done:
        env.reset()
