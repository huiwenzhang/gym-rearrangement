"""
Test of the fetch environment
"""

# Optional fetch env id:
# - FetchReach-v1
# - FetchSlide-v1
# - FetchPush-v1
# - FetchPickAndPlace-v1

import gym
import cv2
import os
import gym_rearrangement
from gym_rearrangement.core.image_env import ImageEnv

# from gym_rearrangement.envs.robotics.cameras_setup import *

train_img_path = '/tmp/rearrange/image/train'
if not os.path.exists(train_img_path):
    os.makedirs(train_img_path)

# Initialize the "maze" environment
env = gym.make("Rearrangement3-v1")
# env = ImageEnv(env, reward_type='img_distance', save_img=True, init_camera=init_sawyer_camera_v1)
env = ImageEnv(env, reward_type='img_distance', save_img=True, img_size=128)

obs = env.reset()
print(obs)
for i in range(500):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)

    # reconstruct image
    im = obs['img_obs']
    im = env.recover_img(im)
    file_name = os.path.join(train_img_path, '{:0>3d}.png'.format(i))
    cv2.imwrite(file_name, im)

    env.cv_render('external_camera_0')

    if done:
        env.reset()
