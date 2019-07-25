"""
Test of the fetch environment
"""

import os

import gym
from PIL import Image

from gym_rearrangement.core.image_env import ImageEnv

# Initialize the "rearrangement" environment
env = gym.make("FetchRearrangement3-v1")
env = ImageEnv(env, reward_type='img_distance', save_img=True, img_size=128)

obs = env.reset()
for i in range(500):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)

    # reconstruct image
    im = obs['img_obs']
    im = env.recover_img(im)
    file_name = os.path.join(env.img_path, '{:0>3d}.png'.format(i))
    im = Image.fromarray(im)
    im.save(file_name)

    env.cv_render('external_camera_0')  # render on screen with opencv

    if done:
        env.reset()
