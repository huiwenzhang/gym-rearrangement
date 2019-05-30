"""
Collect vqa dataset in fetch rearrangement task
"""

import gym
import cv2
import os
import gym_rearrangement
from gym_rearrangement.core.image_env import ImageEnv

env = gym.make("FetchRearrangement6-v1")
env = ImageEnv(env, reward_type='img_distance', save_img=True, img_size=128, collect_data=True,
               data_size=50000)

# To sample dataset, we need reset env every step, so set the max_epiosde_steps to be 1

obs = env.reset()
for i in range(2000):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)

    # reconstruct image
    im = obs['img_obs']
    im = env.recover_img(im)
    file_name = os.path.join(env.save_img_path, '{:0>3d}.png'.format(i))
    cv2.imwrite(file_name, im)
    env.cv_render('external_camera_0')

    if done:
        env.reset()
