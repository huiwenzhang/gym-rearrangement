"""
Collect vqa dataset in fetch rearrangement task
"""

import os

import gym
from PIL import Image

from gym_rearrangement.core.vqa_image_env import FetchVQAImageEnv

env = gym.make("FetchRearrangement6-v1")
env = FetchVQAImageEnv(env, reward_type='img_distance', save_img=True, img_size=224,
                       data_size=2000)

# To sample dataset, we need reset env every step, so set the max_epiosde_steps to be 1
seed = 10  # for random sample action when collect data
obs = env.reset()
env.action_space.seed(seed)
for i in range(2000):
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)

    # reconstruct image
    im = obs['img_obs']
    im = env.recover_img(im)
    file_name = os.path.join(env.vqa_data_path, '{:0>3d}.png'.format(i))
    im = Image.fromarray(im)
    im.save(file_name)
    env.cv_render('external_camera_0')

    if done:
        seed += 10
        env.action_space.seed(seed)
        env.reset()
