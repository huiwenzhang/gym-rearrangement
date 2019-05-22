import random

import cv2
import numpy as np
from PIL import Image
from gym.spaces import Box, Dict

from gym_rearrangement.core.goal_env import GoalEnv
from gym_rearrangement.core.wrapper_env import ProxyEnv
from gym_rearrangement.envs.env_util import concatenate_box_spaces
from gym_rearrangement.envs.env_util import get_stat_in_paths, create_stats_ordered_dict


class ImageEnv(ProxyEnv, GoalEnv):
    """
    A wrapper used to retrieve image based observations
    """

    def __init__(self,
                 wrapped_env,
                 img_size=84,
                 init_camera=None,
                 transpose=False,
                 grayscale=False,
                 normalize=False,
                 reward_type='wrapped_env',
                 threshold=10,
                 img_len=None,
                 presampled_goals=None,
                 recompute_reward=True
                 ):
        """

        :param wrapped_env:
        :param img_size:
        :param init_camera:
        :param transpose:
        :param grayscale:
        :param normalize:
        :param reward_type:
        :param threshold:
        :param img_len:
        :param presampled_goals:
        :param recompute_reward:
        """
        self.quick_init(locals())  # locals() will return all the local variables in a dict
        super().__init__(wrapped_env)  # initialize parent class proxy env for serialize
        self.imsize = img_size
        self.init_camera = init_camera
        self.transpose = transpose
        self.grayscale = grayscale
        self.normalize = normalize
        self.recompute_reward = recompute_reward

        self._render_local = False

        if img_len is not None:
            self.img_len = img_len
        else:
            if grayscale:
                self.img_len = self.imsize ** 2
            else:
                self.img_len = 3 * self.imsize ** 2
        self.channels = 1 if grayscale else 3  # gray or RGB
        self.img_shape = (self.imsize, self.imsize)

        # Init camera
        if init_camera is not None:
            # the initialize_camera func is defined in robot_env class
            sim = self._wrapped_env.initialize_camera(init_camera)

        img_space = Box(0, 1, (self.img_len,), dtype=np.float32)
        # TODO: given a image goal
        self._img_goal = img_space.sample()  # sample an image goal

        # Extended observation space
        spaces = self.wrapped_env.observation_space.spaces.copy()
        spaces['img_obs'] = img_space
        spaces['img_desired_goal'] = img_space
        spaces['img_achieved_goal'] = img_space

        self.return_img_proprio = False
        if 'proprio_observation' in spaces.keys():
            self.return_img_proprio = True
            spaces['img_proprio_obs'] = concatenate_box_spaces(
                spaces['img_obs'],
                spaces['proprio_obs']
            )
            spaces['img_proprio_desired_goal'] = concatenate_box_spaces(
                spaces['img_desired_goal'],
                spaces['proprio_desired_goal']
            )
            spaces['img_proprio_achieved_goal'] = concatenate_box_spaces(
                spaces['img_achieved_goal'],
                spaces['proprio_achieved_goal']
            )

            self.observation_space = Dict(spaces)
            self.action_space = self.wrapped_env.action_space
            self.reward_type = reward_type
            self.threshold = threshold
            self._presampled_goals = presampled_goals
            if self._presampled_goals is None:
                self.num_goals_presampled = 0
            else:
                self.num_goals_presampled = \
                    presampled_goals[random.choice(list(presampled_goals))].shape[0]

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        if self.recompute_reward:
            reward = self.compute_reward(action, new_obs)
        self._update_info(info, obs)
        return new_obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        return self._update_obs(obs)

    def render(self, *args, **kwargs):
        self.wrapped_env.render(*args, **kwargs)

    def enable_render(self):
        self._render_local = True

    def _update_obs(self, obs):
        img_obs = self._get_flat_img()
        obs['img_obs'] = img_obs
        obs['img_desired_goal'] = self._img_goal
        obs['img_achieved_goal'] = img_obs

        # state observation

        if self.return_img_proprio:
            obs['img_proprio_obs'] = np.concatenate(
                (obs['img_obs'], obs['proprio_observation'])
            )
            obs['img_proprio_desired_goal'] = np.concatenate(
                (obs['img_desired_goal'], obs['proprio_desired_goal'])
            )
            obs['img_proprio_achieved_goal'] = np.concatenate(
                (obs['img_achieved_goal'], obs['proprio_achieved_goal'])
            )

        return obs

    def _update_info(self, info, obs):
        achieved_goal = obs['img_achieved_goal']
        desired_goal = self._img_goal
        img_dist = np.linalg.norm(achieved_goal - desired_goal)
        img_success = (img_dist < self.threshold).astype(float)
        info['dist'] = img_dist
        info['img_success'] = img_success

    def _get_flat_img(self):
        # flatten the image to a vector
        img_obs = self._wrapped_env.get_image(
            width=self.imsize,
            height=self.imsize,
        )
        if self._render_local:
            cv2.imshow('env visual', img_obs)
            cv2.waitKey(1)
        if self.grayscale:
            img_obs = Image.fromarray(img_obs).convert('L')
            img_obs = np.array(img_obs)
        if self.normalize:
            img_obs = img_obs / 255.
        if self.transpose:
            img_obs = img_obs.transpose()
        return img_obs.flatten()

    # Goal env methods and abstract method implementations
    def get_goal(self):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._img_goal
        goal['img_desired_goal'] = self._img_goal
        return goal

    def set_goal(self, goal):
        self._img_goal = goal['img_desired_goal']

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goal = obs['desired_goal']
        dist = np.linalg.norm(achieved_goals - desired_goal)
        if self.reward_type == 'img_distance':
            return -dist
        elif self.reward_type == 'img_sparse':
            return -(dist > self.threshold).astype(float)
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    @staticmethod
    def normalize_img(image):
        assert image.dtype == np.uint8
        return np.float64(image) / 255.

    @staticmethod
    def unnormalize_img(image):
        assert image.dtype != np.uint8
        return np.unit8(image * 255.)
