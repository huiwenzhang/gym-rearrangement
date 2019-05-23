import random
import os

import cv2
import numpy as np
from PIL import Image
from gym.spaces import Box, Dict

from gym_rearrangement.core.goal_env import GoalEnv
from gym_rearrangement.core.wrapper_env import ProxyEnv
from gym_rearrangement.envs.env_util import concatenate_box_spaces
from gym_rearrangement.envs.env_util import get_stat_in_paths, create_stats_ordered_dict

# Parameters for random object positions
N_GRID = 3
TABLE_SIZE = 0.5 * 100
TABLE_CENTER = [107, 50]


class ImageEnv(ProxyEnv, GoalEnv):
    """
    A wrapper used to retrieve image based observations
    """

    def __init__(self,
                 wrapped_env,
                 img_size=84,
                 init_camera=None,
                 transform=True,
                 grayscale=False,
                 normalize=True,
                 reward_type='wrapped_env',
                 threshold=10,
                 img_len=None,
                 presampled_goals=None,
                 recompute_reward=True,
                 save_img=False,
                 save_img_path=None,
                 default_camera='external_camera_0'
                 ):
        """

        :param wrapped_env:
        :param img_size:
        :param init_camera:
        :param tansform: necessasry transform for image: flip or rotate
        :param grayscale:
        :param normalize:
        :param reward_type:
        :param threshold:
        :param img_len:
        :param presampled_goals:
        :param recompute_reward:
        :param save_img: save image in a folder
        """
        self.quick_init(locals())  # locals() will return all the local variables in a dict
        super().__init__(wrapped_env)  # initialize parent class proxy env for serialize
        self.imsize = img_size
        self.init_camera = init_camera
        self.transform = transform
        self.grayscale = grayscale
        self.normalize = normalize
        self.recompute_reward = recompute_reward
        # self.wrapped_env = wrapped_env
        self.default_camera = default_camera

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
            sim = self.wrapped_env.init_camera(init_camera)

        img_space = Box(0, 1, (self.img_len,), dtype=np.float32)

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

        self.save_img = save_img
        if save_img_path is None:
            self.save_img_path = '/tmp/rearrange/image/'
        else:
            self.save_img_path = save_img_path

        # TODO: given a image goal
        self._img_goal = self._sample_goal()  # sample an image goal

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        print(new_obs)
        if self.recompute_reward:
            reward = self.compute_reward(new_obs)
        self._update_info(info, obs)
        return new_obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        return self._update_obs(obs)

    def render(self, *args, **kwargs):
        self.wrapped_env.render(*args, **kwargs)

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
        img_obs = self.wrapped_env.get_image(
            width=self.imsize,
            height=self.imsize,
            camera_name=self.default_camera
        )
        if self.grayscale:
            img_obs = Image.fromarray(img_obs).convert('L')
            img_obs = np.array(img_obs)
        if self.transform:
            img_obs = np.flip(img_obs)
        if self.normalize:
            img_obs = img_obs / 255.
        return img_obs.flatten()

    # Goal env methods and abstract method implementations
    def get_goal(self):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._img_goal
        goal['img_desired_goal'] = self._img_goal
        return goal

    def set_goal(self, goal):
        self._img_goal = goal['img_desired_goal']

    def compute_reward(self, obs):
        achieved_goals = obs['img_achieved_goal']
        desired_goal = obs['img_desired_goal']
        dist = np.linalg.norm(achieved_goals - desired_goal)
        if self.reward_type == 'img_distance':
            return -dist
        elif self.reward_type == 'img_sparse':
            return -(dist > self.threshold).astype(float)
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_reward(obs)
        else:
            raise NotImplementedError

    @staticmethod
    def normalize_img(image):
        assert image.dtype == np.uint8
        return np.float64(image) / 255.

    @staticmethod
    def unnormalize_img(image):
        assert image.dtype != np.uint8
        return (image * 255.).astype(np.uint8)

    # Goal env methods
    def _sample_goal(self):
        """
        sample a goal image
        :return:
        """
        # Randomize start position of all objects.
        grid_size = int(TABLE_SIZE * 0.9 / N_GRID)

        idx_coor = np.arange(N_GRID * N_GRID)
        np.random.shuffle(idx_coor)

        for i in range(self.wrapped_env.n_object):
            # block index
            x = idx_coor[i] % N_GRID
            y = (N_GRID - np.floor(idx_coor[i] / N_GRID) - 1).astype(np.uint8)
            # block coordinates
            object_xpos = np.array(
                [(x + 0.5) * grid_size, (y + 0.5) * grid_size]) + np.array(
                TABLE_CENTER)
            object_xpos = object_xpos / 100.

            object_joint_name = 'object{}:joint'.format(i)
            object_qpos = self.wrapped_env.sim.data.get_joint_qpos(object_joint_name)
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.wrapped_env.sim.data.set_joint_qpos(object_joint_name, object_qpos)
        self.wrapped_env.sim.forward()

        flat_img = self._get_flat_img()
        goal_img = self.recover_img(flat_img)

        if self.save_img:
            if not os.path.exists(self.save_img_path):
                os.makedirs(self.save_img_path)
            file_name = os.path.join(self.save_img_path, 'goal_img.png')
            cv2.imwrite(file_name, goal_img)
            cv2.waitKey(1)
        return flat_img

    def cv_render(self, camera_name=None):
        img_obs = self.wrapped_env.get_image(
            width=200,
            height=200,
            camera_name=camera_name
        )
        if camera_name == self.default_camera:
            img_obs = np.flip(img_obs, 0)
        elif camera_name == 'table_camera':
            img_obs = np.rot90(img_obs)
        else:
            print('Unsupported camera')
        cv2.imshow('robot view', img_obs)
        cv2.waitKey(1)

    def recover_img(self, img):
        """
        recover image from flat normalized vector
        :param img:
        :return:
        """
        if self.normalize:
            return self.unnormalize_img(img).reshape(self.imsize, self.imsize, -1)
        else:
            return img.reshape(self.imsize, self.imsize, -1)
