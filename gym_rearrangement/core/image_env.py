import os
import shutil
from datetime import datetime

import cv2
import imageio
import numpy as np
from PIL import Image
from gym.spaces import Box, Dict

from gym_rearrangement.core.goal_env import GoalEnv
from gym_rearrangement.core.wrapper_env import ProxyEnv

# Parameters for random object positions
TABLE_SIZE = 0.5 * 100
TABLE_CORNER = [105, 50]
N_GRID = 3


class ImageEnv(ProxyEnv, GoalEnv):
    def __init__(self,
                 wrapped_env,
                 img_size=84,
                 init_camera=None,
                 transform=True,
                 grayscale=False,
                 normalize=False,
                 reward_type='wrapped_env',
                 img_threshold=5,
                 recompute_reward=True,
                 save_img=False,
                 img_path='/tmp/fetch/image',
                 default_camera='external_camera_0',
                 **_,
                 ):
        """
        A wrapper used to retrieve image-based observations
        :param wrapped_env: base vector state env
        :param img_size:
        :param init_camera:
        :param transform:
        :param grayscale:
        :param normalize:
        :param reward_type:
        :param img_threshold:
        :param recompute_reward:
        :param save_img:
        :param img_path:
        :param default_camera:
        """

        self.quick_init(locals())  # locals() will return all the local variables in a dict
        super().__init__(wrapped_env)  # initialize parent class proxy env for serialize
        self.episode_cnt = 0
        self.img_size = img_size
        self.init_camera = init_camera
        self.transform = transform
        self.grayscale = grayscale
        self.normalize = normalize
        self.recompute_reward = recompute_reward
        self.default_camera = default_camera

        self.channels = (1,) if grayscale else (3,)  # gray or RGB

        self.reward_type = reward_type
        self.img_threshold = img_threshold

        # Init camera
        if init_camera is not None:
            sim = self.wrapped_env.init_camera(init_camera)

        # Using image observations, RGB image with (h, w, c)
        if self.normalize:
            img_space = Box(low=0, high=1, shape=(img_size, img_size) + self.channels,
                            dtype=np.float32)
        else:
            img_space = Box(low=0, high=255, shape=(img_size, img_size) + self.channels,
                            dtype=np.uint8)

        # Extended observation space
        spaces = self.wrapped_env.observation_space.spaces.copy()
        spaces['img_obs'] = img_space
        spaces['img_desired_goal'] = img_space
        spaces['img_achieved_goal'] = img_space

        self.observation_space = Dict(spaces)
        self.action_space = self.wrapped_env.action_space

        self.save_img = save_img  # save image when collect data
        self.img_path = img_path
        # Clean up original files under the folder
        if self.save_img:
            if os.path.exists(self.img_path):
                shutil.rmtree(self.img_path)
            os.makedirs(self.img_path)

        self._img_goal = self._sample_goal()  # sample an image goal

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)

        if self.recompute_reward:
            reward = self.compute_rewards(new_obs)
        self._update_info(info, obs)
        return new_obs, reward, done, info

    def reset(self):
        self.episode_cnt += 1
        obs = self.wrapped_env.reset()
        return self._update_obs(obs)

    def render(self, *args, **kwargs):
        self.wrapped_env.render(*args, **kwargs)

    def _update_obs(self, obs):
        img_obs = self._get_img()
        obs['img_obs'] = img_obs
        obs['img_desired_goal'] = self._img_goal
        obs['img_achieved_goal'] = img_obs
        return obs

    def _update_info(self, info, obs):
        img_achieved_goal = obs['img_achieved_goal']
        img_desired_goal = self._img_goal
        # compute normalized distance
        if not self.normalize:
            img_achieved_goal = self.normalize_img(img_achieved_goal)
            img_desired_goal = self.normalize_img(img_desired_goal)
        # TODO: compute distance in pixel space, is it reasonable?
        img_dist = np.linalg.norm(img_achieved_goal - img_desired_goal)
        img_success = (img_dist < self.img_threshold).astype(float)
        info['dist'] = img_dist
        info['img_success'] = img_success

    def _get_img(self):
        img_obs = self.wrapped_env.get_image(
            width=self.img_size,
            height=self.img_size,
            camera_name=self.default_camera
        )
        if self.grayscale:
            img_obs = Image.fromarray(img_obs).convert('L')
            img_obs = np.array(img_obs)
        if self.transform:
            img_obs = Image.fromarray(img_obs).rotate(180)
            img_obs = np.array(img_obs)
        if self.normalize:
            img_obs = img_obs / 255.
        return img_obs

    # Goal env methods and abstract method implementations
    def get_goal(self):
        goal = self.wrapped_env.get_goal()
        goal['img_desired_goal'] = self._img_goal  # only update img goal
        return goal

    def set_goal(self, goal):
        self._img_goal = goal['img_desired_goal']

    def compute_rewards(self, obs):
        """
        image distance, we can also use state distance by setting reward_type = wrapped_env
        :param obs:
        :return:
        """
        achieved_goal = obs['img_achieved_goal']
        desired_goal = obs['img_desired_goal']
        # compute normalized distance
        if not self.normalize:
            achieved_goal = self.normalize_img(achieved_goal)
            desired_goal = self.normalize_img(desired_goal)
        dist = np.linalg.norm(achieved_goal - desired_goal)
        if self.reward_type == 'img_distance':
            return -dist
        elif self.reward_type == 'img_sparse':
            return -(dist > self.img_threshold).astype(float)
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(obs)
        elif self.reward_type == 'img_hidden_space':  # compute distance in hidden space with encoders
            raise NotImplementedError
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
                [(x + 0.5) * grid_size + x ** 2, (y + 0.5) * grid_size + y ** 2]) + np.array(
                TABLE_CORNER)
            object_xpos = object_xpos / 100.

            object_joint_name = 'object{}:joint'.format(i)
            object_qpos = self.wrapped_env.sim.data.get_joint_qpos(object_joint_name)
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.wrapped_env.sim.data.set_joint_qpos(object_joint_name, object_qpos)
        self.wrapped_env.sim.forward()

        img = self._get_img()
        goal_img = self.recover_img(img)

        if self.save_img:
            file_name = os.path.join(self.img_path, 'goal_img_episode_{:3d}.png'.format(self.episode_cnt))
            im = Image.fromarray(goal_img)
            im.save(file_name)
        return img

    def cv_render(self, camera_name=None):
        if camera_name is None:
            camera_name = self.default_camera
        img_obs = self.wrapped_env.get_image(
            width=256,
            height=256,
            camera_name=camera_name
        )
        if camera_name == self.default_camera:
            img_obs = np.flip(img_obs, 0)
        elif camera_name == 'table_camera':
            img_obs = np.rot90(img_obs)
        else:
            print('Undefined camera')
        cv2.imshow('robot view', img_obs)
        cv2.waitKey(1)

    def recover_img(self, img):
        """
        recover image from normalized vector
        :param img:
        :return:
        """
        if self.normalize:
            return self.unnormalize_img(img)
        else:
            return img

    def make_gif(self, path=None):
        if path is None:
            path = '/tmp/arrange_learning/'
        if not os.path.isdir(path):
            print('A directory should be given')
            return
        path = os.path.join(path, 'fetch-{}.gif'.format(datetime.now().strftime('%Y-%m-%d-%H-%M')))
        with imageio.get_writer(path, mode='I') as writer:
            if not os.listdir(self.img_path):
                raise ValueError('the target directory is empty')
            else:
                for img_file in os.listdir(self.img_path):
                    if img_file.endswith('png') and not img_file.startswith('goal'):
                        img = imageio.imread(os.path.join(self.img_path, img_file))
                        writer.append_data(img)
