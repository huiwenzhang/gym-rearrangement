import random
import os
import h5py
import cv2
import sys
import shutil
import numpy as np
from PIL import Image
import imageio
from gym.spaces import Box, Dict

from gym_rearrangement.core.goal_env import GoalEnv
from gym_rearrangement.core.wrapper_env import ProxyEnv
from collections import deque
import numpy as np
from gym import spaces
import cv2

# Parameters for random object positions
TABLE_SIZE = 0.5 * 100
TABLE_CORNER = [105, 50]
TABLE_CENTER = [1.3, 0.75]  # Unit: meters


class FrameStack(ProxyEnv, GoalEnv):
    """
    Frame stacking wrapper for image environment
    """

    def __init__(self,
                 wrapped_env,
                 n_frames
                 ):
        """

        :param wrapped_env:
        :param n_frames: number of stacked frames
        """
        self.quick_init(locals())  # locals() will return all the local variables in a dict
        super().__init__(wrapped_env)  # initialize parent class proxy env for serialize
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shp = wrapped_env.observation_space.spaces['img_obs'].shape
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(shp[0], shp[1], shp[2] * n_frames),
                                            dtype=wrapped_env.observation_space.spaces[
                                                'img_obs'].dtype)
        self.action_space = self.wrapped_env.action_space

    def reset(self):
        obs = self.wrapped_env.reset()
        img = obs['img_obs']
        for _ in range(self.n_frames):
            self.frames.append(img)
        return self._update_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        self.frames.append(obs['img_obs'])
        return self._update_obs(obs), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.n_frames
        return LazyFrames(list(self.frames))

    def _update_obs(self, obs):
        img_stack_obs = self._get_obs()
        obs['img_obs'] = img_stack_obs
        # state observation
        return obs


class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to np.ndarray before being passed to the model.

        :param frames: ([int] or [float]) environment frames
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
