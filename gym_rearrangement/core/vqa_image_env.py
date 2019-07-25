import os
import shutil
import sys

import h5py
from gym.spaces import Box, Dict

from vqa_utils import *
from .image_env import ImageEnv

TABLE_SIZE = 0.5 * 100
TABLE_CORNER = [105, 50]
TABLE_CENTER = [1.3, 0.75]  # Unit: meter


class FetchVQAImageEnv(ImageEnv):
    """
    A wrapper used to retrieve image-based observations
    """

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
                 data_size=1000,
                 vqa_data_path='/tmp/fetch/vqa_dataset',
                 **_
                 ):
        """
        Used for generate visual question answer dataset
        :param data_size: default size of data
        :param vqa_data_path: path to save data
        :param _:
        """
        self.quick_init(locals())

        super(FetchVQAImageEnv, self).__init__(wrapped_env, img_size, init_camera=init_camera,
                                               transform=transform,
                                               grayscale=grayscale, normalize=normalize,
                                               reward_type=reward_type,
                                               img_threshold=img_threshold, recompute_reward=recompute_reward,
                                               save_img=save_img,
                                               img_path=img_path, default_camera=default_camera,
                                               )
        self.num_shape = self.wrapped_env.n_object
        self.data_size = data_size
        # questions and answer space  for each image
        q_space = Box(low=0, high=1, shape=(self.num_shape * NUM_Q, NUM_COLOR + NUM_Q),
                      dtype=np.bool)
        a_space = Box(low=0, high=1, shape=(self.num_shape * NUM_Q, NUM_COLOR + 4), dtype=np.bool)

        spaces = self.observation_space.spaces.copy()
        spaces['question'] = q_space
        spaces['answer'] = a_space
        self.observation_space = Dict(spaces)
        self.action_space = self.wrapped_env.action_space

        self.vqa_data_path = vqa_data_path
        if os.path.exists(self.vqa_data_path):
            shutil.rmtree(self.vqa_data_path)
        os.makedirs(self.vqa_data_path)

        # output files
        self.f = h5py.File(os.path.join(self.vqa_data_path, 'data.hy'), 'w')
        self.id_file = open(os.path.join(self.vqa_data_path, 'idx.txt'), 'w')
        self.data_cnt = 0

    # vqa methods
    def generate_image(self, img):
        img = self.recover_img(img)
        rep = Representation(np.stack(self.wrapped_env.X).astype(np.float),
                             np.stack(self.wrapped_env.Y).astype(np.float),
                             self.wrapped_env.color, self.wrapped_env.shape)
        return np.array(img), rep

    def generate_questions(self, rep):
        # Generate questions: [# of shape * # of Q, # of color + # of Q]
        # Ask 5 questions for each type of color objects
        # Each row is a Q with the first NUM_COLOR represnts the color and the last 5 column stands for Q idx
        Q = np.zeros((self.num_shape * NUM_Q, NUM_COLOR + NUM_Q), dtype=np.bool)  # object - q
        for i in range(self.num_shape):
            v = np.zeros(NUM_COLOR)
            v[rep.color[i]] = True
            Q[i * NUM_Q:(i + 1) * NUM_Q, :NUM_COLOR] = np.tile(v, (NUM_Q, 1))
            Q[i * NUM_Q:(i + 1) * NUM_Q, NUM_COLOR:] = np.diag(np.ones(NUM_Q))
        return Q

    def generate_answer(self, rep):
        # Generate answers: [# of shape * # of Q, # of color + 4]
        # # of color + 4: [color 1, color 2, ... , circle, rectangle, yes, no]
        A = np.zeros((self.num_shape * NUM_Q, NUM_COLOR + 4), dtype=np.bool)
        for i in range(self.num_shape):
            # Q1: circle or rectangle?
            if rep.shape[i]:
                A[i * NUM_Q, NUM_COLOR] = True
            else:
                A[i * NUM_Q, NUM_COLOR + 1] = True

            # Q2: bottom?
            if rep.x[i] > (TABLE_CENTER[0] + 0.1):
                A[i * NUM_Q + 1, NUM_COLOR + 2] = True
            else:
                A[i * NUM_Q + 1, NUM_COLOR + 3] = True

            # Q3: left?
            if rep.y[i] > (TABLE_CENTER[1]):
                A[i * NUM_Q + 2, NUM_COLOR + 2] = True
            else:
                A[i * NUM_Q + 2, NUM_COLOR + 3] = True

            distance = 1.1 * (rep.y - rep.y[i]) ** 2 + (rep.x - rep.x[i]) ** 2
            idx = distance.argsort()
            # Q4: the color of the nearest object except for itself
            if self.num_shape == 1:
                min_idx = 0
            else:
                min_idx = idx[1]
            A[i * NUM_Q + 3, rep.color[min_idx]] = True
            # Q5: the color of the farthest object
            if self.num_shape == 1:
                max_idx = 0
            else:
                max_idx = idx[-1]
            A[i * NUM_Q + 4, rep.color[max_idx]] = True
        return A

    # Generate vqa dataset
    def generate_vqa_data(self, obs):
        img = obs['img_obs']
        I, R = self.generate_image(img)
        A = self.generate_answer(R)
        Q = self.generate_questions(R)
        for j in range(self.num_shape * NUM_Q):
            idx = '{}'.format(self.data_cnt)
            self.id_file.write(idx + '\n')
            grp = self.f.create_group(idx)
            grp['image'] = I
            grp['question'] = Q[j, :]
            grp['answer'] = A[j, :]
            self.data_cnt += 1

            if self.data_cnt >= self.data_size:
                self.f.close()
                self.id_file.close()
                print('Dataset generated under {} with {} samples.'
                      .format(self.vqa_data_path, self.data_size))
                sys.exit(0)

    # update all kinds of raw observations
    def _update_obs(self, obs):
        img_obs = self._get_img()
        _, R = self.generate_image(img_obs)
        Q = self.generate_questions(R)
        A = self.generate_answer(R)
        obs['img_obs'] = img_obs
        obs['img_desired_goal'] = self._img_goal
        obs['img_achieved_goal'] = img_obs
        obs['question'] = Q
        obs['answer'] = A
        return obs

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)

        # collect data on each step
        self.generate_vqa_data(new_obs)

        if self.recompute_reward:
            reward = self.compute_rewards(new_obs)
        self._update_info(info, obs)

        # only image observation is returned
        return new_obs, reward, done, info
