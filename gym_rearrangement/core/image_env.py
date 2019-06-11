import os
import shutil
import sys
from datetime import datetime

import cv2
import h5py
import imageio
from PIL import Image
from gym.spaces import Box, Dict

from gym_rearrangement.core.goal_env import GoalEnv
from gym_rearrangement.core.wrapper_env import ProxyEnv
from vqa_utils import *

# Parameters for random object positions
TABLE_SIZE = 0.5 * 100
TABLE_CORNER = [105, 50]
TABLE_CENTER = [1.3, 0.75]  # Unit: meters


class ImageEnv(ProxyEnv, GoalEnv):
    """
    A wrapper used to retrieve image-based observations
    """

    def __init__(self,
                 wrapped_env,
                 img_size=84,
                 init_camera=None,
                 transform=True,
                 grayscale=False,
                 normalize=True,
                 reward_type='wrapped_env',
                 threshold=5,
                 recompute_reward=True,
                 save_img=False,
                 save_img_path=None,
                 default_camera='external_camera_0',
                 collect_data=False,
                 data_size=1000,
                 ):
        """

        :param wrapped_env:
        :param img_size:
        :param init_camera:
        :param transform: necessasry transform for image: flip or rotate
        :param grayscale:
        :param normalize:
        :param reward_type:
        :param threshold:
        :param recompute_reward:
        :param save_img: save image in a folder
        :param collect_data: collect vqa datasets on each step
        """
        self.quick_init(locals())  # locals() will return all the local variables in a dict
        super().__init__(wrapped_env)  # initialize parent class proxy env for serialize
        self.imsize = img_size
        self.init_camera = init_camera
        self.transform = transform
        self.grayscale = grayscale
        self.normalize = normalize
        self.recompute_reward = recompute_reward
        self.default_camera = default_camera
        self.collect_data = collect_data
        self.dataset_size = data_size

        self.channels = (1,) if grayscale else (3,)  # gray or RGB
        self.img_shape = (self.imsize, self.imsize)

        self.reward_type = reward_type
        self.threshold = threshold
        self.num_shape = self.wrapped_env.n_object
        self.data_cnt = 0  # number of samples when collection dataset

        # Init camera
        if init_camera is not None:
            # the initialize_camera func is defined in robot_env class
            sim = self.wrapped_env.init_camera(init_camera)

        # img_space = Box(0, 1, (self.img_len,), dtype=np.float32)
        # Using image observations, RGB image with (h, w, c)
        if self.normalize:
            img_space = Box(low=0, high=1, shape=self.img_shape + self.channels,
                            dtype=np.float32)
        else:
            img_space = Box(low=0, high=255, shape=self.img_shape + self.channels,
                            dtype=np.uint8)

        # questions and answer space  for each image
        q_space = Box(low=0, high=1, shape=(self.num_shape * NUM_Q, NUM_COLOR + NUM_Q),
                      dtype=np.bool)
        a_space = Box(low=0, high=1, shape=(self.num_shape * NUM_Q, NUM_COLOR + 4), dtype=np.bool)

        # Extended observation space
        spaces = self.wrapped_env.observation_space.spaces.copy()
        spaces['img_obs'] = img_space
        spaces['img_desired_goal'] = img_space
        spaces['img_achieved_goal'] = img_space
        spaces['question'] = q_space
        spaces['answer'] = a_space
        self.observation_space = Dict(spaces)
        self.action_space = self.wrapped_env.action_space

        self.save_img = save_img or collect_data  # save image when collect data
        if save_img_path is None:
            self.save_img_path = '/tmp/rearrange_dataset/'
        else:
            self.save_img_path = save_img_path

        # Clean up original files under the folder
        if self.save_img:
            if os.path.exists(self.save_img_path):
                shutil.rmtree(self.save_img_path)
            os.makedirs(self.save_img_path)

        if self.collect_data:
            # output files
            self.f = h5py.File(os.path.join(self.save_img_path, 'data.hy'), 'w')
            self.id_file = open(os.path.join(self.save_img_path, 'id.txt'), 'w')

        self._img_goal = self._sample_goal()  # sample an image goal

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        # collect data on each step
        if self.collect_data:
            self.generate_vqa_data(new_obs)
        if self.recompute_reward:
            reward = self.compute_rewards(new_obs)
        self._update_info(info, obs)
        return new_obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        return self._update_obs(obs)

    def render(self, *args, **kwargs):
        self.wrapped_env.render(*args, **kwargs)

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

        # state observation
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
        img_success = (img_dist < self.threshold).astype(float)
        info['dist'] = img_dist
        info['img_success'] = img_success

    def _get_img(self):
        img_obs = self.wrapped_env.get_image(
            width=self.imsize,
            height=self.imsize,
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
        # goal['desired_goal'] = self._img_goal
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
            return -(dist > self.threshold).astype(float)
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
        # Each row is a Q with the first NUM_COLOR represnts the color and the last 5 column stands for Q ID
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
            min_idx = idx[1]
            A[i * NUM_Q + 3, rep.color[min_idx]] = True
            # Q5: the color of the farthest object
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
            id = '{}'.format(self.data_cnt)
            self.id_file.write(id + '\n')
            grp = self.f.create_group(id)
            grp['image'] = I
            grp['question'] = Q[j, :]
            grp['answer'] = A[j, :]
            self.data_cnt += 1

            if self.data_cnt >= self.dataset_size:
                self.f.close()
                self.id_file.close()
                print('Dataset generated under {} with {} samples.'
                      .format(self.save_img_path, self.dataset_size))
                sys.exit(0)

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
            file_name = os.path.join(self.save_img_path, 'goal_img.png')
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
            if not os.listdir(self.save_img_path):
                raise ValueError('the target directory is empty')
            else:
                for img_file in os.listdir(self.save_img_path):
                    if img_file.endswith('png') and not img_file.startswith('goal'):
                        img = imageio.imread(os.path.join(self.save_img_path, img_file))
                        writer.append_data(img)
