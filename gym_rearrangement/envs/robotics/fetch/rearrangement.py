import os
from gym import utils
from gym_rearrangement.envs.robotics import fetch_env
import numpy as np

# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = os.path.join('fetch', 'rearrange_4.xml')

# Parameters for random object positions
N_GRID = 3
TABLE_SIZE = 0.5 * 100
TABLE_CENTER = [107, 50]


class RearrangeFour(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        print("test in FetchPickAndPlaceEnv")
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        model_xml_path = os.path.join('fetch', 'rearrange_4.xml')
        fetch_env.FetchEnv.__init__(
            self, model_xml_path, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        # Number of objects
        self.n_object = 4

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of all objects.
        grid_size = int(TABLE_SIZE * 0.9 / N_GRID)

        idx_coor = np.arange(N_GRID * N_GRID)
        np.random.shuffle(idx_coor)

        for i in range(self.n_object):
            # block index
            x = idx_coor[i] % N_GRID
            y = (N_GRID - np.floor(idx_coor[i] / N_GRID) - 1).astype(np.uint8)
            # block coordinates
            object_xpos = np.array(
                [(x + 0.5) * grid_size, (y + 0.5) * grid_size]) + np.array(
                TABLE_CENTER)
            object_xpos = object_xpos / 100.
            object_joint_name = 'object{}:joint'.format(i)
            if self.has_object:
                object_qpos = self.sim.data.get_joint_qpos(object_joint_name)
                assert object_qpos.shape == (7,)
                object_qpos[:2] = object_xpos
                self.sim.data.set_joint_qpos(object_joint_name, object_qpos)

        self.sim.forward()
        return True


class RearrangeSix(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        print("test in FetchPickAndPlaceEnv")
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        model_xml_path = os.path.join('fetch', 'rearrange_6.xml')
        fetch_env.FetchEnv.__init__(
            self, model_xml_path, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
