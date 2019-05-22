import os
from gym_rearrangement.envs.robotics import fetch_env, utils, rotations, cameras_setup
import numpy as np
from gym.utils import EzPickle

# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = os.path.join('fetch', 'rearrange_4.xml')

# Parameters for random object positions
N_GRID = 3
TABLE_SIZE = 0.5 * 100
TABLE_CENTER = [107, 50]


class Rearrangement(fetch_env.FetchEnv, EzPickle):
    def __init__(self, reward_type='sparse', n_object=4):
        print("test in FetchPickAndPlaceEnv")
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self.n_object = n_object
        model_xml_path = os.path.join('fetch', 'rearrange_{}.xml'.format(self.n_object))
        fetch_env.FetchEnv.__init__(
            self, model_xml_path, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.1,
            initial_qpos=initial_qpos, reward_type=reward_type, fix_goal=False)
        EzPickle.__init__(self)

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
            object_qpos = self.sim.data.get_joint_qpos(object_joint_name)
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos(object_joint_name, object_qpos)

        self.sim.forward()
        return True

    def _get_obs(self):
        # gripper state
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.dt
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        # robot state
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        # object state
        object_pos, object_rot, object_velp, object_velr = [], [], [], []
        if self.has_object:
            for i in range(self.n_object):
                obj_name = 'object{}'.format(i)
                object_pos.append(self.sim.data.get_site_xpos(obj_name))
                # rotations
                object_rot.append(
                    rotations.mat2euler(self.sim.data.get_site_xmat(obj_name)))
                # velocities
                object_velp.append(self.sim.data.get_site_xvelp(obj_name) * dt)
                object_velr.append(self.sim.data.get_site_xvelr(obj_name) * dt)
            object_pos = np.array(object_pos)
            object_rot = np.array(object_rot)
            object_velp = np.array(object_velp)
            object_velr = np.array(object_velr)
        else:
            object_pos = object_rot = object_velp = object_velr = np.zeros(0)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt

        achieved_goal = np.squeeze(object_pos.flatten())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _sample_goal(self):
        # Randomize start position of all objects.
        grid_size = int(TABLE_SIZE * 0.9 / N_GRID)

        idx_coor = np.arange(N_GRID * N_GRID)
        np.random.shuffle(idx_coor)

        goal_pos = []
        for i in range(self.n_object):
            # grid index
            x = idx_coor[i] % N_GRID
            y = (N_GRID - np.floor(idx_coor[i] / N_GRID) - 1).astype(np.uint8)
            # grid coordinates
            object_xpos = np.array(
                [(x + 0.5) * grid_size, (y + 0.5) * grid_size]) + np.array(
                TABLE_CENTER)
            object_xpos = object_xpos / 100.

            object_joint_name = 'object{}:joint'.format(i)
            object_qpos = self.sim.data.get_joint_qpos(object_joint_name)
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_pos = object_qpos[:3]  # only position
            goal_pos.append(object_pos)
        goal_pos = np.array(goal_pos)
        return np.squeeze(goal_pos.flatten())

    # TODO Setup _viewer_setup called by _get_viewer and render

    # TODO redefine _render_callback for multi-objects env
    def _render_callback(self):
        # visualize all targets
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        for i in range(self.n_object):
            site_id = self.sim.model.site_name2id('target{}'.format(i))
            self.sim.model.site_pos[site_id] = self.goal[i * 3:i * 3 + 3] - sites_offset[i]
            self.sim.forward()
