import os

import numpy as np
from gym.utils import EzPickle

from gym_rearrangement.envs.robotics import fetch_env, utils, rotations

# Ensure we get the path separator correct on windows
# MODEL_XML_PATH = os.path.join('fetch', 'rearrange_4.xml')

# Parameters for random object positions
N_GRID = 3
TABLE_SIZE = 0.5 * 100  # width of the table, unit: cm
TABLE_CORNER = [105, 50]  # right bottom corner, robot view
GRID_SIZE = int(TABLE_SIZE * 0.85 / N_GRID)


class Rearrangement(fetch_env.FetchEnv, EzPickle):
    def __init__(self, reward_type='sparse', n_object=4, visual_targets=True, fix_goal=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self.n_object = n_object
        self.vis_targets = visual_targets
        self.fix_goal = fix_goal
        model_xml_path = os.path.join('fetch', 'rearrange_{}.xml'.format(self.n_object))
        fetch_env.FetchEnv.__init__(
            self, model_xml_path, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.03,
            initial_qpos=initial_qpos, reward_type=reward_type, fix_goal=fix_goal, n_obj=n_object
        )
        EzPickle.__init__(self)

        # Save object related infos
        self.X = []  # x coordinates
        self.Y = []  # y coordinates
        # shapes and colors
        if self.n_object == 1:  # task degregate to pick and place
            self.color = np.array([2])
            self.shape = np.array([False])
        elif self.n_object == 2:
            self.color = np.array([2, 0])  # red, blue
            self.shape = np.array([False, True])  # True for sphere, False: box
        elif self.n_object == 3:
            self.color = np.array([2, 5, 0])
            self.shape = np.array([False, False, True])
        elif self.n_object == 4:
            self.color = np.array([2, 5, 0, 4])
            self.shape = np.array([False, False, True, True])
        else:
            self.color = np.array([2, 5, 1, 4, 3, 0])
            self.shape = np.array([False, False, False, True, True, True])

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of all objects.
        idx_coor = np.arange(N_GRID * N_GRID)
        np.random.shuffle(idx_coor)

        self.X = []
        self.Y = []
        obj_pos_id = idx_coor[:self.n_object]  # grid id taken by objects

        # Randomize the goal state
        # TODO: why we cannot put this after the following code?
        if not self.fix_goal:
            self.goal = self._sample_goal(idx_coor, obj_pos_id)

        for i in range(self.n_object):
            # block index
            x = idx_coor[i] % N_GRID
            y = (N_GRID - np.floor(idx_coor[i] / N_GRID) - 1).astype(np.uint8)
            # block coordinates, translate TABLE-CONRER distance
            # square of x, y is used for remove distance ambiguity
            # (means one object may have serval ojbects with the same distance)
            object_xpos = np.array(
                [(x + 0.5) * GRID_SIZE + x ** 2, (y + 0.5) * GRID_SIZE + y ** 2]) + np.array(
                TABLE_CORNER)
            object_xpos = object_xpos / 100.

            self.X.append(object_xpos[0])
            self.Y.append(object_xpos[1])

            object_joint_name = 'object{}:joint'.format(i)
            object_qpos = self.sim.data.get_joint_qpos(object_joint_name)
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos(object_joint_name, object_qpos)

        self.sim.forward()
        return True

    def _get_obs(self):
        dt = self.dt
        # gripper position state
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt

        # robot arm state
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        # gripper joint state
        grip_state = robot_qpos[-2:]
        grip_vel = robot_qvel[-2:] * dt

        # object state for each object
        object_pos, object_rot, object_velp, object_velr, grip_rel_pos = [], [], [], [], []
        for i in range(self.n_object):
            obj_name = 'object{}'.format(i)
            object_pos.append(self.sim.data.get_site_xpos(obj_name))
            # rotations
            object_rot.append(
                rotations.mat2euler(self.sim.data.get_site_xmat(obj_name)))
            # velocities
            object_velp.append(self.sim.data.get_site_xvelp(obj_name) * dt - grip_velp)
            object_velr.append(self.sim.data.get_site_xvelr(obj_name) * dt)
            # relative state between gripper and object
            grip_rel_pos.append(self.sim.data.get_site_xpos(obj_name) - grip_pos)

        object_pos = np.array(object_pos)
        object_rot = np.array(object_rot)
        object_velp = np.array(object_velp)
        object_velr = np.array(object_velr)
        grip_rel_pos = np.array(grip_rel_pos)

        achieved_goal = np.squeeze(object_pos.flatten())
        # 46D
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), grip_rel_pos.ravel(), grip_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, grip_vel
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _sample_goal(self, ids, taken_ids=None):
        """
        Sample goal positons for each objects
        :param taken_ids: the current coordinates id taken by each obj
        :return: goal positions
        """
        idx_coor = ids.tolist()

        # Remove the taken ids from the idx_coor, so that the sampled goal states will not overlapped
        # the current obj ids. However if the No. of obj is bigger than 4, we don't have enough spaces
        # so overlap must be happen
        if taken_ids is not None and len(taken_ids) < 5:
            for id in taken_ids:
                idx_coor.remove(id)

        np.random.shuffle(idx_coor)

        goal_pos = []
        for i in range(self.n_object):
            # grid index
            x = idx_coor[i] % N_GRID
            y = (N_GRID - np.floor(idx_coor[i] / N_GRID) - 1).astype(np.uint8)
            # grid coordinates
            object_xpos = np.array(
                [(x + 0.5) * GRID_SIZE + x ** 2, (y + 0.5) * GRID_SIZE + y ** 2]) + np.array(
                TABLE_CORNER)
            object_xpos = object_xpos / 100.

            object_joint_name = 'object{}:joint'.format(i)
            object_qpos = self.sim.data.get_joint_qpos(object_joint_name)
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            goal_pos.append(object_qpos[:3])

        goal_pos = np.array(goal_pos)
        return np.squeeze(goal_pos.flatten())

    # TODO Setup _viewer_setup called by _get_viewer and render
    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -15.

    def _render_callback(self):
        # visualize all targets
        if self.vis_targets:
            sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
            for i in range(self.n_object):
                site_id = self.sim.model.site_name2id('target{}'.format(i))
                self.sim.model.site_pos[site_id] = self.goal[i * 3:i * 3 + 3] - sites_offset[i]
                # self.sim.model.site_pos[site_id] = self.goal[i * 3:i * 3 + 3]
                self.sim.forward()
