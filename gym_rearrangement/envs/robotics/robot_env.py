import copy
import os

import numpy as np
from gym import error, spaces
from gym.utils import seeding
from interval import Interval

from gym_rearrangement.core.goal_env import GoalEnv

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: "
        "https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

BOX_RANGE_X = Interval(1.0, 1.45)
BOX_RANGE_Y = Interval(0.3, 1.2)
BOX_RANGE_Z = Interval(0.4, 0.7)


class RobotEnv(GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps, n_obj, distance_threshold):
        print("test in RobotEnv")
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
            print(fullpath)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        self.model = mujoco_py.load_model_from_path(fullpath)
        # n_substeps: number of mujoco steps in each step funciton call
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)
        self.viewer = None
        self.data = self.sim.data
        self._viewers = {}
        self._step_cnt = 0  # number of steps run so far
        self.episode_cnt = 0
        self.n_obj = n_obj  # number of objects
        self.threshold = distance_threshold
        self.np_random = None  # random module in numpy
        self.use_reach_policy = True

        self.obj_id = list(range(self.n_obj))  # which to pick first
        np.random.shuffle(self.obj_id)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = None  # initialize goal state

        obs = self.reset()  # get initial setups for goals and objects

        # specify observation space and action space, necessary for goal env
        # for goal env, observation space is a gym.space.Dict instance
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape,
                                    dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape,
                                     dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape,
                                   dtype='float32'),
        ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step_action(self, s, g, axis):
        if abs(s - g) < self.threshold:
            return 0
        elif s < g:
            return 0.4 if axis != 'z' else 0.2
        else:
            return -0.4 if axis != 'z' else -0.2

    def reach_target_policy(self, grip_pos, target_pos):
        """
        Reach to the target to reduce exploration
        :param grip_pos:
        :param target_pos:
        :return:
        """
        x = self._step_action(grip_pos[0], target_pos[0], 'x')
        y = self._step_action(grip_pos[1], target_pos[1], 'y')
        z = self._step_action(grip_pos[2], target_pos[2], 'z')
        return [x, y, z]

    def step(self, action):
        # obj need to reach at current
        action = np.clip(action, self.action_space.low, self.action_space.high)
        done = False
        if len(self.obj_id) > 0:
            obj_id = self.obj_id[-1]

            # gripper joint state, range 0, 0.05
            grip_pos = self.sim.data.get_site_xpos('robot0:grip')

            # block pos
            object_pos = self.sim.data.get_site_xpos('object{}'.format(obj_id))
            target_pos = self.goal[obj_id * 3: (obj_id + 1) * 3]
            # print('Gripper block distance', abs(grip_pos - object_pos) * 100)

            is_far = any(abs(grip_pos - object_pos) > self.threshold)
            target_reached = self.goal_distance(object_pos, target_pos) < self.threshold

            # pop up the reach target if it is reached
            if target_reached:
                print('Success with obj id: {}'.format(obj_id))
                self.obj_id.pop()
                action = np.array([0, 0, 1, 0.01])  # move up and place

            if self.use_reach_policy and is_far and not target_reached:
                # print('Episode step: {}, Reaching object {} with reach target policy'.format(
                #     self.episode_cnt, obj_id))
                pos_act = self.reach_target_policy(grip_pos, object_pos)
                action[:3] = np.array(pos_act)

            action = np.clip(action, self.action_space.low, self.action_space.high)
            self._set_action(action)
            self.sim.step()
            self._step_callback()

            info = {
                'is_success': len(self.obj_id) == 0,
                'is_curr_success': self._is_success(object_pos, target_pos),
                'curr_obj_left': obj_id,
            }
            obs = self._get_obs()
            reward = self.compute_rewards(obs)

            # early termination if the gripper is out of range
            if not self.check_in_range(grip_pos, object_pos):
                done = True
                reward = -50  # penalty to void early terminate policy

        else:  # all task is over
            info = {'is_success': True}
            obs = self._get_obs()
            reward = self.compute_rewards(obs)

        self._step_cnt += 1  # Update step number
        self.episode_cnt += 1

        return obs, reward, done, info

    def check_in_range(self, grip_pos, obj_pos):
        """
        Check if the gripper and block in range of desk, return false if not
        :param gripper: gripper end effector pos
        :param block: block pos
        :return:
        """
        grip_in_range = grip_pos[0] in BOX_RANGE_X and grip_pos[1] in BOX_RANGE_Y and grip_pos[
            2] in BOX_RANGE_Z
        blk_in_range = obj_pos[2] >= 0.4
        if not grip_in_range:
            print('Gripper is out of range ...')
        if not blk_in_range:
            print('Object is out of range ...')
        return grip_in_range and blk_in_range

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        self.episode_cnt = 0
        self.obj_id = list(range(self.n_obj))
        np.random.shuffle(self.obj_id)

        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def compute_rewards(self, obs):
        # Compute distance between goal and the achieved goal.
        # Maybe the distance between gripper and object should be included
        # So it is a two stage task: approximate the object, pick it to the goal
        # rewards = (grip_pos - object_pos)**2 + (target_pos - ojbect_pos)**2
        achieved_goal = obs['achieved_goal']  # achieved goal is the current pos of object
        goal = obs['desired_goal']
        assert len(goal) == len(achieved_goal) == 3 * self.n_obj
        grip_pos = obs['observation'][:3]
        # print('achieved goal: {}, goal: {}, gripper pos: {}'.format(achieved_goal, goal, grip_pos))
        if self.n_obj == 1:  # only 1 object
            d = self.goal_distance(achieved_goal[:2], goal[:2])
        else:  # more objects
            # TODO: we cann't use the sum of each goal-object distance, because they contradict each other
            # we don't use two stage reward because we use the reach to object policy
            d = [self.goal_distance(achieved_goal[3 * i: 3 * (i + 1) - 1], goal[3 * i: 3 * (i + 1) - 1]) for i
                 in
                 range(self.n_obj)]

        # sparse reward: either 0 or 1 reward
        if self.reward_type == 'sparse':
            return (np.array(d) < self.threshold).sum().astype(np.float32)
        else:
            return -(np.array(d).sum())

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()  # set camera etc.
            self._viewers[mode] = self.viewer
        return self.viewer

    # Extension methods

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    # def compute_rewards(self, obs):
    #     """compute rewards.
    #     """
    #     raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act,
                                         old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def get_image(self, width=84, height=84, camera_name=None):
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name
        )

    def init_camera(self, init_fn):
        viewer = mujoco_py.MjRenderContextOffscreen(self.sim)
        init_fn(viewer.cam)
        self.sim.add_render_context(viewer)

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def linear_schedule(self):
        """
        Linear weights schedule for reaching and placing.

        :return: (function)
        """
        decay_total_steps = 3e5
        progress = float(self._step_cnt / decay_total_steps)
        reach_weight = 1 - progress if progress < 0.6 else 0.4
        place_weight = progress if progress < 0.6 else 0.6

        return place_weight, reach_weight
