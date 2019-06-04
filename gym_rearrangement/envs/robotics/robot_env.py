import os
import copy
import numpy as np
from interval import Interval

import gym
from gym import error, spaces
from gym.utils import seeding
from gym_rearrangement.core.goal_env import GoalEnv

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e))

DEFAULT_SIZE = 500

BOX_RANGE_X = Interval(1.0, 1.6)
BOX_RANGE_Y = Interval(0.4, 1.1)
BOX_RANGE_Z = Interval(0.4, 1)


class RobotEnv(GoalEnv):
    def __init__(self, model_path, initial_qpos, n_actions, n_substeps):
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

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()  # target state

        # specify observation space and action space, necessary for goal env
        # for goal env, observation space is a gym.space.Dict instance
        obs = self._get_obs()
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

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs)

        # early termination if the gripper is out of range
        grip_pos = obs['observation'][:3]  # grip pos then object pos ...
        if grip_pos[0] in BOX_RANGE_X and grip_pos[1] in BOX_RANGE_Y and grip_pos[2] in BOX_RANGE_Z:
            done = False
        else:
            print('Early terminate for out of range actions')
            done = True
            reward = -50  # penalty to void early terminate policy

        self._step_cnt += 1  # Update step number
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
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
    # ----------------------------

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
