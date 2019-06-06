from gym.spaces import Box, Dict
import numpy as np

from gym_rearrangement.core.wrapper_env import ProxyEnv


class FlatGoalEnv(ProxyEnv):
    def __init__(
            self,
            wrapped_env,
            obs_keys=None,
            goal_keys=None,
            append_goal_to_obs=False,
    ):
        self.quick_init(locals())
        super(FlatGoalEnv, self).__init__(wrapped_env)

        if obs_keys is None:
            obs_keys = ['observation']
        if goal_keys is None:
            goal_keys = ['desired_goal']

        self._append_goal_to_obs = append_goal_to_obs
        for k in obs_keys:
            assert k in self.wrapped_env.observation_space.spaces
        for k in goal_keys:
            assert k in self.wrapped_env.observation_space.spaces
        assert isinstance(self.wrapped_env.observation_space, Dict)

        self.obs_keys = obs_keys
        self.goal_keys = goal_keys
        # TODO: handle nested dict

        if self._append_goal_to_obs:
            keys = self.obs_keys + self.goal_keys
        else:
            keys = self.obs_keys

        for i in range(len(keys) - 1):
            shp1 = self.wrapped_env.observation_space.spaces[keys[i]].shape
            shp2 = self.wrapped_env.observation_space.spaces[keys[i + 1]].shape
            assert shp1 == shp2, 'Dimension does not match, cannot stack these features'

        self.observation_space = Box(
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].low
                for k in keys
            ]),
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].high
                for k in keys
            ]),
        )
        self.goal_space = Box(
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].low
                for k in goal_keys
            ]),
            np.hstack([
                self.wrapped_env.observation_space.spaces[k].high
                for k in goal_keys
            ]),
        )
        self._goal = None

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        if self._append_goal_to_obs:
            keys = self.obs_keys + self.goal_keys
        else:
            keys = self.obs_keys
        flat_obs = np.hstack([obs[k] for k in keys])
        return flat_obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        self._goal = np.hstack([obs[k] for k in self.goal_keys])
        if self._append_goal_to_obs:
            keys = self.obs_keys + self.goal_keys
        else:
            keys = self.obs_keys
        flat_obs = np.hstack([obs[k] for k in keys])
        return flat_obs

    # def get_goal(self):
    #     return self._goal
