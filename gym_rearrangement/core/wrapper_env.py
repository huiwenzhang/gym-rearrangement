import numpy as np
from gym.spaces import Box
import inspect
from gym_rearrangement.core.serializable import Serializable


class ProxyEnv(Serializable):
    def __init__(self, wrapped_env):
        self.quick_init(locals())
        self._wrapped_env = wrapped_env
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def __getattr__(self, attrname):
        if attrname == '_serializable_initialized':
            return None
        return self.getattr_recursive(attrname)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)

    def _get_all_attributes(self):
        """Get all (inherited) instance and class attributes
        :return: (dict<str, object>) all_attributes
        """
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name):
        """Recursively check wrappers to find attribute.
        :param name (str) name of attribute to look for
        :return: (object) attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.wrapped_env, 'getattr_recursive'):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.wrapped_env.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.wrapped_env, name)

        return attr


class NormalizedBoxEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations.
    """

    def __init__(
            self,
            env,
            obs_means=None,
            obs_stds=None,
            obs_to_normalize_keys=['observation'],
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._wrapped_env = env
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_means is None and obs_stds is None)
        num_obs_types = len(obs_to_normalize_keys)
        if self._should_normalize:
            if obs_means is None:
                obs_means = dict()
                for key in self.obs_to_normalize_keys:
                    obs_means[key] = np.zeros_like(env.observation_space[key].low)
            else:
                obs_means = dict()
                for key in self.obs_to_normalize_keys:
                    obs_means[key] = np.array(obs_means[key])
            if obs_stds is None:
                obs_stds = dict()
                for key in self.obs_to_normalize_keys:
                    obs_stds[key] = np.zeros_like(env.observation_space[key].low)
            else:
                obs_stds = dict()
                for key in self.obs_to_normalize_keys:
                    obs_stds[key] = np.array(obs_stds[key])
        self._obs_means = obs_means
        self._obs_stds = obs_stds
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)
        self.obs_to_normalize_keys = obs_to_normalize_keys

    def estimate_obs_stats(self, obs_batch, override_values=False):
        raise NotImplementedError()

    def _apply_normalize_obs(self, obs):
        for key in self.obs_to_normalize_keys:
            obs[key] = (obs[key] - self._obs_means[key]) / (self._obs_stds[key] + 1e-8)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        d["_obs_means"] = self._obs_means
        d["_obs_stds"] = self._obs_stds
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_means = d["_obs_means"]
        self._obs_stds = d["_obs_stds"]

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env
