import gym
from gym import error
import abc
from collections import OrderedDict


class GoalEnv(gym.Env, metaclass=abc.ABCMeta):
    """
    Effectively a gym.GoalEnv, but we add three more functions:

        - get_goal
        - sample_goals
        - compute_rewards

    We also change the compute_reward interface to take in an action and
    observation dictionary.
    """

    def reset(self):
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error('GoalEnv requires an observation space of type gym.spaces.Dict')
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in self.observation_space.spaces:
                raise error.Error(
                    'GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(
                        key))

    def compute_reward(self, obs):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def get_goal(self):
    #     """
    #     Returns a dictionary
    #     """
    #     pass

    """
    Implement the batch-version of these functions.
    """

    def _sample_goal(self):
        """
        :param batch_size:
        :return: Returns a dictionary mapping desired goal keys to arrays of
        size BATCH_SIZE x Z, where Z depends on the key.
        """
        pass


    def get_diagnostics(self, *args, **kwargs):
        """
        :param rollouts: List where each element is a dictionary describing a
        rollout. Typical dictionary might look like:
        {
            'observations': np array,
            'actions': np array,
            'next_observations': np array,
            'rewards': np array,
            'terminals': np array,
            'env_infos': list of dictionaries,
            'agent_infos': list of dictionaries,
        }
        :return: OrderedDict. Statistics to save.
        """
        return OrderedDict()

    @staticmethod
    def unbatchify_dict(batch_dict, i):
        """
        :param batch_dict: A batch dict is a dict whose values are batch.
        :return: the dictionary returns a dict whose values are just elements of
        the batch.
        """
        new_d = {}
        for k in batch_dict.keys():
            new_d[k] = batch_dict[k][i]
        return new_d

    @staticmethod
    def batchify_dict(batch_dict, i):
        """
        :param batch_dict: A batch dict is a dict whose values are batch.
        :return: the dictionary returns a dict whose values are just elements of
        the batch.
        """
        new_d = {}
        for k in batch_dict.keys():
            new_d[k] = batch_dict[k][i]
        return new_d
