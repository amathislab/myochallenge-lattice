import gym
import numpy as np
from collections import deque
from definitions import POSITIONS_KEY, OBS_ID_MAP


class DictObsMixin:
    """This class enhances the observation by including a history of transitions"""

    def _dict_obs_init_addon(self, include_adapt_state, num_memory_steps):
        """Function to augment an environment's init function by changing
        the observation space and the containers for the transition history.

        Args:
            include_adapt_state (bool): whether to include the transition history
            num_memory_steps (int): number of transitions to include in the state
        """
        obs_dict = self.get_obs_dict(self.sim)
        self._include_adapt_state = include_adapt_state
        if include_adapt_state:
            self.num_memory_steps = num_memory_steps
            self._obs_prev_list = deque(
                [], maxlen=num_memory_steps + 1
            )  # Storing observations from newest to oldest
        else:
            self.num_memory_steps = 0

        obs_boxes_dict = {
            key: gym.spaces.Box(
                -10, 10, shape=(1 + self.num_memory_steps, *np.shape(np.atleast_2d(obs_dict[key])))
            )
            for key in self.obs_keys
        }
        obs_space = gym.spaces.Dict(obs_boxes_dict)
        self.observation_space = obs_space

    def create_history_reset_state(self, obs_dict):
        """Function to augment the state returned by the reset of a myosuite environment

        Args:
            obs_dict (np.array): current observation

        Returns:
            dict with the keys specified in self.obs_keys, if dict mode, otherwise an array with
            the values of the dict
        """
        reduced_obs_dict = {key: obs_dict[key] for key in self.obs_keys}
        if self._include_adapt_state:
            # Empty the history cache and store the new obs
            self._obs_prev_list.clear()
            zero_obs = {key: np.zeros_like(value) for key, value in reduced_obs_dict.items()}
            for _ in range(self.num_memory_steps):
                self._obs_prev_list.append(zero_obs)
            self._obs_prev_list.append(reduced_obs_dict)
            # Create the observation dict with history
            return_dict = self.compute_history_obs_dict()
        else:
            return_dict = reduced_obs_dict
        return return_dict

    def create_history_step_state(self, obs_dict):
        """Function to augment the state returned by the step function of a myosuite environment
        Args:
            obs_dict (np.array): current observation

        Returns:
            dict with the keys specified in self.obs_keys, if dict mode, otherwise an array with
            the values of the dict
        """
        reduced_obs_dict = {key: obs_dict[key] for key in self.obs_keys}
        if self._include_adapt_state:
            self._obs_prev_list.append(reduced_obs_dict)
            return_dict = self.compute_history_obs_dict()
        else:
            return_dict = reduced_obs_dict
        return return_dict
    
    def compute_history_obs_dict(self):
        return_dict = {}
        for key in self.obs_keys:
            shape_one_element = self.observation_space[key].shape[1:]
            obs_list = [obs_t[key].reshape(shape_one_element) for obs_t in self._obs_prev_list]
            return_value = np.stack(obs_list)
            return_dict[key] = return_value
        return return_dict


class ObsEmbeddingMixin:
    def _obs_embedding_init_addon(self):
        obs_space_dict = self.observation_space.spaces
        num_positions = len(self.get_obs_elements())
        obs_space_dict.update({POSITIONS_KEY: gym.spaces.Box(0, 1000, shape=(num_positions,))})
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        
    def add_positions_to_obs(self, obs):
        obs_ids = [OBS_ID_MAP[obs_element] for obs_element in self.get_obs_elements()]
        obs.update({POSITIONS_KEY: obs_ids})
        return obs