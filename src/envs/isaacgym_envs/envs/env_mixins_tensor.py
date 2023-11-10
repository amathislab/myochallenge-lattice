import torch
import gym
from definitions import ACT_KEY, GOAL_KEY, POSITIONS_KEY

class TensorObsMixin:
    """This class enhances the observation by including a history of transitions"""

    def _dict_obs_init_addon(self, include_adapt_state, num_memory_steps, obs_shape_dict : dict, num_envs : int):
        """Function to augment an environment's init function by changing
        the observation space and the containers for the transition history.

        Args:
            include_adapt_state (bool): whether to include the transition history
            num_memory_steps (int): number of transitions to include in the state
            obs_shape_dict: observation shape in terms of a dict
        """
        self.obs_shape_dict = obs_shape_dict
        self._include_adapt_state = include_adapt_state
        if include_adapt_state:
            self.num_memory_steps = num_memory_steps
            self._obs_prev_list = {
                key : torch.zeros(
                    (num_envs, num_memory_steps+1, *obs_shape)
                ) for key, obs_shape in obs_shape_dict.items()
            }
        else:
            self.num_memory_steps = 0
        self._obs_prev_len = 0
        obs_boxes_dict = {
            key : gym.spaces.Box(
                -10, 10, shape=(1 + self.num_memory_steps, *obs_shape)
            )
            for key, obs_shape in self.obs_shape_dict.items()
        }
        obs_space = gym.spaces.Dict(obs_boxes_dict)
        self.observation_space = obs_space

    def create_history_reset_state(self, obs_dict : dict):
        """Function to augment the state returned by the reset of a myosuite environment

        Args:
            obs_dict (np.array): current observation

        Returns:
            dict with the keys specified in self.obs_keys, if dict mode, otherwise an array with
            the values of the dict
        """
        reduced_obs_dict = {key: obs_dict[key] for key in self.obs_shape_dict.keys()}
        if self._include_adapt_state:
            # Empty the history cache and store the new obs
            for prev_value, cur_value in zip(self._obs_prev_list.values(), obs_dict.values()) :
                prev_value.zero_()
                prev_value[:, 0, ...] = cur_value
            self._obs_prev_len = 1
            # Create the observation dict with history
            return_dict = self.compute_history_obs_dict()
        else:
            return_dict = reduced_obs_dict
        return return_dict

    def create_history_step_state(self, obs_dict : dict):
        """Function to augment the state returned by the step function of a myosuite environment
        Args:
            obs_dict (np.array): current observation

        Returns:
            dict with the keys specified in self.obs_keys, if dict mode, otherwise an array with
            the values of the dict
        """
        reduced_obs_dict = {key: obs_dict[key] for key in self.obs_shape_dict.keys()}
        if self._include_adapt_state:
            for prev_value, cur_value in zip(self._obs_prev_list.values(), obs_dict.values()) :
                prev_value[:, self._obs_prev_len, ...] = cur_value
            self._obs_prev_len = (self._obs_prev_len + 1) % (self.num_memory_steps + 1)
            return_dict = self.compute_history_obs_dict()
        else:
            return_dict = reduced_obs_dict
        return return_dict
    
    def compute_history_obs_dict(self):
        return_dict = {}
        for key in self.obs_shape_dict.keys():
            return_value = torch.concat((self._obs_prev_list[key][:, self._obs_prev_len:, ...], self._obs_prev_list[key][:, :self._obs_prev_len, ...]), dim=1)
            return_dict[key] = return_value
        return return_dict

class TensorObsEmbeddingMixin:
    def _obs_embedding_init_addon(self, obs_shape_dict : dict, num_envs : int):
        obs_space_dict = self.observation_space.spaces
        position_shape = self.obs_positions.shape[-1:]
        obs_space_dict.update({POSITIONS_KEY: gym.spaces.Box(0, 1000, shape=position_shape)})
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        
    def add_positions_to_obs(self, obs):
        obs_ids = self.obs_positions
        obs.update({POSITIONS_KEY: obs_ids})
        return obs