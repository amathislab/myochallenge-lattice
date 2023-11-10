import gym
import random
import numpy as np
from definitions import ACT_KEY, OBJ_KEY, GOAL_KEY, POSITIONS_KEY


class MuscleMultiEnv(gym.Env):
    """Environemt which wraps multiple muscle environments and chooses a random one from 
    the list at each episode. Also provides a unique identifier for each observation component.
    """
    def __init__(self, env_list) -> None:
        super().__init__()
        self.obs_keys = [ACT_KEY, OBJ_KEY, GOAL_KEY, POSITIONS_KEY]
        self.env_list = env_list
        self.obs_to_embedding_idx_dict = self.create_obs_to_embedding_mapping()
        self.current_env = self.update_current_env()

    # @property
    # def action_space(self):
    #     return self.current_env.action_space        

    # @property
    # def observation_space(self):
    #     obs_space_dict = self.current_env.observation_space.spaces
    #     num_keys = sum(space.shape[1] for space in obs_space_dict.values())
    #     obs_space_dict.update({POSITIONS_KEY: gym.spaces.Box(0, 100, shape=(num_keys,))})
    #     return gym.spaces.Dict(obs_space_dict)
    #     # return self.current_env.observation_space

    def reset(self):
        self.current_env = self.update_current_env()
        obs = self.current_env.reset()
        return self.add_positions_to_obs(obs)
    
    def step(self, action):
        obs, reward, done, info = self.current_env.step(action)
        return self.add_positions_to_obs(obs), reward, done, info

    def render(self, mode="human"):
        return self.current_env.render(mode)

    def close(self):
        for env in self.env_list:
            env.close()
    
    def seed(self, seed=None):
        for env in self.env_list:
            env.seed(seed)
    
    def add_positions_to_obs(self, obs):
        obs_ids = [self.obs_to_embedding_idx_dict[obs_element] for obs_element in self.current_env.get_obs_elements()]
        obs.update({POSITIONS_KEY: obs_ids})
        return obs
    
    def create_obs_to_embedding_mapping(self):
        obs_elements_set = set()
        for env in self.env_list:
            # actuators, objects, goals = env.get_obs_elements()
            # obs_elements_concat = [*actuators, *objects, *goals]
            obs_elements = env.get_obs_elements()
            obs_elements_set.update(obs_elements)
        
        obs_elements_dict = {key: val for val, key in enumerate(obs_elements_set)}
        return obs_elements_dict
    
    def update_current_env(self):
        current_env = random.choice(self.env_list)
        self.action_space = current_env.action_space
        self.action_space.low = np.mean(self.action_space.low)       
        self.action_space.high = np.mean(self.action_space.high)  
        obs_space_dict = current_env.observation_space.spaces
        num_keys = sum(space.shape[-1] for key, space in obs_space_dict.items() if key != POSITIONS_KEY)
        obs_space_dict.update({POSITIONS_KEY: gym.spaces.Box(0, 100, shape=(num_keys,))})  # maximum 100 different encodings
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        return current_env

    