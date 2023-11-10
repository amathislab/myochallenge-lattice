import torch
from typing import Union

class MyDummyVecEnv :

    def __init__(self, env, using_tensor_buffer) -> None:
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.reward_range = env.reward_range
        self.metadata = env.metadata
        self.spec = env.spec
        self.num_envs = env.num_envs
        self.using_tensor_buffer = using_tensor_buffer
    
    def save(self, path) :
        self.env.save(path)
    
    def load(self, path) :
        self.env.load(path)
    
    def reset(self) :
        return self._prepare_output(self.env.reset())

    def step(self, action) :
        obs, rew, done, info = self.env.step(action)
        if done.any():
            # save final observation where user can get it, then reset
            info["terminal_observation"] = obs
            obs = self.env.reset()
        return self._prepare_output(obs), self._prepare_output(rew), self._prepare_output(done), [info]*self.num_envs

    def _prepare_output(self, d : Union[dict, torch.Tensor]) :
        
        if self.using_tensor_buffer :
            return d
        else :
            if isinstance(d, dict) :
                for item in d.values() :
                    item = item.cpu().numpy()
            elif isinstance(d, torch.Tensor) :
                d = d.cpu().numpy()
            else :
                raise NotImplementedError
            return d