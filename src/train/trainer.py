import json
import os
import pickle
from dataclasses import dataclass, field
from typing import List
from stable_baselines3 import PPO, SAC, TD3
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from models.ppo.ppo_tensor import PPOTensor
from itertools import cycle
from pink import PinkNoiseDist

@dataclass
class MyoTrainer:
    algo: str
    envs_list: List[VecNormalize]  # In each session one env is sampled
    env_config_list: List[dict]
    model_params_path: str
    log_dir: str
    model_config: dict = field(default_factory=dict)
    callbacks_list: List[BaseCallback] = field(default_factory=list)  # List of list of callbacks per environment
    timesteps: int = 10_000_000  # training timesteps per training session
    repeat: int = 1  # How many training session

    def __post_init__(self):
        self.dump_configs(path=self.log_dir)
        self.agent_list = self._init_agent()
        if self.model_params_path is None:
            self.agent_params = self.agent_list[0].get_parameters()
        else:
            self.agent_params = pickle.load(open(self.model_params_path, "rb"))
        self.num_timesteps = 0

    def dump_configs(self, path: str) -> None:
        for config in self.env_config_list:
            env_name = config["env_name"]
            with open(os.path.join(path, f"{env_name}_config.json"), "w", encoding="utf8") as f:
                json.dump(config, f, indent=4, default=lambda _: '<not serializable>')
        with open(os.path.join(path, "model_config.json"), "w", encoding="utf8") as f:
            json.dump(self.model_config, f, indent=4, default=lambda _: '<not serializable>')

    def _init_agent(self):
        algo_class = self.get_algo_class()
        agent_list = []
        for env in self.envs_list:
            agent_list.append(algo_class(
                env=env,
                verbose=2,
                tensorboard_log=self.log_dir,
                **self.model_config,
            ))
        if self.model_params_path is None:
            print("\nNo path to the PyTorch params of the model provided. Initializing new model.\n")
        else:
            print(f"\nLoading PyTorch params from {self.model_params_path}\n")
            model_params = pickle.load(open(self.model_params_path, "rb"))
            for agent in agent_list:
                agent.set_parameters(model_params)
        return agent_list

    def train(self, save_every=1) -> None:
        self.save()
        last_saved = 0
        pool = cycle(list(zip(self.agent_list, self.callbacks_list)))
        for _ in range(self.repeat):
            agent, callbacks = next(pool)
            agent.set_parameters(self.agent_params)
            agent_timesteps_before = agent.num_timesteps
            agent.learn(
                total_timesteps=self.timesteps,
                callback=list(callbacks),
                reset_num_timesteps=False,
            )
            self.agent_params = agent.get_parameters()
            
            timesteps_done = agent.num_timesteps - agent_timesteps_before
            self.num_timesteps += timesteps_done
            if self.num_timesteps - last_saved > save_every:
                last_saved = self.num_timesteps
                self.save()
        self.save()

    def save(self) -> None:
        with open(os.path.join(self.log_dir, f"model_params_{self.num_timesteps}.pkl"), "wb") as file:
            pickle.dump(self.agent_params, file)
        for env, config in zip(self.envs_list, self.env_config_list):
            env_name = config["env_name"]
            env.save(os.path.join(self.log_dir, f"{env_name}_{self.num_timesteps}.pkl"))

    def get_algo_class(self):
        if self.algo == "ppo":
            return PPO
        elif self.algo == "ppo_tensor" :
            return PPOTensor
        elif self.algo == "recurrent_ppo":
            return RecurrentPPO
        elif self.algo == "sac":
            return SAC
        elif self.algo == "td3":
            return TD3
        else:
            raise ValueError("Unknown algorithm ", self.algo)


@dataclass
class SingleEnvTrainer:
    algo: str
    envs: VecNormalize
    env_config: dict
    load_model_path: str
    log_dir: str
    model_config: dict = field(default_factory=dict)
    callbacks: List[BaseCallback] = field(default_factory=list)
    timesteps: int = 10_000_000
    pink_noise: bool = False

    def __post_init__(self):
        self.dump_configs(path=self.log_dir)
        self.agent = self._init_agent()
        if self.pink_noise:
            if self.algo != "sac":
                raise NotImplementedError("Pink noise only implemented for sac")
            else:
                action_dim = self.agent.actor.action_dist.action_dim
                seq_len = 1000  # TODO: make it generic. Now it works as all PyBullet envs have max len 1000
                self.agent.actor.action_dist = PinkNoiseDist(seq_len, action_dim)
                
    def dump_configs(self, path: str) -> None:
        with open(os.path.join(path, "env_config.json"), "w", encoding="utf8") as f:
            json.dump(self.env_config, f, indent=4, default=lambda _: '<not serializable>')
        with open(os.path.join(path, "model_config.json"), "w", encoding="utf8") as f:
            json.dump(self.model_config, f, indent=4, default=lambda _: '<not serializable>')

    def _init_agent(self):
        algo_class = self.get_algo_class()
        if self.load_model_path is not None:
            return algo_class.load(
                self.load_model_path,
                env=self.envs,
                tensorboard_log=self.log_dir,
                custom_objects=self.model_config,
            )
        print("\nNo model path provided. Initializing new model.\n")
        return algo_class(
            env=self.envs,
            verbose=2,
            tensorboard_log=self.log_dir,
            **self.model_config,
        )

    def train(self) -> None:
        self.agent.learn(
            total_timesteps=self.timesteps,
            callback=self.callbacks,
            reset_num_timesteps=False,
        )

    def save(self) -> None:
        self.agent.save(os.path.join(self.log_dir, "final_model.pkl"))
        self.envs.save(os.path.join(self.log_dir, "final_env.pkl"))

    def get_algo_class(self):
        if self.algo == "ppo":
            return PPO
        elif self.algo == "recurrent_ppo":
            return RecurrentPPO
        elif self.algo == "sac":
            return SAC
        elif self.algo == "td3":
            return TD3
        else:
            raise ValueError("Unknown algorithm ", self.algo)

if __name__ == "__main__":
    print("This is a module. Run main.py to train the agent.")
