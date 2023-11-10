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
        self.num_timesteps = 0
        self.dump_configs(path=self.log_dir)
        self.loggers_list = {config["env_name"]: None for config in self.env_config_list}
        self.agent_params = None
        env_manager = self.envs_list[0]
        self.update_current_agent(env_manager)

    def dump_configs(self, path: str) -> None:
        for config in self.env_config_list:
            env_name = config["env_name"]
            with open(os.path.join(path, f"{env_name}_config.json"), "w", encoding="utf8") as f:
                json.dump(config, f, indent=4, default=lambda _: '<not serializable>')
        with open(os.path.join(path, "model_config.json"), "w", encoding="utf8") as f:
            json.dump(self.model_config, f, indent=4, default=lambda _: '<not serializable>')

    def update_current_agent(self, env_manager):
        algo_class = self.get_algo_class()
        self.current_agent = algo_class(
                env=env_manager.envs,
                verbose=2,
                tensorboard_log=self.log_dir,
                **self.model_config,
            )
        self.current_agent.num_timesteps = self.num_timesteps
        logger = self.loggers_list.get(env_manager.env_name)
        if logger is None:
            self.loggers_list[env_manager.env_name] = self.current_agent._logger
        else:
            self.current_agent.set_logger(logger)
        if self.agent_params is None:
            if self.model_params_path is None:
                print("\nNo path to the PyTorch params of the model provided. Initializing new model.\n")
                self.agent_params = self.current_agent.get_parameters()
            else:
                model_params = pickle.load(open(self.model_params_path, "rb"))
                self.current_agent.set_parameters(model_params)
                self.agent_params = model_params
        else:
            self.current_agent.set_parameters(self.agent_params)

    def train(self, save_every=1) -> None:
        self.save()
        last_saved = 0
        pool = cycle(list(zip(self.envs_list, self.callbacks_list)))
        for _ in range(self.repeat):
            env_manager, callbacks = next(pool)
            env_manager.build_env()
            self.update_current_agent(env_manager)
            agent_timesteps_before = self.current_agent.num_timesteps
            self.current_agent.learn(
                total_timesteps=self.timesteps,
                callback=list(callbacks),
                reset_num_timesteps=False,
            )
            self.agent_params = self.current_agent.get_parameters()
            timesteps_done = self.current_agent.num_timesteps - agent_timesteps_before
            self.num_timesteps += timesteps_done
            if self.num_timesteps - last_saved > save_every:
                last_saved = self.num_timesteps
                self.save()
            env_manager.delete_env()
        self.save()

    def save(self) -> None:
        with open(os.path.join(self.log_dir, f"model_params_{self.num_timesteps}.pkl"), "wb") as file:
            pickle.dump(self.agent_params, file)
        for env_manager, config in zip(self.envs_list, self.env_config_list):
            env_name = config["env_name"]
            env_manager.envs.save(os.path.join(self.log_dir, f"{env_name}_{self.num_timesteps}.pkl"))

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


if __name__ == "__main__":
    print("This is a module. Run main.py to train the agent.")
