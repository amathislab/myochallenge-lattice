import os
import time
import json
import numpy as np
from utils.utils import RemoteConnection
from definitions import ROOT_DIR
from main_dataset_recurrent_ppo import load_vecnormalize, load_model
from envs.environment_factory import EnvironmentFactory


################################################
## A -replace with your trained policy.
## HERE an example from a previously trained policy with deprl is shown (see https://github.com/facebookresearch/myosuite/blob/main/docs/source/tutorials/4a_deprl.ipynb)
## additional dependencies such as gym and deprl might be needed
class Agent:
    def __init__(self, model, env_norm):
        self.model = model
        self.env_norm = env_norm
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

        
    def reset(self):
        self.episode_starts = np.ones((1,), dtype=bool)
        self.lstm_states = None
        
    def get_action(self, obs):
        action, self.lstm_states = self.model.predict(
                self.env_norm.normalize_obs(obs),
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=True,
            )
        self.episode_starts = False
        return action


EXPERIMENT_PATH = os.path.join(ROOT_DIR, "output", "trained_agents", "curriculum_step_10")
CHECKPOINT_NUM = 1432000000

if __name__ == "__main__":
    
    print('Loaded this: ',EXPERIMENT_PATH,CHECKPOINT_NUM)
    model = load_model(EXPERIMENT_PATH, CHECKPOINT_NUM)
    
    config_path = os.path.join(EXPERIMENT_PATH, "env_config.json")
    env_config = json.load(open(config_path, "r"))
    norm_env = EnvironmentFactory.create(**env_config)

    env_config_base= {"env_name": "RelocateEnvPhase2", "seed":0}
    base_env = EnvironmentFactory.create(**env_config_base)
 
    envs = load_vecnormalize(EXPERIMENT_PATH, CHECKPOINT_NUM, norm_env)
    envs.training = False

    pi = Agent(model, envs)
    ################################################
    flag_completed = None # this flag will detect then the whole eval is finished
    repetition = 0
    episodes = 0
    perfs = []
    solved = []
    efforts = []
    while not flag_completed and repetition < 1000:
        flag_trial = None # this flag will detect the end of an episode/trial
        counter = 0
        success = 0
        cum_reward = 0
        repetition +=1
        while not flag_trial:

            if counter == 0:
                print('RELOCATE: Trial #'+str(repetition))
                # obs = rc.reset()
                obs = base_env.reset()

            ################################################
            ### B - HERE the action is obtained from the policy and passed to the remote environment
            if counter == 0:
                pi.reset()
                
            action = pi.get_action(obs)
            if (counter > 90) and (base_env.obs_dict['obj_pos'][2] < 1.01):
                action = -1*np.ones_like(action)
            if np.abs(np.linalg.norm(base_env.obs_dict['pos_err'], axis=-1)) < 0.1:
                success +=1
            if success >= 5:
                action = -1*np.ones_like(action)
            ################################################

            ## gets info from the environment
            obs, rewards, done, info = base_env.step(action)

            flag_trial = done #base["feedback"][2]
            counter +=1
            cum_reward += rewards
        
        print(info["rwd_dict"])
        print('Solved? ', info["rwd_dict"]['solved'])
        episodes+= 1
        perfs.append(cum_reward)
        efforts.append(info["rwd_dict"]['act_reg'])
        if info["rwd_dict"]['solved'] == 1:
            solved.append(info["rwd_dict"]['solved'])
        else:
            solved.append(0)

        if (episodes) % 10 == 0:
            perf_error = np.std(perfs) / np.sqrt(episodes + 1)
            solved_error = np.std(solved) / np.sqrt(episodes + 1)
            effort_error = np.std(efforts) / np.sqrt(episodes + 1)

            print(f"\nEpisode {episodes+1}")
            # print(f"Average rew: {np.mean(perfs):.2f} +/- {perf_error:.2f}\n")
            print(f"Average effort: {np.abs(np.mean(efforts)):.4f}+/- {effort_error:.4f}\n")
            print(f"Average solved: {np.sum(solved)/(episodes):.4f}\n")