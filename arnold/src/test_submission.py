import os
import time
import json
import numpy as np
from utils.utils import RemoteConnection
from definitions import ROOT_DIR
from main_dataset_recurrent_ppo import load_vecnormalize, load_model
from envs.environment_factory import EnvironmentFactory


def get_custom_observation(rc):
    """
    Use this function to create an observation vector from the 
    environment provided observation dict for your own policy.
    By using the same keys as in your local training, you can ensure that 
    your observation still works.
    """
    # example of obs_keys for deprl baseline
    obs_keys = ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    obs_keys.append('act')

    obs_dict = rc.get_obs_dict(rc.sim)

    return rc.obsdict2obsvec(obs_dict, obs_keys)


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


EXPERIMENT_PATH = os.path.join(ROOT_DIR, "output/training/mani/_seed_123_max_steps_150_reg_0.1__solved_20.0_pos_dist_10.0_rot_dist_0.0_reach_dist_0.0_lift_0.0_max_app_0.0_reach_z_0.0_job_225")
CHECKPOINT_NUM = 1124000000

if __name__ == "__main__":
    
    print('Loaded this: ',EXPERIMENT_PATH,CHECKPOINT_NUM)
    model = load_model(EXPERIMENT_PATH, CHECKPOINT_NUM)
    
    config_path = os.path.join(EXPERIMENT_PATH, "env_config.json")
    env_config = json.load(open(config_path, "r"))
    norm_env = EnvironmentFactory.create(**env_config)

    env_config_base= {"env_name":"RelocateEnvPhase2", "seed":0}
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
    while not flag_completed:
        flag_trial = None # this flag will detect the end of an episode/trial
        counter = 0
        cum_reward = 0
        repetition +=1
        while not flag_trial :

            if counter == 0:
                print('RELOCATE: Trial #'+str(repetition)+'Start Resetting the environment and get 1st obs')
                # obs = rc.reset()
                obs = base_env.reset()

            ################################################
            ### B - HERE the action is obtained from the policy and passed to the remote environment
            if counter == 0:
                pi.reset()
                
            action = pi.get_action(obs)
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
        if info["rwd_dict"]['solved'] == 1:
            solved.append(info["rwd_dict"]['solved'])
        else:
            solved.append(0)

        if (episodes) % 10 == 0:
            perf_error = np.std(perfs) / np.sqrt(episodes + 1)
            solved_error = np.std(solved) / np.sqrt(episodes + 1)

            print(f"\nEpisode {episodes+1}")
            print(f"Average rew: {np.mean(perfs):.2f} +/- {perf_error:.2f}\n")
            print(f"Average solved: {np.sum(solved)/(episodes):.2f}\n")