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

    obs_dict = rc.get_obsdict()
    # add new features here that can be computed from obs_dict
    # obs_dict['qpos_without_xy'] = np.array(obs_dict['internal_qpos'][2:35].copy())

    return rc.obsdict2obsvec(obs_dict, obs_keys)


# time.sleep(60)

# LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

# if LOCAL_EVALUATION:
#     rc = RemoteConnection("environment:8085")
# else:
#     rc = RemoteConnection("localhost:8085")



# compute correct observation space
# shape = get_custom_observation(rc).shape
# rc.set_observation_space(shape)


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

# EXPERIMENT_PATH = os.path.join(ROOT_DIR, "output/training/2023-10-05/01-08-55_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0")
# CHECKPOINT_NUM = 334478592 #99993600

# EXPERIMENT_PATH = os.path.join(ROOT_DIR, "output/training/2023-10-06/10-36-17_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0")
# CHECKPOINT_NUM = 295981056 #334478592 #99993600

# EXPERIMENT_PATH = os.path.join(ROOT_DIR, "output/training/2023-10-11/14-04-55_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0")
# CHECKPOINT_NUM = 28498176 #295981056 #334478592 #99993600

EXPERIMENT_PATH = os.path.join(ROOT_DIR, "output/training/2023-10-12/16-41-54_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0")
CHECKPOINT_NUM = 104993280 #295981056 #334478592 #99993600

# Current best
# EXPERIMENT_PATH = os.path.join(ROOT_DIR, "output/training/2023-09-28/CustomChaseTag_seed_8_x_-5.0_5.0_y_-5.0_5.0_dist_2.0_hip_0.5_period_100.0_alive_0.0_solved_0.0_early_solved_300.0_joints_2.0_lose_-1000.0_ref_1.0_heel_0_gait_l_0.8_gait_c_1.0_fix_0.1_ran_0.45_mov_0.45_job_63")
# CHECKPOINT_NUM = 139991040

if __name__ == "__main__":
    
    
    model = load_model(EXPERIMENT_PATH, CHECKPOINT_NUM)
    
    config_path = os.path.join(EXPERIMENT_PATH, "env_config.json")
    env_config = json.load(open(config_path, "r"))
    base_env = EnvironmentFactory.create(**env_config)
    envs = load_vecnormalize(EXPERIMENT_PATH, CHECKPOINT_NUM, base_env)
    envs.training = False

    pi = Agent(model, envs)
    ################################################
    flag_completed = None # this flag will detect then the whole eval is finished
    repetition = 0
    episodes = 0
    perfs = []
    while not flag_completed:
        flag_trial = None # this flag will detect the end of an episode/trial
        counter = 0
        cum_reward = 0
        repetition +=1
        while not flag_trial :

            if counter == 0:
                print('RELOCATE: Trial #'+str(repetition)+'Start Resetting the environment and get 1st obs')
                # obs = rc.reset()
                obs = envs.reset()

            ################################################
            ### B - HERE the action is obtained from the policy and passed to the remote environment
            # obs = get_custom_observation(base_env.env)
            if counter == 0:
                pi.reset()
                
            action = pi.get_action(obs)
            ################################################

            ## gets info from the environment
            # base = rc.act_on_environment(action)
            obs, rewards, done, info = envs.step(action)


            # print(info[0].keys())
            # print(base_env.sim) 
            # print(action)

            # print(info[0]['solved'], info[0]['pos_dist']) #info[0]['rwd_dense']
            # obs =  base["feedback"][0]

            flag_trial = done #base["feedback"][2]
            # flag_completed = done # base["eval_completed"]

            # print(f"RELOCATE: Agent Feedback iter {counter} -- trial solved: {flag_trial} -- task solved: {flag_completed}")
            # print("*" * 100)
            # print(info)
            counter +=1
            cum_reward += rewards
        episodes+= 1
        perfs.append(cum_reward)

        if (episodes + 1) % 10 == 0:
            # len_error = np.std(lens) / np.sqrt(counter + 1)
            perf_error = np.std(perfs) / np.sqrt(episodes + 1)

            print(f"\nEpisode {episodes+1}")
            # print(f"Average len: {np.mean(lens):.2f} +/- {len_error:.2f}")
            print(f"Average rew: {np.mean(perfs):.2f} +/- {perf_error:.2f}\n")