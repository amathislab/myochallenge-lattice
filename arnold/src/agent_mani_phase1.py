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
    obs_keys = [
            'hand_qpos',
            'hand_qvel',
            'obj_pos',
            'goal_pos',
            'pos_err',
            'obj_rot',
            'goal_rot',
            'rot_err'
    ]
    obs_keys.append('act')

    obs_dict = rc.get_obsdict()
    # add new features here that can be computed from obs_dict
    # obs_dict['qpos_without_xy'] = np.array(obs_dict['internal_qpos'][2:35].copy())

    return rc.obsdict2obsvec(obs_dict, obs_keys)


time.sleep(60)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8085")
else:
    rc = RemoteConnection("localhost:8085")



# compute correct observation space
shape = get_custom_observation(rc).shape
rc.set_observation_space(shape)


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

# EXPERIMENT_PATH = os.path.join(ROOT_DIR, "output/training/2023-10-11/14-04-55_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0")
# CHECKPOINT_NUM = 28498176 #295981056 #334478592 #99993600

EXPERIMENT_PATH = os.path.join(ROOT_DIR, "output/training/2023-10-12/16-41-54_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0")
CHECKPOINT_NUM = 104993280 #295981056 #334478592 #99993600

if __name__ == "__main__":

    model = load_model(EXPERIMENT_PATH, CHECKPOINT_NUM)
    
    config_path = os.path.join(EXPERIMENT_PATH, "env_config.json")
    env_config = json.load(open(config_path, "r"))
    base_env = EnvironmentFactory.create(**env_config)
    envs = load_vecnormalize(EXPERIMENT_PATH, CHECKPOINT_NUM, base_env)
    envs.training = False
    pi = Agent(model, envs)

    flag_completed = None # this flag will detect then the whole eval is finished
    repetition = 0
    while not flag_completed:
        flag_trial = None # this flag will detect the end of an episode/trial
        counter = 0
        repetition +=1
        while not flag_trial :

            if counter == 0:
                print('Relocate: Trial #'+str(repetition)+'Start Resetting the environment and get 1st obs')
                obs = rc.reset()

            ################################################
            ### B - HERE the action is obtained from the policy and passed to the remote environment
            obs = get_custom_observation(rc)
            if counter == 0:
                pi.reset()
                
            action = pi.get_action(obs)
            ################################################

            ## gets info from the environment
            base = rc.act_on_environment(action)
            obs =  base["feedback"][0]

            flag_trial = base["feedback"][2]
            flag_completed = base["eval_completed"]

            print(f"Relocate: Agent Feedback iter {counter} -- trial solved: {flag_trial} -- task solved: {flag_completed}")
            print("*" * 100)
            counter +=1