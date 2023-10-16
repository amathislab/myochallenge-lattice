import os
import shutil
import argparse
import torch.nn as nn
import json
import numpy as np
from datetime import datetime
from definitions import ROOT_DIR, ENV_INFO
from metrics.custom_callbacks import TensorboardCallback
from train.trainer import SingleEnvTrainer
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.helpers import create_vec_env
from models.ppo.policies import LatticeRecurrentActorCriticPolicy
from envs.environment_factory import EnvironmentFactory
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO
import time

parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--freq", type=int, default=1, help="SDE sample frequency")
parser.add_argument("--use_sde", action="store_true", default=False, help="Flag to use SDE")
parser.add_argument("--use_lattice", action="store_true", default=False, help="Flag to use Lattice")
parser.add_argument("--log_std_init", type=float, default=0.0, help="Initial log standard deviation")
parser.add_argument("--env_name",type=str,default="CustomRelocateEnv", help="Name of the environment",)
parser.add_argument("--load_path", type=str, help="Path to the experiment to load")
parser.add_argument("--checkpoint_num", type=int, default=6499584, help="Number of the checkpoint to load") #1499904 7999488  17998848   2500000  1999872
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--device", type=str, default="cuda", help="Device, cuda or cpu")
parser.add_argument("--std_reg", type=float, default=0, help="Additional independent std for the multivariate gaussian (only for lattice)")
parser.add_argument("--num_steps", type=int, default=1_000_000_000,  help="Number of training steps once an environment is sampled")
parser.add_argument("--save_every", type=int, default=500_000, help="Save a checkpoint every N number of steps")
parser.add_argument("--batch_size", type=int, default=2048, help="Size of the minibatch")
parser.add_argument("--steps_per_env", type=int, default=512, help="Steps per environment")
parser.add_argument("--done_weight", type=float, default=0)
parser.add_argument("--act_reg_weight", type=float, default=0)
parser.add_argument("--alive_weight", type=float, default=1)
# parser.add_argument("--lose_weight", type=float, default=0)
parser.add_argument("--sparse_weight", type=float, default=0)
parser.add_argument("--solved_weight", type=float, default=0)
# parser.add_argument("--alive_weight", type=float, default=1)
parser.add_argument("--pos_dist_weight", type=float, default=1)
parser.add_argument("--rot_dist_weight", type=float, default=1)
parser.add_argument("--lift_bonus_weight", type=float, default=1)

# "pos_th": args.pos_th,
# "rot_th": args.rot_th,
# "drop_th": args.drop_th,

parser.add_argument("--pos_th", type=float, default=.025)
parser.add_argument("--rot_th", type=float, default=0.262)
parser.add_argument("--drop_th", type=float, default=0.50)
parser.add_argument("--lift_th", type=float, default=0.02)

parser.add_argument("--x_min", type=float, default=0.0)
parser.add_argument("--x_max", type=float, default=0.2)
parser.add_argument("--y_min", type=float, default=-.1)
parser.add_argument("--y_max", type=float, default=-.35)
parser.add_argument("--z_min", type=float, default=0.9)
parser.add_argument("--z_max", type=float, default=0.9)

parser.add_argument("--x_min_rot", type=float, default=0.0)
parser.add_argument("--x_max_rot", type=float, default=0.0)
parser.add_argument("--y_min_rot", type=float, default=0.0)
parser.add_argument("--y_max_rot", type=float, default=0.0)
parser.add_argument("--z_min_rot", type=float, default=0.0)
parser.add_argument("--z_max_rot", type=float, default=0.0)
parser.add_argument("--out_suffix", type=str, default="", help="Suffix added to the experiment folder name")
parser.add_argument("--max_episode_steps", type=int, default=100, help="Maximum episode duration")
parser.add_argument("--network_arch", type=int, nargs="*", default=[256, 256], help="Hidden layer size",)

args = parser.parse_args()

if args.use_sde == False and args.freq > 1:
    raise ValueError("Cannot have sampling freq > 1 without sde")

now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

if args.load_path is not None:
    experiment_name = args.load_path.split("/")[-1]
    model_path = os.path.join(
        ROOT_DIR, args.load_path, f"rl_model_{args.checkpoint_num}_steps"
    )
else:
    experiment_name = None
    model_path = None


weighted_reward_keys = {
    "done": args.done_weight,
    "act_reg": args.act_reg_weight,
    "sparse": args.sparse_weight,
    "solved": args.solved_weight,
    "alive": args.alive_weight,
    "pos_dist": args.pos_dist_weight,
    "rot_dist": args.rot_dist_weight,
    "lift_bonus": args.lift_bonus_weight,
}

# Reward structure and task parameters:
env_config = {
    "env_name": args.env_name,
    "seed": args.seed,
    "weighted_reward_keys": weighted_reward_keys,
    "pos_th": args.pos_th,
    "rot_th": args.rot_th,
    "drop_th": args.drop_th,
    "lift_th": args.lift_th,
    "target_xyz_range": {'high':[args.x_max, args.y_max, args.z_max], 'low':[args.x_min, args.y_min, args.z_min]}, # args.target_xyz_range,
    "target_rxryrz_range": {'high':[args.x_max_rot, args.y_max_rot, args.z_max_rot], 'low':[args.x_min_rot, args.y_min_rot, args.z_min_rot]},
}

# Function that creates and monitors vectorized environments:
def make_parallel_envs(
    env_name, env_config, num_env, start_index=0
):  # pylint: disable=redefined-outer-name
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(env_name, **env_config)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

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

max_episode_steps = 100  # default: 100
num_envs = 1  # 16 for training, fewer for debugging
num_episodes = 1000
render = False
SAVE_DIR = '/media2/data/alessandro/goodman_data/'
# FOLDER_TO_SAVE = 'rl_activation_as_goodman_w_mlp'

# output/training/2023-09-20/00-02-39_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0

if __name__ == "__main__":

    checkpoint_num = args.checkpoint_num

    # load_path = '2023-09-18/15-59-30_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0'
    # PATH_TO_AGENT = os.path.join(
    #      SAVE_DIR + "/output/training",
    #     load_path)

    # load_path = '2023-09-20/00-02-39_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0'
    # load_path = '2023-09-20/08-27-07_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0'
    # load_path = '2023-09-20/10-44-17_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0'
    
    ## Move object z distance
    # load_path = '2023-09-20/14-12-39_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0'
    # checkpoint_num = 34997760

    ## Keep reach dist min
    # load_path = '2023-09-20/19-29-09_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0'

    ## Max aperture and reach dist min
    # checkpoint_num = 19998720
    # load_path = '2023-09-20/23-46-01_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0'

    ## Max aperture and reach closer with contact? - starting from2023-09-20/23-46-01
    # load_path = "2023-09-21/10-39-33_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 2999808

    ## Max aperture and reach closer - starting from2023-09-20/23-46-01
    # load_path = "2023-09-21/10-41-36_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 499968 #1999872

    ## Max aperture, close and palm rotation - starting from /// nothing is bad
    # load_path = "2023-09-21/12-09-37_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 499968

    ## 
    # load_path = "2023-09-21/12-38-30_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 1999872

    ## Min aperture, grasp, pos z - starting from 2023-09-21/10-41-36_
    # load_path = "2023-09-21/14-09-21_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 12499200

    ## Min aperture, grasp, pos z - starting from 2023-09-21/10-41-36_ first check
    # load_path = "2023-09-21/21-39-16_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 3499776

    ## From scratch - max app and min app based on reach dist and then pos z
    # load_path = "2023-09-21/21-46-24_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 2999808

    ## From max ape
    # load_path = "2023-09-21/21-39-16_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 19998720

    ## From scratch - max app and min app based on reach dist (between) and then pos z
    # load_path = "2023-09-22/00-02-14_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 19998720

    ## From scratch - max app and min app based on reach dist (between) and then pos z
    # load_path = "2023-09-22/11-54-21_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 72495360

    # load_path = "2023-09-22/11-54-31_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 72495360

    # load_path = "2023-09-27/11-08-30_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 30998016

    ## Grasping object
    # load_path = "2023-10-04/00-30-35_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 56496384

    ## Object in the box
    load_path = "2023-10-05/01-08-55_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    checkpoint_num = 334478592 #131491584 #91994112 #61996032 #52996608 #26998272 #37997568 #26998272 #34997760

    # Object in the box with solved
    # load_path = "2023-10-06/10-36-17_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0"
    # checkpoint_num = 219485952 #37497600 #91994112 #61996032 #52996608 #26998272 #37997568 #26998272 #34997760

    # PATH_TO_AGENT = os.path.join(
    #     # ROOT_DIR,
    #     SAVE_DIR + "output/training",
    #     "2023-09-18/15-59-30_CustomRelocateEnv_sde_False_lattice_False_freq_1_log_std_init_0.0_ppo_seed_0/")

    PATH_TO_AGENT = os.path.join(
         ROOT_DIR + "/output/training",
        load_path)
    
    PATH_TO_PRETRAINED_NET = os.path.join(PATH_TO_AGENT,'rl_model_'+str(checkpoint_num)+'_steps.zip')

    PATH_TO_NORMALIZED_ENV = os.path.join(PATH_TO_AGENT,'rl_model_vecnormalize_'+str(checkpoint_num)+'_steps.pkl')
    
    env_name = args.env_name
    policy_type = 'lstm'

    # Create vectorized environments:
    # envs = make_parallel_envs(env_name, env_config, num_env=1)
    # Create the environment
    envs = create_vec_env(env_config, 
                          num_envs=num_envs, 
                          load_path=load_path, 
                          checkpoint_num=checkpoint_num, 
                          tensorboard_log=None,
                          seed=args.seed,
                          max_episode_steps=args.max_episode_steps)

    # Normalize environment:
    if PATH_TO_NORMALIZED_ENV is not None:
        envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)
    else:
        envs = VecNormalize(envs)
    envs.training = False
    envs.norm_reward = False

    # Create model
    custom_objects = {
        "learning_rate": lambda _: 0,
        "lr_schedule": lambda _: 0,
        "clip_range": lambda _: 0,
    }
    if PATH_TO_PRETRAINED_NET is not None:
        model = RecurrentPPO.load(
            PATH_TO_PRETRAINED_NET,
            env=envs,
            device="cpu",
            custom_objects=custom_objects,
        )
    else:
        print('Add pretrained net path')
        # model = RecurrentPPO(
        #     "MultiInputPolicy",
        #     env=envs,
        #     verbose=2,
        #     **model_config,
        # )

    # EVALUATE
    eval_model = model
    # eval_env = EnvironmentFactory.create(env_name, **env_config)
    eval_env = EnvironmentFactory.create(**env_config)

    # Enjoy trained agent
    perfs = []
    lens = []
    for i in range(num_episodes):
        lstm_states = None
        cum_rew = 0
        step = 0
        # eval_env.reset()
        # eval_env.step(np.zeros(39))
        obs = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        print(eval_env.obs_dict['goal_pos'])
        # if render:
        #     eval_env.sim.render(mode="window")
            # time.sleep(0)
        # while not done:
        for step in range(1,500):
            # time.sleep(0.5)
            if render:
                eval_env.mj_render()
                # eval_env.sim.render(mode="window")
            action, lstm_states = eval_model.predict(
                envs.normalize_obs(obs),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, rewards, done, info = eval_env.step(action)
            if step == 0:
                print(info["rwd_dict"])
            episode_starts = done
            cum_rew += rewards
            step += 1
        lens.append(step)
        perfs.append(cum_rew)
        print('Final obj pose: ',eval_env.obs_dict['obj_pos'])
        print('time: ',eval_env.obs_dict['time'])
        print('Error: ',np.abs(np.linalg.norm(eval_env.obs_dict['pos_err'], axis=-1)))
        print('Error rot: ',np.abs(np.linalg.norm(eval_env.obs_dict['rot_err'], axis=-1)))
        print("Episode", i, ", len:", step, ", cum rew: ", cum_rew)
        # if step == 0:
        print(info["rwd_dict"])
        print('***********')

        if (i + 1) % 10 == 0:
            len_error = np.std(lens) / np.sqrt(i + 1)
            perf_error = np.std(perfs) / np.sqrt(i + 1)

            print(f"\nEpisode {i+1}/{num_episodes}")
            print(f"Average len: {np.mean(lens):.2f} +/- {len_error:.2f}")
            print(f"Average rew: {np.mean(perfs):.2f} +/- {perf_error:.2f}\n")

    print(f"\nFinished evaluating {PATH_TO_PRETRAINED_NET}!")


















# def main(args):

#     PATH_TO_AGENT = os.path.join(
#         # ROOT_DIR,
#         PATH_TO_DATA + "/output/training",
#         "2023-08-08/12-37-47_hand_traj_hand_grasp_as_goodman_muscle_transformer_sde_False_latice_False_freq_1_log_std_init_0.0_std_reg_0.001_recurrent_ppo_seed_0_resume_None/")

#     env_name = 'MusclePoseTrajEnv-v5' #MusclePoseGraspAvg-v2  MusclePoseTrajEnv-v2
#     task = 'traj_hand'
#     policy_type = 'muscle_transformer'

#     index = args.index #0
#     beh_input = args.beh_input
#     save_act = args.save_act
#     untrained = args.untrained
#     num_episodes = 100
#     render = True
#     verbose = True

#     path_to_beh_stimuli = os.path.join(PATH_TO_DATA,'converted_behavioral_stimuli_as_goodman')

#     eval_policy = EvalActivationPolicyTransformerTrajHand(algo='ppo',env_name=env_name,path_to_agent=PATH_TO_AGENT,config=config,device='cpu',seed=0,untrained=untrained,beh_input=beh_input,save_act=save_act)

#     ## Load behavioral data
#     beh_stimuli_data, target_jnt_dict, monkey_name, monkey_session = eval_policy.load_behavioral_data(index=index,path_to_beh_stimuli=path_to_beh_stimuli)

#     ## Define path to save activation
#     path_to_save = os.path.join(PATH_TO_DATA,FOLDER_TO_SAVE)
#     path_to_save = eval_policy.define_save_path(task,policy_type,path_to_save=path_to_save)
    
#     filename = monkey_name + '_' + str(monkey_session) + '_act.h5'
#     path_to_save_act = os.path.join(path_to_save,filename)

#     trial_list = list(beh_stimuli_data.keys())

#     # Enjoy trained agent
#     perfs = []
#     lens = []
#     all_solved = []
#     all_activation = {}
#     all_beh_metric = {}
#     all_beh_metric['joint_pos_err'] = []
#     all_beh_metric['joint_vel_err'] = []
#     all_beh_metric['cum_rew'] = []
#     all_beh_metric['solved'] = []

#     for i in trial_list:
#         all_activation[i] = {}
#     # for i in range(num_episodes):

#         cum_rew, solved, step, trial_act, trial_beh_error = eval_policy.rollout_episode(beh_stimuli_data, target_jnt_dict, trial_idx=i, render=render)
        
#         all_activation[i] = trial_act
#         all_beh_metric['joint_pos_err'].append(trial_beh_error['joint_pos'].mean())
#         all_beh_metric['joint_vel_err'].append(trial_beh_error['joint_vel'].mean())
#         all_beh_metric['cum_rew'].append(cum_rew)
#         all_beh_metric['solved'].append(solved*100/step)

#         if verbose:
#             lens.append(step)
#             perfs.append(cum_rew)
#             all_solved.append(solved*100/step)
#             print("Episode", i, ", len:", step, ", cum rew: ", cum_rew, ", solved:", solved*100/step)

#             print('Behavioral error: pos {:.2f} - vel {:.2f}'.format(trial_beh_error['joint_pos'].sum()/len(trial_beh_error['joint_pos']), trial_beh_error['joint_vel'].sum()/len(trial_beh_error['joint_vel'])))

#             if (i + 1) % 10 == 0:
#                 len_error = np.std(lens) / np.sqrt(i + 1)
#                 perf_error = np.std(perfs) / np.sqrt(i + 1)
#                 solv_error = np.std(all_solved) / np.sqrt(i + 1)

#                 print(f"\nEpisode {i+1}/{num_episodes}")
#                 print(f"Average len: {np.mean(lens):.2f} +/- {len_error:.2f}")
#                 print(f"Average rew: {np.mean(perfs):.2f} +/- {perf_error:.2f}\n")
#                 print(f"Average solved: {np.mean(all_solved):.2f} +/- {solv_error:.2f}\n")
