import os
import shutil
import argparse
import torch.nn as nn
import json
import numpy as np
import glob
from definitions import ROOT_DIR, ENV_INFO
from metrics.custom_callbacks import TensorboardCallback
from train.trainer import SingleEnvTrainer
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.helpers import create_vec_env
from models.ppo.policies import LatticeRecurrentActorCriticPolicy
from main_dataset_recurrent_ppo import MODEL_PATTERN, get_number


parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--freq", type=int, default=1, help="SDE sample frequency")
parser.add_argument("--use_sde", action="store_true", default=False, help="Flag to use SDE")
parser.add_argument("--use_lattice", action="store_true", default=False, help="Flag to use Lattice")
parser.add_argument("--log_std_init", type=float, default=0.0, help="Initial log standard deviation")
parser.add_argument("--env_name",type=str,default="CustomChaseTag", help="Name of the environment",)
parser.add_argument("--load_path", type=str, help="Path to the experiment to load")
parser.add_argument("--checkpoint_num", type=int, default=None, help="Number of the checkpoint to load")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--device", type=str, default="cuda", help="Device, cuda or cpu")
parser.add_argument("--std_reg", type=float, default=0, help="Additional independent std for the multivariate gaussian (only for lattice)")
parser.add_argument("--num_steps", type=int, default=1_000_000_000,  help="Number of training steps once an environment is sampled")
parser.add_argument("--save_every", type=int, default=500_000, help="Save a checkpoint every N number of steps")
parser.add_argument("--batch_size", type=int, default=2048, help="Size of the minibatch")
parser.add_argument("--steps_per_env", type=int, default=512, help="Steps per environment")
parser.add_argument("--done_weight", type=float, default=0)
parser.add_argument("--act_reg_weight", type=float, default=0)
parser.add_argument("--lose_weight", type=float, default=0)
parser.add_argument("--sparse_weight", type=float, default=0)
parser.add_argument("--solved_weight", type=float, default=0)
parser.add_argument("--alive_weight", type=float, default=1)
parser.add_argument("--distance_weight", type=float, default=0)
parser.add_argument("--vel_reward_weight", type=float, default=0)
parser.add_argument("--cyclic_hip_weight", type=float, default=0)
parser.add_argument("--ref_rot_weight", type=float, default=0)
parser.add_argument("--joint_angle_rew_weight", type=float, default=0)
parser.add_argument("--early_solved_weight", type=float, default=0)
parser.add_argument("--joints_in_range_weight", type=float, default=0)
parser.add_argument("--feet_height_weight", type=float, default=0)
parser.add_argument("--gait_prod_weight", type=float, default=0)
parser.add_argument("--alternating_foot_weight", type=float, default=0)
parser.add_argument("--lateral_foot_position_weight", type=float, default=0)
parser.add_argument("--min_height", type=float, default=0.6)
parser.add_argument("--prob_fixed", type=float, default=1.0)
parser.add_argument("--prob_random", type=float, default=0.0)
parser.add_argument("--prob_moving", type=float, default=0.0)
parser.add_argument("--x_min", type=float, default=-5.0)
parser.add_argument("--x_max", type=float, default=5.0)
parser.add_argument("--y_min", type=float, default=-5.0)
parser.add_argument("--y_max", type=float, default=5.0)
parser.add_argument("--theta_min", type=float, default=-2 * np.pi)  # Weird but agrees with the source code of the library
parser.add_argument("--theta_max", type=float, default=2 * np.pi)
parser.add_argument("--agent_x_min", type=float, default=-5.0)
parser.add_argument("--agent_x_max", type=float, default=5.0)
parser.add_argument("--agent_y_min", type=float, default=-5.0)
parser.add_argument("--agent_y_max", type=float, default=5.0)
parser.add_argument("--agent_theta_min", type=float, default=0)
parser.add_argument("--agent_theta_max", type=float, default=2 * np.pi)
parser.add_argument("--min_spawn_distance", type=float, default=2)
parser.add_argument("--hip_period", type=float, default=100)
parser.add_argument("--out_suffix", type=str, default="", help="Suffix added to the experiment folder name")
parser.add_argument("--max_episode_steps", type=int, default=2000, help="Maximum episode duration")
parser.add_argument("--network_arch", type=int, nargs="*", default=[256, 256], help="Hidden layer size",)
parser.add_argument("--lstm_hidden_size", type=int, default=256, help="LSTM hidden layer size",)
parser.add_argument("--stop_on_win", action="store_true", default=False, help="Flag to stop when the target is reached")
parser.add_argument("--heel_pos_weight", type=float, default=0, help="Reward for heel position during gait")
parser.add_argument("--gait_stride_length", type=float, default=0.8, help="Target stride length (in meters)")
parser.add_argument("--opponent_speed", type=float, default=10.0, help="Speed magnitude of the opponent")
parser.add_argument("--gait_cadence", type=float, default=0.01, help="Target stride cadence (in strides per sim step)")
parser.add_argument("--target_speed", type=float, default=0, help="Target speed (in m per second), should match gait speed")
parser.add_argument("--include_hfield", action="store_true", default=False, help="Flag to include the height field in the observation")
parser.add_argument("--task_choice", type=str, default="CHASE", help="CHASE, EVADE or random")
parser.add_argument("--terrain", type=str, default="FLAT", help="FLAT or random")
parser.add_argument("--reset_type", type=str, default="init", help="init or random")
parser.add_argument("--hills_min", type=float, default=0.03, help="Min hills height")
parser.add_argument("--hills_max", type=float, default=0.23, help="Max hills height")
parser.add_argument("--rough_min", type=float, default=0.05, help="Min rough height")
parser.add_argument("--rough_max", type=float, default=0.1, help="Max rough height")
parser.add_argument("--relief_min", type=float, default=0.1, help="Min relief height")
parser.add_argument("--relief_max", type=float, default=0.3, help="Max relief height")
parser.add_argument("--traj_mode", type=str, default="opponent", help="Follow opponent or virtual_traj")


args = parser.parse_args()

if args.use_sde == False and args.freq > 1:
    raise ValueError("Cannot have sampling freq > 1 without sde")

TENSORBOARD_LOG = (
    os.path.join(ROOT_DIR, "output", "training", "ongoing",
    (f"{args.env_name}_seed_{args.seed}_x_{args.x_min}_{args.x_max}_y_{args.y_min}_{args.y_max}"
    f"_dist_{args.distance_weight}_hip_{args.cyclic_hip_weight}_period_{args.hip_period}"
    f"_alive_{args.alive_weight}_solved_{args.solved_weight}_early_solved_{args.early_solved_weight}"
    f"_joints_{args.joints_in_range_weight}_lose_{args.lose_weight}_ref_{args.ref_rot_weight}"
    f"_heel_{args.heel_pos_weight}_gait_l_{args.gait_stride_length}_gait_c_{args.gait_cadence}"
    f"_fix_{args.prob_fixed}_ran_{args.prob_random}_mov_{args.prob_moving}_traj_{args.traj_mode}{args.out_suffix}")
    )
)
if os.path.isdir(TENSORBOARD_LOG):
    # The folder already exists, then we resume the training if there are already checkpoints
    load_path = TENSORBOARD_LOG
    model_list = sorted(
        glob.glob(os.path.join(load_path, MODEL_PATTERN)),
        key=get_number,
    )
    checkpoints_list = [get_number(el) for el in model_list]
    if len(checkpoints_list) > 0:
        checkpoint = max(checkpoints_list)
    else:
        print(f"WARNING: No checkpoints at the given path {load_path}, let's see if there is a model at the cli path")
        load_path = args.load_path
        checkpoint = None
else:
    # There is no such training with this setup
    load_path = args.load_path
if load_path is not None:
    experiment_name = load_path.split("/")[-1]
    if args.checkpoint_num is None:
            # Get the list of checkpoints
            model_list = sorted(
                glob.glob(os.path.join(load_path, MODEL_PATTERN)),
                key=get_number,
            )
            checkpoints_list = [get_number(el) for el in model_list]
            if len(checkpoints_list) > 0:
                checkpoint = max(checkpoints_list)
            else:
                print(f"WARNING: No checkpoints in the given path {load_path}, starting a new training")
                checkpoint = None
    else:
        checkpoint = args.checkpoint_num
    if checkpoint is None:
        model_path = None
        print("Creating a new model")
    else:
        model_path = os.path.join(
            ROOT_DIR, load_path, f"rl_model_{checkpoint}_steps.zip"
        )
        print("loading model from ", model_path)
else:
    experiment_name = None
    model_path = None
    checkpoint = None

weighted_reward_keys = {
    "done": args.done_weight,
    "act_reg": args.act_reg_weight,
    "lose": args.lose_weight,
    "sparse": args.sparse_weight,
    "solved": args.solved_weight,
    "alive": args.alive_weight,
    "distance": args.distance_weight,
    "vel_reward": args.vel_reward_weight,
    "cyclic_hip": args.cyclic_hip_weight,
    "ref_rot": args.ref_rot_weight,
    "joint_angle_rew": args.joint_angle_rew_weight,
    "early_solved": args.early_solved_weight,
    "joints_in_range": args.joints_in_range_weight,
    "heel_pos": args.heel_pos_weight,
    "gait_prod": args.gait_prod_weight,
    "feet_height": args.feet_height_weight,
    "alternating_foot": args.alternating_foot_weight,
    "lateral_foot_position": args.lateral_foot_position_weight
}

obs_keys = [
    'internal_qpos',
    'internal_qvel',
    'grf',
    'torso_angle',
    'opponent_pose',
    'opponent_vel',
    'model_root_pos',
    'model_root_vel',
    'muscle_length',
    'muscle_velocity',
    'muscle_force',
]
if args.include_hfield:
    obs_keys.append("hfield")

env_config = {
    "env_name": args.env_name,
    "seed": args.seed,
    "obs_keys": obs_keys,
    "weighted_reward_keys": weighted_reward_keys,
    "min_height": args.min_height,
    "opponent_probabilities": [args.prob_fixed, args.prob_random, args.prob_moving],
    "reset_type": args.reset_type,
    "task_choice": args.task_choice,
    "hills_range": (args.hills_min, args.hills_max),
    "rough_range": (args.rough_min, args.rough_max),
    "relief_range": (args.relief_min, args.relief_max),
    "terrain": args.terrain,
    "stop_on_win": args.stop_on_win,
    "hip_period": args.hip_period,
    "opponent_x_range": (args.x_min, args.x_max),
    "opponent_y_range": (args.y_min, args.y_max),
    "opponent_orient_range": (args.theta_min, args.theta_max),
    "agent_x_range": (args.agent_x_min, args.agent_x_max),
    "agent_y_range": (args.agent_y_min, args.agent_y_max),
    "agent_orient_range": (args.agent_theta_min, args.agent_theta_max),
    "min_spawn_distance": args.min_spawn_distance,
    "gait_stride_length": args.gait_stride_length,
    "gait_cadence": args.gait_cadence,
    "opponent_speed": args.opponent_speed,
    "target_speed": args.target_speed,
    "traj_mode": args.traj_mode,
}

net_arch = [dict(pi=args.network_arch, vf=args.network_arch)]

model_config = dict(
    policy=LatticeRecurrentActorCriticPolicy,
    device=args.device,
    batch_size=args.batch_size,
    n_steps=args.steps_per_env,
    learning_rate=2e-05,
    ent_coef=1e-05,
    clip_range=0.2,
    gamma=0.99,
    gae_lambda=0.9,
    max_grad_norm=0.7,
    vf_coef=0.5,
    n_epochs=5,
    use_sde=args.use_sde,
    sde_sample_freq=args.freq,  # number of steps
    policy_kwargs=dict(
        use_lattice=args.use_lattice,
        use_expln=True,
        ortho_init=False,
        log_std_init=args.log_std_init,  # TODO: tune
        activation_fn=nn.ReLU,
        net_arch=net_arch,
        std_clip=(1e-3, 10),
        expln_eps=1e-6,
        full_std=False,
        std_reg=args.std_reg,
        lstm_hidden_size=args.lstm_hidden_size
    ),
)


if __name__ == "__main__":
    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)
    with open(os.path.join(TENSORBOARD_LOG, "args.json"), "w") as file:
        json.dump(args.__dict__, file, indent=4, default=lambda _: "<not serializable>")

    # Define the callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_every // args.num_envs, 1),
        save_path=TENSORBOARD_LOG,
        save_vecnormalize=True,
        verbose=1,
    )
    tensorboard_callback = TensorboardCallback(args.env_name, info_keywords=ENV_INFO[args.env_name])

    # Create the environment
    envs = create_vec_env(env_config, 
                          num_envs=args.num_envs, 
                          load_path=load_path, 
                          checkpoint_num=checkpoint,
                          tensorboard_log=TENSORBOARD_LOG,
                          seed=args.seed,
                          max_episode_steps=args.max_episode_steps)
    
    
    # Define trainer
    trainer = SingleEnvTrainer(
        algo="recurrent_ppo",
        envs=envs,
        env_config=env_config,
        load_model_path=model_path,
        log_dir=TENSORBOARD_LOG,
        model_config=model_config,
        callbacks=[checkpoint_callback, tensorboard_callback],
        timesteps=args.num_steps,
    )

    # Train agent
    trainer.train()
    trainer.save()
