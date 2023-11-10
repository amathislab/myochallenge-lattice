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
parser.add_argument("--env_name",type=str,default="CustomRelocateEnvPhase2", help="Name of the environment",)
parser.add_argument("--load_path", type=str, help="Path to the experiment to load")
parser.add_argument("--checkpoint_num", type=int, default=None, help="Number of the checkpoint to load")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--device", type=str, default="cuda", help="Device, cuda or cpu")
parser.add_argument("--std_reg", type=float, default=0, help="Additional independent std for the multivariate gaussian (only for lattice)")
parser.add_argument("--num_steps", type=int, default=1_000_000_000,  help="Number of training steps once an environment is sampled")
parser.add_argument("--save_every", type=int, default=500_000, help="Save a checkpoint every N number of steps")
parser.add_argument("--batch_size", type=int, default=2048, help="Size of the minibatch")
parser.add_argument("--steps_per_env", type=int, default=512, help="Steps per environment")
parser.add_argument("--done_weight", type=float, default=0)
parser.add_argument("--act_reg_weight", type=float, default=1e-3)
parser.add_argument("--alive_weight", type=float, default=0)
parser.add_argument("--sparse_weight", type=float, default=0)
parser.add_argument("--solved_weight", type=float, default=0)    #Part 3: obj in the obx- solved 1
parser.add_argument("--pos_dist_weight", type=float, default=0) #Part 3: obj in the obx- pos_dist 10
parser.add_argument("--rot_dist_weight", type=float, default=0)
parser.add_argument("--reach_dist_weight", type=float, default=0) #10
parser.add_argument("--reach_dist_xy_weight", type=float, default=0)
parser.add_argument("--reach_dist_z_weight", type=float, default=0)
parser.add_argument("--lift_bonus_weight", type=float, default=0)
parser.add_argument("--pos_dist_z_weight", type=float, default=0)
parser.add_argument("--max_app_weight", type=float, default=0)
parser.add_argument("--min_app_weight", type=float, default=0)
parser.add_argument("--contact_hand_obj_weight", type=float, default=0)
parser.add_argument("--rot_palm_obj_weight", type=float, default=0)
parser.add_argument("--close_bonus_weight", type=float, default=0)
parser.add_argument("--obj_shift_weight", type=float, default=0)
parser.add_argument("--palm_dist_weight", type=float, default=0)
parser.add_argument("--open_hand_weight", type=float, default=0)
parser.add_argument("--tip_dist_weight", type=float, default=0)

parser.add_argument("--obj_shift_x", type=float, default=0)
parser.add_argument("--obj_shift_y", type=float, default=0)
parser.add_argument("--obj_shift_z", type=float, default=0)

parser.add_argument("--pos_z_offset", type=float, default=0)
parser.add_argument("--reach_z_offset", type=float, default=0)
parser.add_argument("--pos_th", type=float, default=.075) #0.025
parser.add_argument("--rot_th", type=float, default=0.262)
parser.add_argument("--drop_th", type=float, default=0.50)
parser.add_argument("--lift_th", type=float, default=0.08)
parser.add_argument("--contact_th", type=float, default=0.01)

parser.add_argument("--x_min", type=float, default=0.0)
parser.add_argument("--x_max", type=float, default=0.3)
parser.add_argument("--y_min", type=float, default=-.45)
parser.add_argument("--y_max", type=float, default=-.1)
parser.add_argument("--z_min", type=float, default=0.9)
parser.add_argument("--z_max", type=float, default=1.05)

parser.add_argument("--x_min_rot", type=float, default=-.2)
parser.add_argument("--x_max_rot", type=float, default=0.2)
parser.add_argument("--y_min_rot", type=float, default=-.2)
parser.add_argument("--y_max_rot", type=float, default=0.2)
parser.add_argument("--z_min_rot", type=float, default=-.2)
parser.add_argument("--z_max_rot", type=float, default=0.2)

parser.add_argument("--out_suffix", type=str, default="", help="Suffix added to the experiment folder name")
parser.add_argument("--max_episode_steps", type=int, default=100, help="Maximum episode duration")
parser.add_argument("--network_arch", type=int, nargs="*", default=[256, 256], help="Hidden layer size",)

args = parser.parse_args()

if args.use_sde == False and args.freq > 1:
    raise ValueError("Cannot have sampling freq > 1 without sde")

TENSORBOARD_LOG = (
    os.path.join(ROOT_DIR, "output", "training_mani", "ongoing",
    f"_seed_{args.seed}_max_steps_{args.max_episode_steps}_reg_{args.act_reg_weight}_"
    f"_solved_{args.solved_weight}_pos_dist_{args.pos_dist_weight}_rot_dist_{args.rot_dist_weight}"
    f"_reach_dist_{args.reach_dist_weight}_lift_{args.lift_bonus_weight}"
    f"_max_app_{args.max_app_weight}_reach_z_{args.reach_z_offset}{args.out_suffix}"
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
    print("Checkpoints list: ", checkpoints_list)
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
    "sparse": args.sparse_weight,
    "solved": args.solved_weight,
    "alive": args.alive_weight,
    "pos_dist": args.pos_dist_weight,
    "rot_dist": args.rot_dist_weight,
    "reach_dist": args.reach_dist_weight,
    "reach_dist_xy": args.reach_dist_xy_weight,
    "reach_dist_z": args.reach_dist_z_weight,
    "lift_bonus": args.lift_bonus_weight,
    "pos_dist_z": args.pos_dist_z_weight,
    "max_app": args.max_app_weight,
    "contact_hand_obj": args.contact_hand_obj_weight,
    "rot_palm_obj": args.rot_palm_obj_weight,
    "min_app": args.min_app_weight,
    "close_bonus" : args.close_bonus_weight,
    "obj_shift": args.obj_shift_weight,
    "palm_dist": args.palm_dist_weight,
    "open_hand": args.open_hand_weight,
    "tip_dist": args.tip_dist_weight
}

env_config = {
    "env_name": args.env_name,
    "seed": args.seed,
    "weighted_reward_keys": weighted_reward_keys,
    "pos_th": args.pos_th,
    "rot_th": args.rot_th,
    "drop_th": args.drop_th,
    "lift_th": args.lift_th,
    "reach_z_offset": args.reach_z_offset,
    'pos_z_offset': args.pos_z_offset,
    "target_xyz_range": {'high':[args.x_max, args.y_max, args.z_max], 'low':[args.x_min, args.y_min, args.z_min]}, # args.target_xyz_range,
    "target_rxryrz_range": {'high':[args.x_max_rot, args.y_max_rot, args.z_max_rot], 'low':[args.x_min_rot, args.y_min_rot, args.z_min_rot]},
    "obj_rel_target_pos": (args.obj_shift_x, args.obj_shift_y, args.obj_shift_z),
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
    gamma=0.9999,
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
    ),
)

if __name__ == "__main__":
    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)
    shutil.copy(os.path.abspath(os.path.join(ROOT_DIR, "src", "envs", "relocate.py")), TENSORBOARD_LOG)
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
