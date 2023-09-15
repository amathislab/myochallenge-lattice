import os
import shutil
import argparse
import torch.nn as nn
import json
import numpy as np
from datetime import datetime
from datetime import datetime
from definitions import ROOT_DIR, ENV_INFO
from metrics.custom_callbacks import TensorboardCallback
from train.trainer import SingleEnvTrainer
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.helpers import create_vec_env
from models.ppo.policies import LatticeRecurrentActorCriticPolicy


parser = argparse.ArgumentParser(description="Main script to train an agent")

parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
parser.add_argument("--freq", type=int, default=1, help="SDE sample frequency")
parser.add_argument("--use_sde", action="store_true", default=False, help="Flag to use SDE")
parser.add_argument("--use_lattice", action="store_true", default=False, help="Flag to use Lattice")
parser.add_argument("--log_std_init", type=float, default=0.0, help="Initial log standard deviation")
parser.add_argument("--env_name",type=str,default="CustomChaseTag", help="Name of the environment",)
parser.add_argument("--load_path", type=str, help="Path to the experiment to load")
parser.add_argument("--checkpoint_num", type=int, default=0, help="Number of the checkpoint to load")
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
parser.add_argument("--hip_period", type=float, default=100)
parser.add_argument("--out_suffix", type=str, default="", help="Suffix added to the experiment folder name")


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


TENSORBOARD_LOG = (
    os.path.join(ROOT_DIR, "output", "training", now)
    + f"_{args.env_name}_sde_{args.use_sde}_lattice_{args.use_lattice}_freq_{args.freq}_log_std_init_{args.log_std_init}_ppo_seed_{args.seed}{args.out_suffix}"
)

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
    "joint_angle_rew": args.joint_angle_rew_weight
}


env_config = {
    "env_name": args.env_name,
    "seed": args.seed,
    "weighted_reward_keys": weighted_reward_keys,
    "min_height": args.min_height,
    "opponent_probabilities": [args.prob_fixed, args.prob_random, args.prob_moving],
    "stop_on_win": False,
    "hip_period": args.hip_period,
    "opponent_x_range": (args.x_min, args.x_max),
    "opponent_y_range": (args.y_min, args.y_max),
    "opponent_orient_range": (args.theta_min, args.theta_max)
}

net_arch = [dict(pi=[256, 256], vf=[256, 256])]

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
                          load_path=args.load_path, 
                          checkpoint_num=args.checkpoint_num, 
                          tensorboard_log=TENSORBOARD_LOG,
                          seed=args.seed)
    
    
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
