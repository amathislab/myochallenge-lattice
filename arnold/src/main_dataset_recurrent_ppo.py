import argparse
import os
import json
import numpy as np
import pandas as pd
import glob
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from scipy.signal import savgol_filter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import Iterable
import skvideo
import platform


MODEL_PATTERN = "rl_model_*_steps.zip"
ENV_PATTERN = "rl_model_vecnormalize_*_steps.pkl"
TB_DIR_NAME = "RecurrentPPO_1"  # "RecurrentPPO_1", "SAC_1"
CKPT_CHOICE_CRITERION = "rollout/ep_rew_mean"  # "rollout/ep_rew_mean", "rollout/solved"
VIDEO_DIR = os.path.join(ROOT_DIR, "data", "videos")


def get_number(filename):
    return int(filename.split("_steps")[0].split("_")[-1])


def load_model(experiment_path, checkpoint_number=None, action_space=None, observation_space=None):
    custom_objects = {
        "learning_rate": lambda _: 0,
        "lr_schedule": lambda _: 0,
        "clip_range": lambda _: 0,
    }
    if action_space is not None:
        custom_objects["action_space"] = action_space
    if observation_space is not None:
        custom_objects["observation_space"] = observation_space
    if checkpoint_number is None:
        model_file = "best_model.zip"
    else:
        model_file = MODEL_PATTERN.replace("*", str(checkpoint_number))
    model_path = os.path.join(experiment_path, model_file)
    model = RecurrentPPO.load(model_path, custom_objects=custom_objects)
    return model


def load_vecnormalize(experiment_path, checkpoint_number, base_env):
    if checkpoint_number is None:
        env_file = "training_env.pkl"
    else:
        env_file = ENV_PATTERN.replace("*", str(checkpoint_number))
    env_path = os.path.join(experiment_path, env_file)
    venv = DummyVecEnv([lambda: base_env])
    print("env path", env_path)
    vecnormalize = VecNormalize.load(env_path, venv)
    return vecnormalize


def get_best_checkpoint(steps, rewards, checkpoints, verbose=1):
    # Lowpass filter the rewards to avoid choosing a checkpoint at a peak due to noise
    clean_rewards = savgol_filter(rewards, window_length=51, polyorder=3)
    steps = list(steps)
    # Get the list of the closest steps to the checkpoints and the corresponding rewards
    closest_step_list = [
        min(steps, key=lambda x: abs(x - ckpt)) for ckpt in checkpoints
    ]
    closest_reward_list = [
        clean_rewards[steps.index(closest_step)] for closest_step in closest_step_list
    ]
    reward_ckpt_max = max(closest_reward_list)
    step_ckpt_max_approx = closest_step_list[closest_reward_list.index(reward_ckpt_max)]
    step_ckpt_max = min(checkpoints, key=lambda x: abs(x - step_ckpt_max_approx))
    if verbose:
        print(
            "Best checkpoint:",
            step_ckpt_max,
            ", corresponding reward:",
            reward_ckpt_max,
        )
    return step_ckpt_max


def get_data_from_tb_log(path, y, x="step", tb_config=None):
    if tb_config is None:
        tb_config = {}

    event_acc = EventAccumulator(path, tb_config)
    event_acc.Reload()

    if not isinstance(y, Iterable) or isinstance(y, str):
        y = [y]

    out_dict = {}
    for attr_name in y:
        if attr_name in event_acc.Tags()["scalars"]:
            x_vals, y_vals = np.array(
                [(getattr(el, x), el.value) for el in event_acc.Scalars(attr_name)]
            ).T
            out_dict[attr_name] = (x_vals, y_vals)
        else:
            out_dict[attr_name] = None
    return out_dict


def get_experiment_data(tb_dir_path, attributes, tb_config=None):
    experiment_data = {}
    folder_content = os.listdir(tb_dir_path)
    assert len(folder_content) == 1
    tb_file_name = folder_content[0]
    tb_file_path = os.path.join(tb_dir_path, tb_file_name)
    data_dict = get_data_from_tb_log(tb_file_path, attributes, tb_config=tb_config)
    for key, values in data_dict.items():
        if values is not None:
            x_vals, y_vals = values
            experiment_data_el = experiment_data.get(key)
            if experiment_data_el is None:
                experiment_data[key] = {}
                experiment_data[key]["x"] = [x_vals]
                experiment_data[key]["y"] = [y_vals]
            else:
                experiment_data[key]["x"].append(x_vals)
                experiment_data[key]["y"].append(y_vals)
    return experiment_data


def main(args):
    if args.experiment_path is None:
        env = EnvironmentFactory.create(args.env_name)
        env.seed(args.seed)
        model = RecurrentPPO(policy="MlpLstmPolicy", env=env)
        venv = DummyVecEnv([lambda: env])
        vecnormalize = VecNormalize(venv)
        sde_sample_freq = 1
        use_latice = False
    else:
        config_path = os.path.join(args.experiment_path, "env_config.json")
        env_config = json.load(open(config_path, "r"))
        env = EnvironmentFactory.create(**env_config)
        env.seed(args.seed)
        if args.checkpoint is None:
            # First get the training data from the tensorboard log
            tb_dir_path = os.path.join(args.experiment_path, TB_DIR_NAME)
            experiment_data = get_experiment_data(tb_dir_path, CKPT_CHOICE_CRITERION)
            steps = experiment_data[CKPT_CHOICE_CRITERION]["x"][0]
            rewards = experiment_data[CKPT_CHOICE_CRITERION]["y"][0]

            # Get the list of checkpoints
            model_list = sorted(
                glob.glob(os.path.join(args.experiment_path, MODEL_PATTERN)),
                key=get_number,
            )
            checkpoints = [
                get_number(el)
                for el in model_list
                if get_number(el) < args.max_checkpoint
            ]
            if len(checkpoints):
                # Select the checkpoint corresponding to the best reward
                checkpoint = get_best_checkpoint(steps, rewards, checkpoints)
            else:
                checkpoint = None
        else:
            checkpoint = args.checkpoint
        model = load_model(args.experiment_path, checkpoint, action_space=env.action_space, observation_space=env.observation_space)
        vecnormalize = load_vecnormalize(args.experiment_path, checkpoint, env)
        model_config_path = os.path.join(args.experiment_path, "model_config.json")
        model_config = json.load(open(model_config_path, "r"))
        sde_sample_freq = model_config["sde_sample_freq"]
        use_latice = model_config["policy_kwargs"]["use_lattice"]

    # Collect rollouts and store them
    vecnormalize.training = False
    episode_data = []
    if args.render:
        env.mujoco_render_frames = True
    if args.save_video:
        env.mujoco_render_frames = False
        frames = []
    for i in range(args.num_episodes):
        lstm_states = None
        cum_rew = 0
        step = 0
        obs = env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        while not done:
            if args.render:
                env.sim.renderer.render_to_window()
            if args.save_video:
                curr_frame = env.sim.renderer.render_offscreen(
                        # cameras=[None],
                        width=640,
                        height=480,
                        camera_id=1,
                        device_id=0
                    )
                frames.append(curr_frame)
            if model.use_sde and not args.deterministic and step % sde_sample_freq == 0:
                model.policy.reset_noise()
            action, lstm_states = model.predict(
                vecnormalize.normalize_obs(obs),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=args.deterministic,
            )
            if use_latice:
                action_cov = (
                    model.policy.action_dist.distribution.covariance_matrix.squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                action_cov = np.diag(
                    model.policy.action_dist.distribution.variance.squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
            det_action, _ = model.predict(
                vecnormalize.normalize_obs(obs),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            next_obs, rewards, done, _ = env.step(action)
            episode_data.append(
                [
                    i,
                    step,
                    obs,
                    action,
                    rewards,
                    next_obs,
                    env.last_ctrl,
                    det_action,
                    action_cov,
                    env.rwd_dict,
                ]
            )
            obs = next_obs
            episode_starts = done
            cum_rew += rewards
            step += 1
        print("Episode", i, ", len:", step, ", cum rew: ", cum_rew)
        
    env.close()
    
    if args.save_video:
        if not os.path.exists(VIDEO_DIR):
            os.mkdir(VIDEO_DIR)
        file_name = os.path.join(VIDEO_DIR, "video.mp4")
        # check if the platform is OS -- make it compatible with quicktime
        if platform == "darwin":
            skvideo.io.vwrite(file_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
        else:
            skvideo.io.vwrite(file_name, np.asarray(frames))
        print("saved", file_name)
    if not args.no_save_df:
        df = pd.DataFrame(
            episode_data,
            columns=[
                "episode",
                "step",
                "observation",
                "action",
                "reward",
                "next_observation",
                "muscle_act",
                "action_mean",
                "action_cov",
                "rew_dict",
            ],
        )
        if args.deterministic:
            suffix = "_deterministic.h5"
        else:
            suffix = "_stochastic.h5"
        df.to_hdf(
            os.path.join(
                ROOT_DIR, "data", "rollouts", args.experiment_path.split("/")[-1] + suffix
            ),
            key="data",
        )
        print(
            "Saved to ",
            os.path.join(
                ROOT_DIR, "data", "rollouts", args.experiment_path.split("/")[-1] + suffix
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main script to create a dataset of episodes with a trained agent"
    )

    parser.add_argument(
        "--experiment_path",
        type=str,
        default=None,
        help="Path to the folder where the experiment results are stored",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Number of the checkpoint to select. Otherwise the checkpoint corresponding to the highest reward is selected.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CustomMyoReorientP2",
        help="Name of the environment where to test the agent",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Flag to use the deterministic policy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed of the environment.",
    )
    parser.add_argument(
        "--max_checkpoint",
        type=float,
        default=float("inf"),
        help="Do not consider checkpoints past this number (to be fair across trainings)",
    )
    parser.add_argument(
        "--no_save_df",
        action="store_true",
        default=False,
        help="Flag to not save the dataframe",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Flag to render at the screen",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        default=False,
        help="Flag to save a video",
    )
    args = parser.parse_args()
    main(args)
    
    """Example:
    mjpython src/main_dataset_recurrent_ppo.py --experiment_path=output/training/2023-09-13/16-09-28_CustomChaseTag_sde_False_lattice_True_freq_1_log_std_init_0.0_ppo_seed_0_xrange_-2_2_yrange_0_5 \
        --num_episodes=100 --no_save_df --render
        
    rsync -r -u -v chiappa@sv-rcp-gateway.intranet.epfl.ch:/storage-rcp-pure/upamathis_scratch/alberto/arnold/output/training/2023-09-13/16-09-28_CustomChaseTag_sde_False_lattice_True_freq_1_log_std_init_0.0_ppo_seed_0_xrange_-2_2_yrange_0_5 \
        /Users/albertochiappa/Dev/rl/arnold/output/training/2023-09-13
    """
