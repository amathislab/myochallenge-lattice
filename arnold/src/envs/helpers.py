import os
from envs.environment_factory import EnvironmentFactory
from models.monitors import MonitorTensor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from models.monitors import MonitorTensor
from stable_baselines3.common.vec_env import VecNormalize

'''
from envs.isaacgym_envs.envs.dummy_vecenv import MyDummyVecEnv



class VecEnvManager:
    """This class handles the creation and the deletion of a vecenv. This can be used
    to save memory when the environment does not need to be used."""

    def __init__(
        self,
        config,
        num_envs=1,
        load_path=None,
        checkpoint_num=None,
        tensorboard_log=None,
        using_isaac=False,
        using_tensor_buffer=False,
        seed=0,
    ):
        self.config = config
        self.num_envs = num_envs
        self.tensorboard_log = tensorboard_log
        self.using_isaac = using_isaac
        self.using_tensor_buffer = using_tensor_buffer
        self.seed = seed
        self.env_name = config["env_name"]
        if not using_isaac:
            # Create a VecNormalize without the venv inside it
            self.envs = create_vec_env(
                config,
                num_envs,
                load_path,
                checkpoint_num,
                tensorboard_log,
                using_isaac,
                using_tensor_buffer,
                seed,
            )
            del self.envs.venv
            self.envs.venv = None
        else:
            del self.envs
            self.envs = None

    def build_env(self):
        venv = make_parallel_envs(
            self.config,
            self.num_envs,
            tensorboard_log=self.tensorboard_log,
            using_isaac=self.using_isaac,
            using_tensor_buffer=self.using_tensor_buffer,
            seed=self.seed,
        )
        if self.using_isaac:
            self.envs = venv
        else:
            self.envs.venv = venv

    def delete_env(self):
        if self.using_isaac:
            del self.envs
            self.envs = None
        else:
            del self.envs.venv
            self.envs.venv = None

'''


# Function that creates and monitors vectorized environments:
def make_parallel_envs(
    env_config,
    num_env,
    start_index=0,
    tensorboard_log=None,
    using_isaac=False,
    using_tensor_buffer=False,
    seed=0,
    max_episode_steps=None
):
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(**env_config)
            env.seed(seed)
            if max_episode_steps is not None:
                env._max_episode_steps = max_episode_steps
            env = Monitor(env, tensorboard_log)
            return env

        return _thunk

    if using_isaac:
        env = EnvironmentFactory.create(**env_config)
        env.seed(seed)
        env = MonitorTensor(env, tensorboard_log)
        env = MyDummyVecEnv(env, using_tensor_buffer)
        return env
    else:
        return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


def create_vec_env(
    config,
    num_envs=1,
    load_path=None,
    checkpoint_num=None,
    tensorboard_log=None,
    using_isaac=False,
    using_tensor_buffer=False,
    seed=0,
    max_episode_steps=None
):
    envs = make_parallel_envs(
        config,
        num_envs,
        tensorboard_log=tensorboard_log,
        using_isaac=using_isaac,
        using_tensor_buffer=using_tensor_buffer,
        seed=seed,
        max_episode_steps=max_episode_steps
    )
    if not using_isaac:
        if load_path is None:
            print(f"Creating a new environment for {config['env_name']}")
            envs = VecNormalize(
                envs, norm_obs=True
            )
        else:
            env_path = os.path.join(
                load_path, f"{config['env_name']}_{checkpoint_num}.pkl"
            )
            if os.path.exists(env_path):
                envs = VecNormalize.load(env_path, envs)
            else:
                env_path = os.path.join(
                    load_path, f"rl_model_vecnormalize_{checkpoint_num}_steps.pkl"
                ) 
                if os.path.exists(env_path):
                    envs = VecNormalize.load(env_path, envs)     
                else:
                    print(f"Creating a new environment for {config['env_name']}")
                    envs = VecNormalize(
                        envs, norm_obs=True
                    )
    return envs
