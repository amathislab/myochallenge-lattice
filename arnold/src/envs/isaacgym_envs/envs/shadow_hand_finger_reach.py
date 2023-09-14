import torch
import numpy as np
from isaacgym import gymtorch
from isaacgym import gymapi
from .base_motor_hand import BaseMotorHand
from .base_motor_hand import update_cfg
from .base_shadow_hand import ShadowHandBase
from .env_mixins_tensor import TensorObsMixin, TensorObsEmbeddingMixin
from ..utils.isaac_util import to_torch, scale, tensor_clamp, quat_apply, unscale, get_euler_xyz
from definitions import ROOT_DIR, ACT_KEY, OBJ_KEY, GOAL_KEY
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import os

class ShadowHandFingerReach(ShadowHandBase) :

    def __init__(self, cfg) :

        self.cfg = cfg
        self.movable_arm = cfg["env"]["movable_arm"]
        if not hasattr(self,"num_obs") :
            self.num_obs = (24 + 6*self.movable_arm) * 3 + 3*5
            self.num_actions = 24 + 6*self.movable_arm

            self.observation_space = Dict(
                {
                    ACT_KEY: Box(low=-np.inf, high=np.inf, shape=(3, 24 + 6*self.movable_arm)),
                    GOAL_KEY: Box(low=-np.inf, high=np.inf, shape=(1, 15)),
                }
            )
            self.action_space = Box(low=-1, high=1, shape=(self.num_actions,))
            self.obs_shape_dict = {
                ACT_KEY: (3, 24 + 6*self.movable_arm),
                GOAL_KEY: (1, 15),
            }

            self.max_episode_length = 512

        super().__init__(cfg)

        self.obs_buf_target = self.obs_buf[:, 3*(self.num_shadow_hand_dofs+6*self.movable_arm):].view(-1, 1, 15)
        asset_root =  os.path.join(ROOT_DIR, "src", "envs", "isaacgym_envs", "assets")
        target_asset_file = "fingertip_poses.pkl"
        self.possible_targets = torch.load(os.path.join(asset_root, target_asset_file), map_location=self.device)

        # self.target_positions = torch.randn((self.num_envs, 3), device=self.device) * 0.2
        # self.target_positions[:, 2] = torch.clamp(self.target_positions[:, 2] + 0.8, 0, 1)
        target_indices = torch.randint(0, self.possible_targets.shape[0], (self.num_envs,))
        self.target_positions = self.possible_targets[target_indices, :]

    def _compute_observation(self) :
        """
        Compute observations and store them in self.obs_buf
        """

        super()._compute_observation()

        self.obs_buf_target[:, 0, 0:3*5] = self.target_positions.view(-1, 15)

    def pre_physics_step(self, actions):
        """
        Apply actions on the shadow hand
        """

        super().pre_physics_step(actions)

    def _compute_reward(self, actions):
        """
        Compute the reward of all environments
        Save the reward in self.rew_buf
        """

        fingertip_pos = self.rigid_body_states[:, self.fingertip_indices, :3]
        # self._draw_line(fingertip_pos[0][0], self.target_positions[0][0], clear_lines=True)
        # self._draw_line(fingertip_pos[0][1], self.target_positions[0][1])
        # self._draw_line(fingertip_pos[0][2], self.target_positions[0][2])
        # self._draw_line(fingertip_pos[0][3], self.target_positions[0][3])
        # self._draw_line(fingertip_pos[0][4], self.target_positions[0][4])
        dist = fingertip_pos - self.target_positions
        self.rew_buf[:] = -torch.sum(torch.norm(dist, dim=2), dim=1)
    
    def post_physics_step(self):
        """
        Compute observation and reward after physics step
        """

        self._compute_observation()
        self._compute_reward(self.actions)
    
    def step(self, actions):
        super().step(actions)
        obs_dict = {
            "actuator_obs": self.obs_buf_joint,
            "goal_obs": self.obs_buf_target
        }
        reward = self.rew_buf
        done = self.reset_buf
        info = {}
        return obs_dict, reward, done, info
    
    def reset(self) :

        super().reset()

        target_indices = torch.randint(0, self.possible_targets.shape[0], (self.num_envs,))
        self.target_positions = self.possible_targets[target_indices, :]

        obs_dict = {
            "actuator_obs": self.obs_buf_joint,
            "goal_obs": self.obs_buf_target
        }

        return obs_dict

default_cfg = {
    "device_type": "cuda",
    "device_id": 0,
    "headless": False,
    "env": {
        "num_envs": 2,
        "movable_arm": False,
        "control_frequency_inv": 1,
        "env_spacing": 1.0,
        "actions_moving_average": 1.0,
        "transition_scale": 0.2,
        "orientation_scale": 1.0,
        "reset_dof_pos_noise": 0.1
    }
}

class ShadowHandFingerReachEnv(ShadowHandFingerReach, TensorObsMixin, TensorObsEmbeddingMixin) :
    
    def __init__(
        self,
        cfg=default_cfg,
        seed=42,
        include_adapt_state=False,
        num_memory_steps=30
    ) :
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        super().__init__(update_cfg(default_cfg, cfg))
        self._dict_obs_init_addon(include_adapt_state, num_memory_steps, self.obs_shape_dict, self.num_envs)
        self._obs_embedding_init_addon(self.obs_shape_dict, self.num_envs)
        pass

    def step(self, action) :

        obs, reward, done, info = super().step(action)
        obs = self.create_history_step_state(obs)
        obs = self.add_positions_to_obs(obs)
        return obs, reward, done, info

    def reset(self) :

        obs = super().reset()
        obs = self.create_history_reset_state(obs)
        obs = self.add_positions_to_obs(obs)
        return obs