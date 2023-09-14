from .base_shadow_hand import ShadowHandBase
from .base_motor_hand import update_cfg
from .env_mixins_tensor import TensorObsMixin, TensorObsEmbeddingMixin
from ..utils.isaac_util import to_torch, scale, tensor_clamp, quat_apply, unscale, get_euler_xyz
import torch
from definitions import ROOT_DIR, ACT_KEY, OBJ_KEY, GOAL_KEY
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import os
import numpy as np

class ShadowHandReach(ShadowHandBase) :

    def __init__(self, cfg) :

        self.cfg = cfg
        self.movable_arm = cfg["env"]["movable_arm"]
        if not hasattr(self,"num_obs") :
            self.num_obs = (24 + 6*self.movable_arm) * 3 + 3
            self.num_actions = 24 + 6*self.movable_arm
            self.observation_space = Dict(
                {
                    ACT_KEY: Box(low=-np.inf, high=np.inf, shape=(3, 24 + 6*self.movable_arm)),
                    GOAL_KEY: Box(low=-np.inf, high=np.inf, shape=(1, 3)),
                }
            )
            self.action_space = Box(low=-1, high=1, shape=(self.num_actions,))
            self.obs_shape_dict = {
                ACT_KEY: (3, 24 + 6*self.movable_arm),
                GOAL_KEY: (1, 3),
            }
            self.max_episode_length = 256

        super().__init__(cfg)

        self.obs_buf_target = self.obs_buf[:, 3*(self.num_shadow_hand_dofs+6*self.movable_arm):].view(-1, 1, 3)

        self.target_positions = torch.randn((self.num_envs, 3), device=self.device) * 0.1
        self.target_positions[:, 2] = torch.clamp(self.target_positions[:, 2] + 0.5, 0, 1)

    def _compute_observation(self) :
        """
        Compute observations and store them in self.obs_buf
        """

        super()._compute_observation()

        self.obs_buf_target[:, 0, 0:3] = self.target_positions

        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)

        # self.right_hand_pos = self.rigid_body_states[:, 3, 0:3]
        # self.right_hand_rot = self.rigid_body_states[:, 3, 3:7]
        # # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        # # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        # self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        # self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
        #                                                     self.shadow_hand_dof_lower_limits.view(1, -1), self.shadow_hand_dof_upper_limits.view(1, -1))
        # self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        # self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

        # fingertip_obs_start = 72
        # num_ft_states = 13 * self.num_fingertipss
        # num_ft_force_torques = 6 * self.num_fingertips
        # self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        # self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
        #             num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor
    
        # hand_pose_start = fingertip_obs_start + 95
        # self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.hand_positions[self.hand_indices, :]
        # self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        # self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        # self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        # action_obs_start = hand_pose_start + 6
        # self.obs_buf[:, action_obs_start:action_obs_start + 26] = self.actions[:, :26]

        # target_obs_start = action_obs_start + 26
        # self.obs_buf[:, target_obs_start:target_obs_start + 3] = self.target_positions

    def pre_physics_step(self, actions):
        """
        Apply actions on the shadow hand
        """

        super().pre_physics_step(actions)

        # self.cur_targets[:, self.actuated_dof_indices] = scale(
        #     self.actions[:, 6:26],
        #     self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
        #     self.shadow_hand_dof_upper_limits[self.actuated_dof_indices]
        # )
        # self.cur_targets[:, self.actuated_dof_indices] = (
        #     self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
        #     + (1.0 - self.act_moving_average) * self.pre_targets[:, self.actuated_dof_indices]
        # )
        # self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
        #     self.cur_targets[:, self.actuated_dof_indices],
        #     self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
        #     self.shadow_hand_dof_upper_limits[self.actuated_dof_indices]
        # )
        # self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
        # self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000

        # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces), gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # self.pre_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
    
    def _compute_reward(self, actions):
        """
        Compute the reward of all environments
        Save the reward in self.rew_buf
        """

        cur_positions = self.hand_positions[:, :]
        diff = cur_positions - self.target_positions
        self.rew_buf[:] = -torch.norm(diff, dim=1)
        # self._draw_line(cur_positions[0], self.target_positions[0])



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

        self.target_positions = torch.randn((self.num_envs, 3), device=self.device) * 0.1
        self.target_positions[:, 2] = torch.clamp(self.target_positions[:, 2] + 0.5, 0, 1)

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
        "movable_arm": True,
        "control_frequency_inv": 1,
        "env_spacing": 1.0,
        "actions_moving_average": 1.0,
        "transition_scale": 0.2,
        "orientation_scale": 1.0,
        "reset_dof_pos_noise": 0.1
    }
}

class ShadowHandReachEnv(ShadowHandReach, TensorObsMixin, TensorObsEmbeddingMixin) :
    
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