from isaacgym import gymtorch
from isaacgym import gymapi
from .base_shadow_hand import ShadowHandBase
from .base_motor_hand import update_cfg
from .env_mixins_tensor import TensorObsMixin, TensorObsEmbeddingMixin
from ..utils.isaac_util import to_torch, quat_from_euler_xyz, get_euler, quat_axis, quat_distance
from definitions import ROOT_DIR
import torch
import numpy as np
import os
from definitions import ROOT_DIR, ACT_KEY, OBJ_KEY, GOAL_KEY
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

class ShadowHandReorient(ShadowHandBase) :

    def __init__(self, cfg) :

        self.cfg = cfg
        self.movable_arm = cfg["env"]["movable_arm"]
        if not hasattr(self,"num_obs") :
            self.num_obs = (24 + 6*self.movable_arm) * 3 + 6
            self.num_actions = 24 + 6*self.movable_arm
            self.observation_space = Dict(
                {
                    ACT_KEY: Box(low=-np.inf, high=np.inf, shape=(3, 24 + 6*self.movable_arm)),
                    GOAL_KEY: Box(low=-np.inf, high=np.inf, shape=(1, 6)),
                }
            )
            self.action_space = Box(low=-1, high=1, shape=(self.num_actions,))
            self.obs_shape_dict = {
                ACT_KEY: (3, 24 + 6*self.movable_arm),
                GOAL_KEY: (1, 6),
            }
            self.max_episode_length = 1024

        super().__init__(cfg)

        self.object_positions = self.root_state_tensor[:, self.object_index, 0:3]
        self.object_orientations = self.root_state_tensor[:, self.object_index, 3:7]
        self.object_linvels = self.root_state_tensor[:, self.object_index, 7:10]
        self.object_angvels = self.root_state_tensor[:, self.object_index, 10:13]

        self.obs_buf_target = self.obs_buf[:, 3*(self.num_shadow_hand_dofs+6*self.movable_arm):].view(-1, 1, 6)

        self._generate_target()

    def _generate_target(self) :

        self.target_positions = torch.randn((self.num_envs, 3), device=self.device) * 0.1
        self.target_positions[:, 1] = torch.clamp(self.target_positions[:, 1] + 0.35, 0, 1)
        self.target_positions[:, 2] = torch.clamp(self.target_positions[:, 2] + 0.52, 0, 1)
        roll = torch.rand((self.num_envs,), device=self.device) * np.pi * 2
        pitch = torch.rand((self.num_envs,), device=self.device) * np.pi * 2
        yaw = torch.rand((self.num_envs,), device=self.device) * np.pi * 2
        self.target_orientations = quat_from_euler_xyz(roll, pitch, yaw)
    
    def _create_envs(self, num_envs, spacing, num_per_row) :
        """
        Create multiple parallel isaacgym environments

        Args:
            num_envs (int): The total number of environment 

            spacing (float): Specifies half the side length of the square area occupied by each environment

            num_per_row (int): Specify how many environments in a row
        """

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root =  os.path.join(ROOT_DIR, "src", "envs", "isaacgym_envs", "assets")
        shadow_hand_asset_file = "shadowhand_with_fingertips.urdf"
        object_asset_file = "cube.urdf"

        # Prepare hand asset options
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = not self.movable_arm
        hand_asset_options.collapse_fixed_joints = False
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 100
        hand_asset_options.linear_damping = 100
        if self.physics_engine == gymapi.SIM_PHYSX:
            hand_asset_options.use_physx_armature = True
        hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        # Prepare object asset options
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.flip_visual_attachments = False
        object_asset_options.fix_base_link = False
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.angular_damping = 0
        object_asset_options.linear_damping = 0
        if self.physics_engine == gymapi.SIM_PHYSX:
            hand_asset_options.use_physx_armature = True
        
        # Prepare the asset for the simulation
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, hand_asset_options)
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        # print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        # print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        # # Setup tendons
        # limit_stiffness = 30
        # t_damping = 0.1
        # relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        # tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)

        # for i in range(self.num_shadow_hand_tendons):
        #     for rt in relevant_tendons:
        #         if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
        #             tendon_props[i].limit_stiffness = limit_stiffness
        #             tendon_props[i].damping = t_damping
        
        # self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)

        # actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        # self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]

        # Setup shadow_hand pose
        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0, 0, 0.5)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(-np.pi/2, 0.0, 0.0)

        # Setup shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        # Setup object pose
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.35, 0.6)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        # self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # Setup sensors
        self.sensors = []
        sensor_pose = gymapi.Transform()
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)

        # Create env instances
        self.hand_index = None
        self.fingertip_indices = []
        self.envs = []
        self.shadow_hands = []
        self.objects = []
        self.hand_start_states = []
        self.object_start_states = []
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            # Create hand
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.object_start_states.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                            object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                            0, 0, 0, 0, 0, 0])
            # create fingertip force-torque sensors
            self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)
            self.objects.append(object_actor)

        self.hand_index = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_ENV)
        self.object_index = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_ENV)
        self.fingertip_indices = [
            self.gym.find_actor_rigid_body_index(env_ptr, shadow_hand_actor, tip_name, gymapi.DOMAIN_ENV)
                for tip_name in self.fingertips
        ]
        self.fingertip_indices = to_torch(self.fingertip_indices, dtype=torch.long, device=self.device)

    def _compute_observation(self) :
        """
        Compute observations and store them in self.obs_buf
        """

        super()._compute_observation()

        self.obs_buf_target[:, 0, 0:3] = self.target_positions
        self.obs_buf_target[:, 0, 3:6] = get_euler(self.target_orientations)

        dx = quat_axis(self.object_orientations, 0)[0]
        dy = quat_axis(self.object_orientations, 1)[0]
        dz = quat_axis(self.object_orientations, 2)[0]
        o = self.object_positions[0]

        # self._draw_line(o, o+dx*0.2, [1,0,0], True)
        # self._draw_line(o, o+dy*0.2, [0,1,0])
        # self._draw_line(o, o+dz*0.2, [0,0,1])

        dx = quat_axis(self.target_orientations, 0)[0]
        dy = quat_axis(self.target_orientations, 1)[0]
        dz = quat_axis(self.target_orientations, 2)[0]
        o = self.target_positions[0]

        # self._draw_line(o, o+dx*0.2, [1,0,0])
        # self._draw_line(o, o+dy*0.2, [0,1,0])
        # self._draw_line(o, o+dz*0.2, [0,0,1])

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

        self.rew_buf -= quat_distance(self.hand_orientations, self.target_orientations)
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
        
        self.object_positions[:, :] = self.saved_root_tensor[:, self.object_index, 0:3] + torch.randn((self.num_envs, 3), device=self.device) * 0.01
        roll = torch.rand((self.num_envs,), device=self.device) * np.pi * 2
        pitch = torch.rand((self.num_envs,), device=self.device) * np.pi * 2
        yaw = torch.rand((self.num_envs,), device=self.device) * np.pi * 2
        self.object_orientations[:, :] = quat_from_euler_xyz(roll, pitch, yaw)

        super().reset()

        self._generate_target()

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

class ShadowHandReorientEnv(ShadowHandReorient, TensorObsMixin, TensorObsEmbeddingMixin) :
    
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