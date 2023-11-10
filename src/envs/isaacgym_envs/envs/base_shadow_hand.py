from isaacgym import gymtorch
from isaacgym import gymapi
from .base_motor_hand import BaseMotorHand
from ..utils.isaac_util import to_torch, scale, tensor_clamp, quat_apply, unscale, get_euler
import os
import torch
import numpy as np
from definitions import ROOT_DIR, ACT_KEY, OBJ_KEY, GOAL_KEY
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

class ShadowHandBase(BaseMotorHand) :

    def __init__(self, cfg) :

        self.cfg = cfg
        self.fingertips = ["fftip", "mftip", "rftip", "lftip", "thtip"]
        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]
        self.act_moving_average = self.cfg["env"]["actions_moving_average"]
        self.reset_dof_pos_noise = self.cfg["env"]["reset_dof_pos_noise"]
        self.up_axis = 'z'
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.num_fingertips = len(self.fingertips)
        self.movable_arm = cfg["env"]["movable_arm"]
        if not hasattr(self,"num_obs") :
            self.num_obs = (24 + 6*self.movable_arm) * 3
            self.num_actions = 24 + 6*self.movable_arm
            self.observation_space = Dict(
                {
                    ACT_KEY: Box(low=-1, high=1, shape=(3, 24 + 6*self.movable_arm)),
                }
            )
            self.action_space = Box(low=-1, high=1, shape=(self.num_actions,))
            self.obs_shape_dict = {
                ACT_KEY: (3, 24 + 6*self.movable_arm),
            }
            self.max_episode_length = 128
        
        # Following are data required in gym
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}
        self.spec = None

        super().__init__(cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
        # Acquiring tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        # Making handy slices
        self.hand_positions = self.root_state_tensor[:, self.hand_index, 0:3]
        self.hand_orientations = self.root_state_tensor[:, self.hand_index, 3:7]
        self.hand_linvels = self.root_state_tensor[:, self.hand_index, 7:10]
        self.hand_angvels = self.root_state_tensor[:, self.hand_index, 10:13]
        self.shadow_hand_dof_state = self.dof_state[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        self.saved_root_tensor = self.root_state_tensor.clone()

        print("dof_force_tensor.shape: ", self.dof_force_tensor.shape)
        print("rigid_body_states.shape: ", self.rigid_body_states.shape)
        print("root_state_tensor.shape: ", self.root_state_tensor.shape)
        print("shadow_hand_dof_state.shape: ", self.shadow_hand_dof_state.shape)
        print("vec_sensor_tensor.shape: ", self.vec_sensor_tensor.shape)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.num_bodies = self.rigid_body_states.shape[1]
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.pre_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        # Handy observation slices
        self.obs_buf_joint = self.obs_buf[:, :3*(self.num_shadow_hand_dofs+6*self.movable_arm)].view(-1, 3, self.num_shadow_hand_dofs+6*self.movable_arm)

        # Required in mixins
        self.obs_positions = self._get_obs_elements()

    def _create_ground(self) :
        """
        Adds ground plane to simulation
        """

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        pass
    
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

        # Prepare asset options
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = not self.movable_arm
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        # Prepare the asset for the simulation
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        # self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        # self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

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
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0.3925, -1.57)

        # Setup shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        # shadow_hand_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)

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
        self.hand_start_states = []
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            # Create hand
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            # create fingertip force-torque sensors
            self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        self.hand_index = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_ENV)
        self.fingertip_indices = [
            self.gym.find_actor_rigid_body_index(env_ptr, shadow_hand_actor, tip_name, gymapi.DOMAIN_ENV)
                for tip_name in self.fingertips
        ]
        self.fingertip_indices = to_torch(self.fingertip_indices, dtype=torch.long, device=self.device)

    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(compute_device, graphics_device, physics_engine, sim_params)
        self._create_ground()
        self._create_envs(self.num_envs, self.cfg["env"]['env_spacing'], int(np.sqrt(self.num_envs)))
        return self.sim
    
    def _compute_observation(self) :
        """
        Compute observations and store them in self.obs_buf
        The observation space is 4*(24+7)-dimentional:

        Index   Description
        0       Shadow hand DOF positions
        1       Shadow hand DOF velocities
        2       Shadow hand DOF forces
        3       Shadow hand last action
        """

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf_joint[:, 0, :self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                self.shadow_hand_dof_lower_limits.view(1, -1), self.shadow_hand_dof_upper_limits.view(1, -1))
        # self.obs_buf_joint[:, 1, :self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf_joint[:, 1, :self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor
        self.obs_buf_joint[:, 2, :self.num_shadow_hand_dofs] = self.actions[:, :self.num_shadow_hand_dofs]

        # TODO: Check if this is correct
        if self.movable_arm :
            self.obs_buf_joint[:, 0, self.num_shadow_hand_dofs+0:self.num_shadow_hand_dofs+3] = self.hand_positions
            self.obs_buf_joint[:, 0, self.num_shadow_hand_dofs+3:self.num_shadow_hand_dofs+6] = get_euler(self.hand_orientations)
            self.obs_buf_joint[:, 1, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs+6] = 0
            self.obs_buf_joint[:, 2, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs+6] = self.actions[:, self.num_shadow_hand_dofs:]

        # # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        # # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        # self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        # self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
        #                                                     self.shadow_hand_dof_lower_limits.view(1, -1), self.shadow_hand_dof_upper_limits.view(1, -1))
        # self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        # self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

        # fingertip_obs_start = 72
        # num_ft_states = 13 * self.num_fingertips
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

    def pre_physics_step(self, actions):
        """
        Apply actions on the shadow hand
        The action space is 30-dimensional:

        Index   Description
        0 - 23 	shadow hand actuated joint
        24 - 29 shadow hand base force
        """

        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        elif isinstance(actions, torch.Tensor):
            actions = actions.to(self.device)
        else :
            raise Exception("Invalid type for actions")

        self.actions.copy_(actions)

        self.cur_targets[:, :] = scale(
            self.actions[:, 0:self.num_shadow_hand_dofs],
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits
        )
        self.cur_targets[:, :] = (
            self.act_moving_average * self.cur_targets
            + (1.0 - self.act_moving_average) * self.pre_targets
        )
        self.cur_targets[:, :] = tensor_clamp(
            self.cur_targets,
            self.shadow_hand_dof_lower_limits,
            self.shadow_hand_dof_upper_limits
        )
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        
        if self.movable_arm :
            self.apply_forces[:, 1, :] = self.actions[:, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs+3] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, 1, :] = self.actions[:, self.num_shadow_hand_dofs+3:self.num_shadow_hand_dofs+6] * self.dt * self.orientation_scale * 10000

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces), gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.pre_targets[:, :] = self.cur_targets[:, :]
    
    def _compute_reward(self, actions):
        """
        Compute the reward of all environments
        Save the reward in self.rew_buf
        """

        self.rew_buf[:] = 0.0
    
    def post_physics_step(self):
        """
        Compute observation and reward after physics step
        """

        self._compute_observation()
        self._compute_reward(self.actions)
    
    def reset(self) :

        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * torch.rand((self.num_envs, self.num_shadow_hand_dofs), device=self.device)

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[:, :] = pos

        self.pre_targets[:, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[:, :self.num_shadow_hand_dofs] = pos

        self.hand_positions[:, :] = self.saved_root_tensor[:, self.hand_index, 0:3]
        self.hand_orientations[:, :] = self.saved_root_tensor[:, self.hand_index, 3:7]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pre_targets))
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))
        
        self.progress_buf[:] = 0
        self.reset_buf[:] = 0

    def _draw_line(self, start, end, colors=[1,0,0], clear_lines=False) :

        prepared_vertices = torch.zeros((2, 3))
        prepared_vertices[0] = start
        prepared_vertices[1] = end
        prepared_colors = torch.zeros((1, 3))
        prepared_colors[0] = torch.tensor(colors)
        if self.viewer :
            if clear_lines :
                self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, self.envs[0], 1, prepared_vertices, prepared_colors)

    def _get_obs_elements(self) :
        
        length = 0
        for item in self.obs_shape_dict.values() :
            length += item[-1]

        obs_elements_positions = torch.zeros((self.num_envs, length))
        obs_elements_positions[:, :] = torch.arange(0, length).view(1, -1)
        return obs_elements_positions