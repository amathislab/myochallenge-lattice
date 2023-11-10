from isaacgym import gymapi
from ..utils.isaac_util import parse_sim_params
import torch
import sys
import gym
import numpy as np

class BaseMotorHand():

    def __init__(self, cfg):

        self.gym = gymapi.acquire_gym()
        self.enable_viewer_sync = True
        self.viewer = None

        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "gpu":
            self.device = "cuda" + ":" + str(self.device_id)
        
        self.headless = cfg["headless"]
        self.num_envs = cfg["env"]["num_envs"]
        # self.num_obs = cfg["env"]["num_observations"]
        # self.num_actions = cfg["env"]["num_actions"]
        self.control_freq_inv = cfg["env"]["control_frequency_inv"]
        self.max_episode_length = 128

        self.physics_engine = gymapi.SIM_PHYSX
        self.sim_params = parse_sim_params(cfg)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        
        self.sim = self.create_sim(self.device_id, self.device_id, self.physics_engine, self.sim_params)
        self.gym.prepare_sim(self.sim)

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(10, 10, 3.0)
                cam_target = gymapi.Vec3(0, 0, 0.0)
            else:
                cam_pos = gymapi.Vec3(10, 10, 3.0)
                cam_target = gymapi.Vec3(0, 0, 0.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)
        
    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1
    
    def reset(self, idx=None) :
        raise NotImplementedError        

    def step(self, actions) :

        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        elif isinstance(actions, torch.Tensor):
            actions = actions.to(self.device)
        else :
            raise ValueError("actions must be numpy.ndarray or torch.Tensor")

        # apply actions
        self.pre_physics_step(actions)
        self.progress_buf[:] += 1
        self.reset_buf[:] = self.progress_buf >= self.max_episode_length
        # print(self.progress_buf)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()


    def render(self, sync_frame_time=False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)
    
    def pre_physics_step(self, actions):
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError
    
    def seed(self, seed) :
        # TODO: set seed
        pass

    def save(self, path):
        # TODO: save
        pass

def update_cfg(old : dict, new : dict) :

    for key, val in new.items():

        if isinstance(old[key], dict) :
            update_cfg(old[key], val)
        else :
            old[key] = val
    
    return old