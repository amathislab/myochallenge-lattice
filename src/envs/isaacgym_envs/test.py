from envs.base_motor_hand import BaseMotorHand
from envs.shadow_hand_reach import ShadowHandReachEnv
from envs.shadow_hand_finger_reach import ShadowHandFingerReachEnv
from envs.shadow_hand_reorient import ShadowHandReorientEnv
import torch

cfg = {
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

# env = ShadowHandFingerReachEnv(cfg, True, 4)
# env = ShadowHandReachEnv(cfg, True, 4)
env = ShadowHandReorientEnv(cfg, True, 4)

while True :

    for i in range(100) :
        action = torch.randn((cfg["env"]["num_envs"], env.num_actions), device=env.device, dtype=torch.float)
        obs, rew, done, info = env.step(action)
        print(obs["actuator_obs"][:, 0, 0, 0])
    env.reset()