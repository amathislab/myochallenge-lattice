import numpy as np
import collections
import gym
from myosuite.envs.myo.myochallenge.relocate_v0 import RelocateEnvV0
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2euler
from myosuite.utils.quat_math import mat2euler, euler2quat

class CustomRelocateEnv(RelocateEnvV0):
    CUSTOM_RWD_KEYS_AND_WEIGHTS = {
        "done": 0,
        "act_reg": 0,
        "sparse": 0,
        "solved": 1,
        "pos_dist": 100.0,
        "rot_dist": 1.0
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        BaseV0.__init__(self, model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
            target_xyz_range,       # target position range (relative to initial pos)
            target_rxryrz_range,    # target rotation range (relative to initial rot)
            obs_keys:list = RelocateEnvV0.DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = CUSTOM_RWD_KEYS_AND_WEIGHTS,
            pos_th = .025,          # position error threshold
            rot_th = 0.262,         # rotation error threshold
            drop_th = 0.50,         # drop height threshold
            **kwargs,
        ):
        self.palm_sid = self.sim.model.site_name2id("S_grasp")
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.goal_sid = self.sim.model.site_name2id("target_o")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.goal_bid = self.sim.model.body_name2id("target")
        self.target_xyz_range = target_xyz_range
        self.target_rxryrz_range = target_rxryrz_range
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    target_xyz_range=target_xyz_range,
                    target_rxryrz_range=target_rxryrz_range,
                    pos_th=pos_th,
                    rot_th=rot_th,
                    drop_th=drop_th,
                    **kwargs,
        )
        keyFrame_id = 0
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()
    
    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:-7].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-6].copy()*self.dt
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['goal_pos'] = sim.data.site_xpos[self.goal_sid]
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]
        obs_dict['pos_err'] = obs_dict['goal_pos'] - obs_dict['obj_pos']
        obs_dict['reach_err'] = obs_dict['palm_pos'] - obs_dict['obj_pos']
        obs_dict['obj_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.object_sid],(3,3)))
        obs_dict['goal_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.goal_sid],(3,3)))
        obs_dict['rot_err'] = obs_dict['goal_rot'] - obs_dict['obj_rot']

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = np.abs(np.linalg.norm(self.obs_dict['reach_err'], axis=-1))
        pos_dist = np.abs(np.linalg.norm(self.obs_dict['pos_err'], axis=-1))
        rot_dist = np.abs(np.linalg.norm(self.obs_dict['rot_err'], axis=-1))
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        drop = reach_dist > self.drop_th
        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Examples: Env comes pre-packaged with two keys pos_dist and rot_dist

            # Optional Keys
            ('pos_dist', -1.*pos_dist),
            ('rot_dist', -1.*rot_dist),
            ('reach_dist', 1.*reach_dist),
            # Must keys
            ('act_reg', -1.*act_mag),
            ('sparse', -rot_dist-10.0*pos_dist),
            ('solved', (pos_dist<self.pos_th) and (rot_dist<self.rot_th) and (not drop) ),
            ('done', drop),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        # Success Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array([2, 0])
        self.sim.model.site_size[self.success_indicator_sid, :] = np.array([.25,]) if rwd_dict['solved'] else np.array([0.1,])
        return rwd_dict

    def step(self, action):
        if any(~np.isfinite(action)):
            print(action)
        obs, reward, done, info = super().step(action)
        obs = np.nan_to_num(obs)
        info.update(info.get("rwd_dict"))
        return obs, reward, done, info

    def reset(self, reset_qpos=None, reset_qvel=None):
        self.sim.model.body_pos[self.goal_bid] = self.np_random.uniform(**self.target_xyz_range)
        self.sim.model.body_quat[self.goal_bid] =  euler2quat(self.np_random.uniform(**self.target_rxryrz_range))
        obs = super().reset(reset_qpos, reset_qvel)
        return obs