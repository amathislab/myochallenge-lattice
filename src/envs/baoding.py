# pylint: disable=attribute-defined-outside-init, dangerous-default-value, protected-access, abstract-method, arguments-renamed, import-error
import collections
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler
from myosuite.envs.myo.myochallenge.baoding_v1 import WHICH_TASK, BaodingEnvV1, Task
from envs.env_mixins import DictObsMixin, ObsEmbeddingMixin
from definitions import ACT_KEY, OBJ_KEY, GOAL_KEY


class CustomBaodingEnv(BaodingEnvV1):
    CUSTOM_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist_1": 1,
        "pos_dist_2": 1,
        "act_reg": 0,
        "alive": 1,
        "solved": 5,
        "done": 0,
        "sparse": 0,
    }

    def _setup(
        self,
        frame_skip: int = 10,
        drop_th=1.25,  # drop height threshold
        proximity_th=0.015,  # object-target proximity threshold
        goal_time_period=(5, 5),  # target rotation time period
        goal_xrange=(0.025, 0.025),  # target rotation: x radius (0.03)
        goal_yrange=(0.028, 0.028),  # target rotation: x radius (0.02 * 1.5 * 1.2)
        obj_size_range=None,  # Object size range. Nominal 0.022
        obj_mass_range=None,  # Object weight range. Nominal 43 gms
        obj_friction_change=None,
        task_choice="fixed",  # fixed/ random
        obs_keys: list = BaodingEnvV1.DEFAULT_OBS_KEYS,
        weighted_reward_keys: list = CUSTOM_RWD_KEYS_AND_WEIGHTS,
        enable_rsi=False,  # random state init for balls
        rsi_probability=1,  # probability of implementing RSI
        balls_overlap=False,
        overlap_probability=0,
        limit_init_angle=None,
        **kwargs,
    ):
        # user parameters
        self.task_choice = task_choice
        self.which_task = (
            self.np_random.choice(Task) if task_choice == "random" else Task(WHICH_TASK)
        )
        self.drop_th = drop_th
        self.proximity_th = proximity_th
        self.goal_time_period = goal_time_period
        self.goal_xrange = goal_xrange
        self.goal_yrange = goal_yrange
        self.rsi = enable_rsi
        self.rsi_probability = rsi_probability
        self.balls_overlap = balls_overlap
        self.overlap_probability = overlap_probability
        self.limit_init_angle = limit_init_angle

        # balls start at these angles
        #   1= yellow = top right
        #   2= pink = bottom left

        if np.random.uniform(0, 1) < self.overlap_probability:
            self.ball_1_starting_angle = 3.0 * np.pi / 4.0
            self.ball_2_starting_angle = -1.0 * np.pi / 4.0
        else:
            self.ball_1_starting_angle = 1.0 * np.pi / 4.0
            self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

        # init desired trajectory, for rotations
        self.center_pos = [-0.0125, -0.07]  # [-.0020, -.0522]
        self.x_radius = self.np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = self.np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )

        self.counter = 0
        self.goal = self.create_goal_trajectory(
            time_step=frame_skip * self.sim.model.opt.timestep, time_period=6
        )

        # init target and body sites
        self.object1_bid = self.sim.model.body_name2id("ball1")
        self.object2_bid = self.sim.model.body_name2id("ball2")
        self.object1_sid = self.sim.model.site_name2id("ball1_site")
        self.object2_sid = self.sim.model.site_name2id("ball2_site")
        self.object1_gid = self.sim.model.geom_name2id("ball1")
        self.object2_gid = self.sim.model.geom_name2id("ball2")
        self.target1_sid = self.sim.model.site_name2id("target1_site")
        self.target2_sid = self.sim.model.site_name2id("target2_site")
        self.sim.model.site_group[self.target1_sid] = 2
        self.sim.model.site_group[self.target2_sid] = 2

        # setup for task randomization
        self.obj_mass_range = (
            {"low": obj_mass_range[0], "high": obj_mass_range[1]}
            if obj_mass_range
            else None
        )
        self.obj_size_range = (
            {"low": obj_size_range[0], "high": obj_size_range[1]}
            if obj_mass_range
            else None
        )
        self.obj_friction_range = (
            {
                "low": self.sim.model.geom_friction[self.object1_gid]
                - obj_friction_change,
                "high": self.sim.model.geom_friction[self.object1_gid]
                + obj_friction_change,
            }
            if obj_friction_change
            else None
        )

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            frame_skip=frame_skip,
            **kwargs,
        )

        # reset position
        self.init_qpos[:-14] *= 0  # Use fully open as init pos
        self.init_qpos[0] = -1.57  # Palm up

    def get_reward_dict(self, obs_dict):
        # tracking error
        target1_dist = np.linalg.norm(obs_dict["target1_err"], axis=-1)
        target2_dist = np.linalg.norm(obs_dict["target2_err"], axis=-1)
        target_dist = target1_dist + target2_dist
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        # detect fall
        object1_pos = (
            obs_dict["object1_pos"][:, :, 2]
            if obs_dict["object1_pos"].ndim == 3
            else obs_dict["object1_pos"][2]
        )
        object2_pos = (
            obs_dict["object2_pos"][:, :, 2]
            if obs_dict["object2_pos"].ndim == 3
            else obs_dict["object2_pos"][2]
        )
        is_fall_1 = object1_pos < self.drop_th
        is_fall_2 = object2_pos < self.drop_th
        is_fall = np.logical_or(is_fall_1, is_fall_2)  # keep both balls up

        rwd_dict = collections.OrderedDict(
            (
                # Perform reward tuning here --
                # Update Optional Keys section below
                # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
                # Examples: Env comes pre-packaged with two keys pos_dist_1 and pos_dist_2
                # Optional Keys
                ("pos_dist_1", -1.0 * target1_dist),
                ("pos_dist_2", -1.0 * target2_dist),
                # Must keys
                ("act_reg", -1.0 * act_mag),
                ("alive", ~is_fall),
                ("sparse", -target_dist),
                (
                    "solved",
                    (target1_dist < self.proximity_th)
                    * (target2_dist < self.proximity_th)
                    * (~is_fall),
                ),
                ("done", is_fall),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        # Sucess Indicator
        self.sim.model.geom_rgba[self.object1_gid, :2] = (
            np.array([1, 1])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )
        self.sim.model.geom_rgba[self.object2_gid, :2] = (
            np.array([0.9, 0.7])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )

        return rwd_dict

    def reset(self, reset_pose=None, reset_vel=None, reset_goal=None, time_period=None):
        # reset task
        if self.task_choice == "random":
            self.which_task = self.np_random.choice(Task)

            if np.random.uniform(0, 1) <= self.overlap_probability:
                self.ball_1_starting_angle = 3.0 * np.pi / 4.0
            elif self.limit_init_angle is not None:
                random_phase = self.np_random.uniform(
                    low=-self.limit_init_angle, high=self.limit_init_angle
                )
            else:
                self.ball_1_starting_angle = self.np_random.uniform(
                    low=0, high=2 * np.pi
                )

            self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

        # reset counters
        self.counter = 0
        self.x_radius = self.np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = self.np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )

        # reset goal
        if time_period is None:
            time_period = self.np_random.uniform(
                low=self.goal_time_period[0], high=self.goal_time_period[1]
            )
        self.goal = (
            self.create_goal_trajectory(time_step=self.dt, time_period=time_period)
            if reset_goal is None
            else reset_goal.copy()
        )

        # balls mass changes
        if self.obj_mass_range:
            self.sim.model.body_mass[self.object1_bid] = self.np_random.uniform(
                **self.obj_mass_range
            )  # call to mj_setConst(m,d) is being ignored. Derive quantities wont be updated. Die is simple shape. So this is reasonable approximation.
            self.sim.model.body_mass[self.object2_bid] = self.np_random.uniform(
                **self.obj_mass_range
            )  # call to mj_setConst(m,d) is being ignored. Derive quantities wont be updated. Die is simple shape. So this is reasonable approximation.

        # balls friction changes
        if self.obj_friction_range:
            self.sim.model.geom_friction[self.object1_gid] = self.np_random.uniform(
                **self.obj_friction_range
            )
            self.sim.model.geom_friction[self.object2_gid] = self.np_random.uniform(
                **self.obj_friction_range
            )

        # balls size changes
        if self.obj_size_range:
            self.sim.model.geom_size[self.object1_gid] = self.np_random.uniform(
                **self.obj_size_range
            )
            self.sim.model.geom_size[self.object2_gid] = self.np_random.uniform(
                **self.obj_size_range
            )

        # reset scene
        qpos = self.init_qpos.copy() if reset_pose is None else reset_pose
        qvel = self.init_qvel.copy() if reset_vel is None else reset_vel
        self.robot.reset(qpos, qvel)

        if self.rsi and np.random.uniform(0, 1) <= self.rsi_probability:
            random_phase = np.random.uniform(low=-np.pi, high=np.pi)
            self.ball_1_starting_angle = 3.0 * np.pi / 4.0 + random_phase
            self.ball_2_starting_angle = -1.0 * np.pi / 4.0 + random_phase

            # # reset scene (MODIFIED from base class MujocoEnv)
            self.robot.reset(qpos, qvel)
            self.step(np.zeros(39))
            # update ball positions
            obs_dict = self.get_obs_dict(self.sim)
            target_1_pos = obs_dict["target1_pos"]
            target_2_pos = obs_dict["target2_pos"]
            qpos[23] = target_1_pos[0]  # ball 1 x-position
            qpos[24] = target_1_pos[1]  # ball 1 y-position
            qpos[30] = target_2_pos[0]  # ball 2 x-position
            qpos[31] = target_2_pos[1]  # ball 2 y-position
            self.set_state(qpos, qvel)

            if self.balls_overlap is False:
                self.ball_1_starting_angle = self.np_random.uniform(
                    low=0, high=2 * np.pi
                )
                self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

        return self.get_obs()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info.update(info.get("rwd_dict"))
        return obs, reward, done, info


class MuscleBaodingEnv(CustomBaodingEnv, DictObsMixin, ObsEmbeddingMixin):
    OBS_KEYS = [
        ACT_KEY,
        OBJ_KEY,
        GOAL_KEY,
    ]

    def __init__(
        self,
        model_path,
        obsd_model_path=None,
        seed=None,
        include_adapt_state=False,
        num_memory_steps=30,
        **kwargs,
    ):
        self._init_done = False
        super().__init__(
            model_path, obsd_model_path=obsd_model_path, seed=seed, **kwargs
        )
        self.action_dim = self.sim.model.nu
        self._dict_obs_init_addon(include_adapt_state, num_memory_steps)
        self._obs_embedding_init_addon()
        self._init_done = True

    def _setup(
        self,
        obs_keys: list = OBS_KEYS,
        **kwargs,
    ):
        super()._setup(
            obs_keys=obs_keys,
            **kwargs,
        )
        self.obs_keys.remove("act")

    def reset(self, reset_pose=None, reset_vel=None, reset_goal=None, time_period=None):
        super().reset(
            reset_pose=reset_pose,
            reset_vel=reset_vel,
            reset_goal=reset_goal,
            time_period=time_period,
        )
        obs = self.create_history_reset_state(self.obs_dict)
        obs = self.add_positions_to_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info.update(info.get("rwd_dict"))
        if self._init_done:
            obs = self.create_history_step_state(self.obs_dict)
            obs = self.add_positions_to_obs(obs)
        return obs, reward, done, info

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["muscle_len"] = np.nan_to_num(sim.data.actuator_length.copy())
        obs_dict["muscle_vel"] = np.nan_to_num(sim.data.actuator_velocity.copy())
        obs_dict["muscle_force"] = np.nan_to_num(sim.data.actuator_force.copy())
        
        obs_dict["object1_rot"] = mat2euler(sim.data.get_body_xmat("ball1"))
        obs_dict["object2_rot"] = mat2euler(sim.data.get_body_xmat("ball2"))
        obs_dict["object1_velr"] = sim.data.get_body_xvelr("ball1")
        obs_dict["object2_velr"] = sim.data.get_body_xvelr("ball2")

        # Actuator obs shape (num_features, num_muscles) = (4, 39)
        muscle_keys = ("muscle_len", "muscle_vel", "muscle_force", "act")
        obs_dict["actuator_obs"] = np.row_stack([obs_dict[key] for key in muscle_keys])

        # Object obs shape (num_object_features, num_objects) = (12, 2)
        obj_1_keys = ("object1_pos", "object1_velp", "object1_rot", "object1_velr")
        obj_2_keys = ("object2_pos", "object2_velp", "object2_rot", "object2_velr")
        obj_1_obs = np.concatenate([obs_dict[key] for key in obj_1_keys])
        obj_2_obs = np.concatenate([obs_dict[key] for key in obj_2_keys])
        obs_dict["object_obs"] = np.column_stack((obj_1_obs, obj_2_obs))

        # Goal obs shape (num_goal_features, num_goals) = (1, 6)
        goal_keys = ("target1_err", "target2_err")
        obs_dict["goal_obs"] = np.concatenate([obs_dict[key] for key in goal_keys]).reshape(1, -1)

        return obs_dict

    def get_obs_elements(self):
        actuators = list(self.sim.model.actuator_names)
        # objects = ("ball_1", "ball_2")
        # goals = ("ball_1_pos", "ball_2_pos")
        objects = ["ball1", "ball2"]
        goals = []
        for target in ["obj1_target_pos", "obj2_target_pos"]:
            goals.extend([f"{target}_x", f"{target}_y", f"{target}_z"])
        return [*actuators, *objects, *goals]
