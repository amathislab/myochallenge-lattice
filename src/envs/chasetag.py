import numpy as np
import collections
import gym
import pink
from myosuite.envs.myo.myochallenge.chasetag_v0 import ChaseTagEnvV0, ChallengeOpponent
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2euler, euler2quat


class CustomChallengeOpponent(ChallengeOpponent):
    def __init__(
        self,
        sim,
        rng,
        probabilities,
        min_spawn_distance,
        opponent_x_range,
        opponent_y_range,
        opponent_orient_range,
        speed=10,
    ):
        self.x_min, self.x_max = opponent_x_range
        self.y_min, self.y_max = opponent_y_range
        self.theta_min, self.theta_max = opponent_orient_range
        self.speed = speed
        if self.x_min > self.x_max:
            raise ValueError("Invalid x range:", self.x_min, self.x_max)
        if self.y_min > self.y_max:
            raise ValueError("Invalid y range:", self.y_min, self.y_max)
        if self.theta_min > self.theta_max:
            raise ValueError("invalid theta range:", self.theta_min, self.theta_max)
        max_distance = np.linalg.norm(
            (
                max(abs(self.x_min), abs(self.x_max)),
                max(abs(self.y_min), abs(self.y_max)),
            )
        )
        if max_distance <= min_spawn_distance:
            raise ValueError(
                "The provided spawn ranges are incompatible with the min spawn distance",
                opponent_x_range,
                opponent_y_range,
                min_spawn_distance,
            )
        super().__init__(
            sim=sim,
            rng=rng,
            probabilities=probabilities,
            min_spawn_distance=min_spawn_distance,
        )

    def reset_noise_process(self):
        self.noise_process = pink.ColoredNoiseProcess(beta=2, size=(2, 2000), scale=self.speed, rng=self.rng)
        
    def reset_opponent(self, player_task="CHASE", rng=None):
        """
        This function should initially place the opponent on a random position with a
        random orientation with a minimum radius to the model.
        :task: Task for the PLAYER, I.e. 'CHASE' means that the player has to chase and the opponent has to evade.
        :rng: np_random generator
        """
        if rng is not None:
            self.rng = rng
            self.reset_noise_process()

        self.opponent_vel = np.zeros((2,))

        if player_task == "CHASE":
            self.sample_opponent_policy()
        elif player_task == "EVADE":
            self.opponent_policy = "chase_player"
        else:
            raise NotImplementedError

        dist = 0
        while dist < self.min_spawn_distance:
            pose = [
                self.rng.uniform(self.x_min, self.x_max),
                self.rng.uniform(self.y_min, self.y_max),
                self.rng.uniform(self.theta_min, self.theta_max),
            ]
            dist = np.linalg.norm(pose[:2] - self.sim.data.body("root").xpos[:2])
        if self.opponent_policy == "static_stationary":
            pose[:] = [0, -5, 0]
        self.set_opponent_pose(pose)
        self.opponent_vel[:] = 0.0


class CustomChaseTagEnv(ChaseTagEnvV0):
    CUSTOM_RWD_KEYS_AND_WEIGHTS = {
        "done": 0,
        "act_reg": 0,
        "lose": -10,
        "sparse": 0,
        "solved": 1,
        "alive": 1,
        "distance": 0,
        "vel_reward": 0,
        "cyclic_hip": 0,
        "ref_rot": 0,
        "joint_angle_rew": 0,
        "early_solved": 0,
        "joints_in_range": 0,
        "heel_pos": 0,
        "gait_prod": 0,
    }

    CUSTOM_DEFAULT_OBS_KEYS = [
        "internal_qpos",
        "internal_qvel",
        "grf",
        "torso_angle",
        "opponent_pose",
        "opponent_vel",
        "model_root_pos",
        "model_root_vel",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
        # 'gait_phase', # added to improve loco better loco
        # 'feet_rel_positions'
    ]

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # This flag needs to be here to prevent the simulation from starting in a done state
        # Before setting the key_frames, the model and opponent will be in the cartesian position,
        # causing the step() function to evaluate the initialization as "done".
        self.startFlag = False

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        BaseV0.__init__(
            self, model_path=model_path, obsd_model_path=obsd_model_path, seed=seed
        )
        self._setup(**kwargs)

    def _setup(
        self,
        obs_keys: list = CUSTOM_DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = CUSTOM_RWD_KEYS_AND_WEIGHTS,
        opponent_probabilities=[0.1, 0.45, 0.45],
        reset_type="none",
        win_distance=0.5,
        min_spawn_distance=2,
        task_choice="CHASE",
        terrain="FLAT",
        hills_range=(0, 0),
        rough_range=(0, 0),
        relief_range=(0, 0),
        max_time=20,
        min_height=0,
        stop_on_win=True,
        hip_period=100,
        opponent_x_range=(-5, 5),
        opponent_y_range=(-5, 5),
        opponent_orient_range=(-2 * np.pi, 2 * np.pi),
        gait_cadence=0.01,
        gait_stride_length=0.8,
        opponent_speed=10,
        target_speed=0,
        agent_x_range=(-5, 5),
        agent_y_range=(-5, 5),
        agent_orient_range=(0, 2 * np.pi),
        traj_mode="opponent",
        **kwargs,
    ):
        self.gait_cadence = gait_cadence
        self.gait_stride_length = gait_stride_length
        self.target_speed = target_speed
        self.agent_x_range = agent_x_range
        self.agent_y_range = agent_y_range
        self.agent_orient_range = agent_orient_range
        self.traj_mode = traj_mode
        self.should_be_foot_in_front = 0
        self.init_phase = 0
        self.arena_size = 6
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            opponent_probabilities=opponent_probabilities,
            reset_type=reset_type,
            win_distance=win_distance,
            min_spawn_distance=min_spawn_distance,
            task_choice=task_choice,
            terrain=terrain,
            hills_range=hills_range,
            rough_range=rough_range,
            relief_range=relief_range,
            min_height=min_height,
            hip_period=hip_period,
            **kwargs,
        )
        self.opponent = CustomChallengeOpponent(
            sim=self.sim,
            rng=self.np_random,
            probabilities=opponent_probabilities,
            min_spawn_distance=min_spawn_distance,
            opponent_x_range=opponent_x_range,
            opponent_y_range=opponent_y_range,
            opponent_orient_range=opponent_orient_range,
            speed=opponent_speed,
        )
        self.target_x_vel = 0.0
        self.target_y_vel = 0.0
        self.maxTime = max_time
        self.stop_on_win = stop_on_win

    def reset(self):
        self.steps = 0
        obs = super().reset()
        self.should_be_foot_in_front = self._get_foot_in_front()
        self.init_phase = - np.pi / 2 if self.should_be_foot_in_front == 1 else np.pi
        return np.nan_to_num(obs)

    def get_obs_dict(self, sim):
        obs_dict = {}

        # Time
        obs_dict["time"] = np.array([sim.data.time])

        # proprioception
        obs_dict["internal_qpos"] = sim.data.qpos[7:35].copy()
        obs_dict["internal_qvel"] = sim.data.qvel[6:34].copy() * self.dt
        obs_dict["grf"] = self._get_grf().copy()
        obs_dict["torso_angle"] = self.sim.data.body("pelvis").xquat.copy()

        obs_dict["muscle_length"] = self.muscle_lengths()
        obs_dict["muscle_velocity"] = self.muscle_velocities()
        obs_dict["muscle_force"] = self.muscle_forces()

        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        # exteroception
        if self.traj_mode == "virtual_traj":
            obs_dict["opponent_pose"] = self.get_target_pos()[:].copy()
        elif self.traj_mode == "opponent":
            obs_dict["opponent_pose"] = self.opponent.get_opponent_pose()[:].copy()
        # modify opponent_pose self.opponent.get_target_pos()[:].copy()
        obs_dict["opponent_vel"] = self.opponent.opponent_vel[:].copy()
        obs_dict["model_root_pos"] = sim.data.qpos[:2].copy()
        obs_dict["model_root_vel"] = sim.data.qvel[:2].copy()

        # active task
        obs_dict["task"] = np.array(self.current_task.value, ndmin=2, dtype=np.int16)
        # heightfield view of 10x10 grid of points around agent. Reshape to (10, 10) for visual inspection
        if not self.heightfield is None:
            obs_dict["hfield"] = self.heightfield.get_heightmap_obs()

        # obs_dict['gait_phase'] = self.get_gait_phase()
        # obs_dict['gait_phase'] = np.array([sim.data.time*self.gait_cadence % 1])
        obs_dict["gait_phase"] = np.array([(self.steps * self.gait_cadence) % 1]).copy()

        # Get the feet positions relative to the pelvis. (f_l, f_r)
        obs_dict["feet_rel_positions"] = self._get_feet_relative_position()
        # phase between 0 and 1. If hip period is in same unit as steps then phase no units

        return obs_dict

    def get_reward_dict(self, obs_dict):
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1).item() / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )
        win_cdt = self._win_condition()
        lose_cdt = self._lose_condition()
        score = self._get_score(float(self.obs_dict["time"])) if win_cdt else 0
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew(
            ["hip_adduction_l", "hip_adduction_r", "hip_rotation_l", "hip_rotation_r"]
        )
        rwd_dict = collections.OrderedDict(
            (
                # Perform reward tuning here --
                # Update Optional Keys section below
                # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
                # Examples: Env comes pre-packaged with two keys act_reg and lose
                # Optional Keys
                ("act_reg", act_mag),
                ("lose", lose_cdt),
                ("distance", np.exp(-self.get_distance_from_opponent())),
                ("alive", not self._get_done()),
                ("vel_reward", vel_reward),
                ("cyclic_hip", np.exp(-cyclic_hip)),
                ("ref_rot", ref_rot),
                ("joint_angle_rew", joint_angle_rew),
                (
                    "early_solved",
                    win_cdt * (self.maxTime - self.obs_dict["time"]).item(),
                ),
                ("joints_in_range", np.exp(- 5 * (1 - self._frac_joints_in_range()))),
                ("heel_pos", np.exp(-self._get_heel_rew())),
                (
                    "gait_prod",
                    np.exp(-cyclic_hip)
                    * np.exp(-self._get_alternating_hip_rew())
                    * vel_reward
                    * ref_rot,
                ),
                ("feet_height", np.sum(self._get_feet_heights().clip(0, 0.4))),
                ("alternating_foot", self._get_alternating_foot_rew()),
                ("lateral_foot_position", self._get_lateral_position_rew()),
                # Must keys
                ("sparse", score),
                ("solved", win_cdt),
                ("done", self._get_done()),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        # Success Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :] = (
            np.array([0, 2, 0, 0.1]) if rwd_dict["solved"] else np.array([2, 0, 0, 0])
        )
        return rwd_dict

    def get_distance_from_opponent(self):
        root_pos = self.sim.data.body("pelvis").xpos[:2]
        opp_pos = self.obs_dict["opponent_pose"][..., :2]
        return np.linalg.norm(root_pos - opp_pos)

    def step(self, action):
        obs, reward, done, info = super().step(action) 
        if self.steps % (self.hip_period // 2) == 0:  # Need to alternate feet
            self.should_be_foot_in_front = (self.should_be_foot_in_front + 1) % 2
        obs = np.nan_to_num(obs)
        reward = np.nan_to_num(reward)
        info.update(info.get("rwd_dict"))
        return obs, reward, done, info

    def _get_fallen_condition(self):
        fallen = self._get_height() < self.min_height
        episode_over = float(self.obs_dict["time"]) >= self.maxTime
        root_pos = self.sim.data.body("pelvis").xpos[:2]
        out_of_grid = np.abs(root_pos[0]) > 6.5 or np.abs(root_pos[1]) > 6.5
        return fallen or episode_over or out_of_grid

    def _chase_win_condition(self):
        root_pos = self.sim.data.body('pelvis').xpos[:2]
        opp_pos = self.opponent.get_opponent_pose()[:].copy()[:2]
        if np.linalg.norm(root_pos - opp_pos) <= self.win_distance and self.startFlag:
            return 1
        return 0

    def _evade_lose_condition(self):
        root_pos = self.sim.data.body('pelvis').xpos[:2]
        opp_pos = self.opponent.get_opponent_pose()[:].copy()[:2]

        # got caught
        if np.linalg.norm(root_pos - opp_pos) <= self.win_distance and self.startFlag:
            return 1
        # out-of-bounds
        if np.abs(root_pos[0]) > 6.5 or np.abs(root_pos[1]) > 6.5:
            return 1
        if self._get_fallen_condition(): # for training stop if agent falls
            return 1 
        return 0
    
    def _get_done(self):
        if self._lose_condition():
            return 1
        if self._win_condition() and self.stop_on_win:
            return 1
        return 0

    def get_root_orientation(self):
        quat = self.sim.data.qpos[3:7].copy()
        xy_angle = quat2euler(quat)[-1]
        return np.array((np.cos(xy_angle), np.sin(xy_angle)))

    def get_opponent_relative_orientation(self):
        root_pos = self.sim.data.body("pelvis").xpos[:2].copy()
        opp_pos = self.obs_dict["opponent_pose"][..., :2].copy()
        dist_versor = opp_pos - root_pos
        versor_norm = np.linalg.norm(dist_versor)
        if versor_norm > 0:
            dist_versor /= versor_norm
        return dist_versor

    def _get_ref_rotation_rew(self):
        """
        Incentivize orienting the root towards the target.
        """
        root_rot = self.get_root_orientation()
        opponent_rot = self.get_opponent_relative_orientation()
        return np.exp(-5.0 * np.linalg.norm(root_rot - opponent_rot))

    def _frac_joints_in_range(self):
        joints_lower_bound = self.joint_ranges[:, 0] <= self.sim.data.qpos[7:35]
        joints_upper_bound = self.sim.data.qpos[7:35] <= self.joint_ranges[:, 1]
        joints_in_range = np.logical_and(joints_lower_bound, joints_upper_bound)
        return np.mean(joints_in_range)

    @property
    def joint_ranges(self):
        return self.sim.model.jnt_range[1:, :].copy()

    def _get_heel_target(self):
        """
        Returns desired rel position of foot (rel to pelvis) during gait
        """
        phase = self.steps * self.gait_cadence
        heel_pos = np.array(
            [
                0.5 * self.gait_stride_length * np.cos(phase * 2 * np.pi + np.pi),
                0.5 * self.gait_stride_length * np.cos(phase * 2 * np.pi),
            ],
            dtype=np.float32,
        )
        return heel_pos

    def _get_heel_rew(self):
        """
        Relative heel position in gait rewarded to incentivize a walking gait.
        max distance is stride
        """
        l_heel, r_heel = self._get_feet_relative_position()
        des = self._get_heel_target()
        l_des = des[0]
        r_des = des[1]
        # TODO: check the initial position of the heels, one feet can be in front of the other at random.
        return abs(np.linalg.norm(l_heel[:2]) - l_des) + abs(np.linalg.norm(r_heel[:2]) - r_des)

    def _get_vel_reward(self):
        """
        Gaussian that incentivizes a walking velocity. Going
        over only achieves flat rewards.
        If both target vel are zero, follow a target speed
        """
        vel = self._get_com_velocity()

        # Check if both target velocities are zero
        # Compute the reward for both x and y velocities
        if self.target_x_vel !=0 and self.target_y_vel !=0:
            return np.exp(-np.square(self.target_y_vel - vel[1])) + np.exp(
            -np.square(self.target_x_vel - vel[0])
            )
    
        # otherwise scalar velocity
        return np.exp(-np.square(self.target_speed - np.linalg.norm(vel)))

        

    def _get_alternating_hip_rew(self):
        """
        Alternating extension of hip angles is rewarded to incentivize a walking gait.
        Reward if sum of hip extension of left and right is zero
        """
        # phase_var = (self.steps/self.hip_period) % 1
        # des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)], dtype=np.float32)
        angles = self._get_angle(["hip_flexion_l", "hip_flexion_r"])
        return np.linalg.norm(angles[0] + angles[1])

    def _randomize_position_orientation(self, qpos, qvel):
        qpos[0] = self.np_random.uniform(*self.agent_x_range)
        qpos[1] = self.np_random.uniform(*self.agent_y_range)
        orientation = self.np_random.uniform(*self.agent_orient_range)
        euler_angle = quat2euler(qpos[3:7])
        euler_angle[-1] = orientation
        qpos[3:7] = euler2quat(euler_angle)
        return qpos, qvel
    
    def _get_feet_xy_position(self):
        """
        Get the absolute feet position on the xy plane.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        return self.sim.data.body_xpos[foot_id_l][:2], self.sim.data.body_xpos[foot_id_r][:2]
    
    def _get_foot_in_front(self):
        """Get which foot is in front according to the root orientation (0: left, 1: right)
        """
        left_pos, right_pos = self._get_feet_xy_position()
        root_orientation = self.get_root_orientation()
        return int(np.dot(right_pos, root_orientation) > np.dot(left_pos, root_orientation))

    def _get_alternating_foot_rew(self):
        """Rewards with 1 if the foot in front corresponds to that of the current phase
        """
        return int(self._get_foot_in_front() == self.should_be_foot_in_front)
    
    def _get_feet_lateral_position(self):
        """Projects the relative position of the feet (wrt the pelvis) on the orthogonal direction to the root.
        The values are positive if the feet is on its side (left foot -> left of the root, etc)"""
        root_orientation = self.get_root_orientation()
        root_orth_orientation = [-root_orientation[1], root_orientation[0]]        
        left_pos, right_pos = self._get_feet_relative_position()
        left_lateral_pos = np.dot(left_pos[:2], root_orth_orientation)
        right_lateral_pos = np.dot(right_pos[:2], root_orth_orientation)
        return left_lateral_pos, - right_lateral_pos
    
    def _get_lateral_position_rew(self):
        """Give a reward proportional to the lateral position of the feet. If the right (left) foot is on
        the left (right) of the pelvis, the reward will be negative. Otherwise the reward is positive, 
        but not larger than 0.1 (10 cm), as opening more might not be desirable."""
        left_lateral_pos, right_lateral_pos = self._get_feet_lateral_position()
        return np.clip(left_lateral_pos, None, 0) + np.clip(right_lateral_pos, None, 0)

    def _get_cyclic_rew(self):
        """
        Cyclic extension of hip angles is rewarded to incentivize a walking gait.
        """
        phase_var = (self.steps/self.hip_period) % 1
        des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + self.init_phase), 0.8 * np.cos(phase_var * 2 * np.pi - self.init_phase)], dtype=np.float32)
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return np.linalg.norm(des_angles - angles)
    
    def normalize(self, vec):
        """return normalized vec"""
        if np.linalg.norm(vec) == 0:
            return vec
        else:
            return 1/np.linalg.norm(vec) * vec
        
    def is_cornered(self, pose, opponent_pose):
        """
        Returns true if the agent is between the corner and the opponent
        """
        in_corner = False
        x_a = pose[0]
        y_a = pose[1]
        x_o = opponent_pose[0]
        y_o = opponent_pose[1]
        if x_o*(x_a-(x_o-0.5*np.sign(x_o)))>=0 and y_o*(y_a-(y_o-0.5*np.sign(y_o)))>=0:
            in_corner = True
            # print("in the corner")
        return in_corner
        
    def vec_opponent_to_agent(self, pose, opponent_pose):
        """ vector from opponent to agent """
        # pose = self.obs_dict["model_root_pos"]
        # opponent_pose = self.opponent.get_opponent_pose()[:].copy()
        return np.array(np.array(pose) - np.array(opponent_pose[:2]))
        # return np.array(self.pose[:2]) - np.array(self.opponent.pose[:2])
    
    def chase_vel(self, pose, opponent_pose, theta):
        """
        Returns the velocity vector in x-y plane to chase the opponent
        """
        agent_to_opponent = - self.vec_opponent_to_agent(pose, opponent_pose)
        direction = agent_to_opponent
        dist_to_opp = np.linalg.norm(agent_to_opponent)
        opp_dir_comp = np.sqrt(dist_to_opp) # scaling of dist to opponent
        ### smooth over with current orientation
        theta_dir = np.arctan2(direction[-1], direction[0]) # angle of direction
        angle = theta_dir - theta
        # angular velocity between -1 and 1 
        angle_vel = np.sign(np.sin(angle)) * np.abs(np.sin(angle/2))
        # speed depends on how close to opponent and how much we need to turn
        speed = 1 * opp_dir_comp / (2*np.abs(angle_vel)+1.0)
        # return [speed, angle_vel]
        # return velocity in x_y plane 
        return [speed * np.cos(theta + angle_vel), speed * np.sin(theta + angle_vel), angle_vel]

    def evade_vel(self, pose, opponent_pose, theta):
        in_corner = self.is_cornered(pose, opponent_pose)
        opponent_to_agent = self.vec_opponent_to_agent(pose, opponent_pose)
        if not in_corner:
            agent_to_center = - np.array(pose[:2])
            dist_to_opp = np.linalg.norm(opponent_to_agent)
            theta_op = opponent_pose[-1]
            opp_or = np.array([-np.cos(theta_op + 0.5*np.pi),-np.sin(theta_op + 0.5*np.pi)])
            opp_running_dir_comp = np.maximum(1.0, 2*np.dot(self.normalize(opponent_to_agent),opp_or))
            opp_dir_comp = 1.5/np.sqrt(dist_to_opp) * opp_running_dir_comp
            center_dir_comp = np.sqrt(np.linalg.norm(agent_to_center))
            direction = center_dir_comp * self.normalize(agent_to_center) + opp_dir_comp * self.normalize(opponent_to_agent)
        else:
            dist_to_opp = np.linalg.norm(opponent_to_agent)
            opp_dir_comp = 2/np.sqrt(dist_to_opp)
            x_a = pose[0]
            y_a = pose[1]
            if x_a == 0.0:
                x_dir_comp = 0.0
            else:
                x_dir_comp = - (self.arena_size * np.sign(x_a))/(np.abs(x_a - np.sign(x_a)*self.arena_size)) 
            if y_a == 0.0:
                y_dir_comp = 0.0
            else:
                y_dir_comp = - (self.arena_size * np.sign(y_a))/(np.abs(y_a - np.sign(y_a)*self.arena_size)) 
            opp_dir_vec = opp_dir_comp * self.normalize(opponent_to_agent)
            x_dir_vec = x_dir_comp * np.array([1,0]) 
            y_dir_vec = y_dir_comp * np.array([0,1])
            direction = opp_dir_vec + x_dir_vec + y_dir_vec
            # dir_comp = np.sqrt(opp_dir_comp**2 + x_dir_comp**2 + y_dir_comp**2)
            # print(dir_comp)
            # direction = self.normalize(direction) * dir_comp
        ### smooth over with current orientation
        theta_dir = np.arctan2(direction[-1], direction[0]) # angle of direction
        angle = theta_dir - theta
        angle_vel = np.sign(np.sin(angle)) * np.abs(np.sin(angle/2))
        speed = 1 * opp_dir_comp / (2*np.abs(angle_vel)+1.0)
        # return [speed, angle_vel]
        return [speed * np.cos(theta + angle_vel), speed * np.sin(theta + angle_vel), angle_vel]

    def get_target_pos(self):
        """
        Target position according to virtual trajectory. Depends on chase and evade. 
        :return: The  pose.
        :rtype: list -> [x, y, angle]
        """
        # if self.target_speed != 0:
        #     target_speed = self.target_speed
        # else: 
        #     target_speed = 2.0
        pose = self.sim.data.qpos[:2].copy() # x, y pos of the agent
        opponent_pose = self.opponent.get_opponent_pose()[:].copy()
        agent_or = self.get_root_orientation()
        theta = np.arctan2(agent_or[-1], agent_or[0]) # current orientation angle zero facing right
        vel = np.zeros(3)
        if self.current_task.name == 'CHASE':
            opponent_to_agent = self.vec_opponent_to_agent(pose, opponent_pose)
            dist_to_opp = np.linalg.norm(opponent_to_agent)
            if dist_to_opp < 3.0: # if close just set opponent as target
                vel = - opponent_to_agent
                # vel_n = target_speed * self.normalize(vel)
                self.target_y_vel = vel[1]
                self.target_x_vel = vel[0]
                return opponent_pose
            else:
                vel = self.chase_vel(pose, opponent_pose, theta)
                self.target_y_vel = vel[1]
                self.target_x_vel = vel[0]
                target_dist_scaling = 2.0
        elif self.current_task.name == 'EVADE':
            vel = self.evade_vel(pose, opponent_pose, theta)
            self.target_y_vel = vel[1]
            self.target_x_vel = vel[0]
            target_dist_scaling = 2.0
        virtual_target = np.array([pose[0] + target_dist_scaling * vel[0], pose[1] + target_dist_scaling * vel[1], theta + vel[2]])
        virtual_target =  np.clip(virtual_target, - self.arena_size, self.arena_size)
        # get current opponent pos
        # opponent_pose = self.opponent.get_opponent_pose()[:].copy()
        return virtual_target
