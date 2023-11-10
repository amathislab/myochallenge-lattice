import os
import time
import json
import numpy as np
from utils.utils import RemoteConnection
from definitions import ROOT_DIR
from main_dataset_recurrent_ppo import load_vecnormalize, load_model
from envs.environment_factory import EnvironmentFactory


def normalize(vec):
    """return normalized vec"""
    if np.linalg.norm(vec) == 0:
        return vec
    else:
        return 1/np.linalg.norm(vec) * vec
    
def is_cornered(pose, opponent_pose):
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
    
def vec_opponent_to_agent(pose, opponent_pose):
    """ vector from opponent to agent """
    return np.array(np.array(pose) - np.array(opponent_pose[:2]))

def chase_vel(pose, opponent_pose, theta):
    """
    Returns the velocity vector in x-y plane to chase the opponent
    """
    agent_to_opponent = - vec_opponent_to_agent(pose, opponent_pose)
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

def evade_vel( pose, opponent_pose, theta):
    in_corner = is_cornered(pose, opponent_pose)
    opponent_to_agent = vec_opponent_to_agent(pose, opponent_pose)
    if not in_corner:
        agent_to_center = - np.array(pose[:2])
        dist_to_opp = np.linalg.norm(opponent_to_agent)
        theta_op = opponent_pose[-1]
        opp_or = np.array([-np.cos(theta_op + 0.5*np.pi),-np.sin(theta_op + 0.5*np.pi)])
        opp_running_dir_comp = np.maximum(1.0, 2*np.dot(normalize(opponent_to_agent),opp_or))
        opp_dir_comp = 1.5/np.sqrt(dist_to_opp) * opp_running_dir_comp
        center_dir_comp = np.sqrt(np.linalg.norm(agent_to_center))
        direction = center_dir_comp * normalize(agent_to_center) + opp_dir_comp * normalize(opponent_to_agent)
    else:
        dist_to_opp = np.linalg.norm(opponent_to_agent)
        opp_dir_comp = 2/np.sqrt(dist_to_opp)
        x_a = pose[0]
        y_a = pose[1]
        if x_a == 0.0:
            x_dir_comp = 0.0
        else:
            x_dir_comp = - (6 * np.sign(x_a))/(np.abs(x_a - np.sign(x_a)*6)) 
        if y_a == 0.0:
            y_dir_comp = 0.0
        else:
            y_dir_comp = - (6 * np.sign(y_a))/(np.abs(y_a - np.sign(y_a)*6)) 
        opp_dir_vec = opp_dir_comp * normalize(opponent_to_agent)
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

# def get_root_orientation(self):
#     quat = self.sim.data.qpos[3:7].copy()
#     xy_angle = quat2euler(quat)[-1]
#     return np.array((np.cos(xy_angle), np.sin(xy_angle)))

def get_target_pos(pose, opponent_pose, agent_vel, task):
    """
    Target position according to virtual trajectory. Depends on chase and evade. 
    :return: The  pose.
    :rtype: list -> [x, y, angle]
    """
    # agent_or = self.get_root_orientation()
    agent_or = normalize(agent_vel)
    theta = np.arctan2(agent_or[-1], agent_or[0]) # current orientation angle zero facing right
    vel = np.zeros(3)
    if task == 'CHASE':
        opponent_to_agent = vec_opponent_to_agent(pose, opponent_pose)
        dist_to_opp = np.linalg.norm(opponent_to_agent)
        if dist_to_opp < 5.0: # if close just set opponent as target
            vel = - opponent_to_agent
            # vel_n = target_speed * self.normalize(vel)
            # self.target_y_vel = vel[1]
            # self.target_x_vel = vel[0]
            return opponent_pose
        else:
            vel = chase_vel(pose, opponent_pose, theta)
            # self.target_y_vel = vel[1]
            # self.target_x_vel = vel[0]
            target_dist_scaling = 3.0
    elif task == 'EVADE':
        vel = evade_vel(pose, opponent_pose, theta)
        # self.target_y_vel = vel[1]
        # self.target_x_vel = vel[0]
        target_dist_scaling = 2.0
    virtual_target = np.array([pose[0] + target_dist_scaling * vel[0], pose[1] + target_dist_scaling * vel[1], theta + vel[2]])
    virtual_target =  np.clip(virtual_target, -5.0, 5.0)
    # get current opponent pos
    # opponent_pose = self.opponent.get_opponent_pose()[:].copy()
    return virtual_target

def obsdict2obsvec(obs_dict, ordered_obs_keys):
    """
    Create observation vector from obs_dict
    """
    obsvec = np.zeros(0)
    for key in ordered_obs_keys:
        obsvec = np.concatenate([obsvec, obs_dict[key].ravel()]) # ravel helps with images
    return obsvec


def get_custom_observation(env, virtual_traj=False):
    """
    Use this function to create an observation vector from the 
    environment provided observation dict for your own policy.
    By using the same keys as in your local training, you can ensure that 
    your observation still works.
    """
    # example of obs_keys for deprl baseline
    obs_keys = [
      'internal_qpos',
      'internal_qvel',
      'grf',
      'torso_angle',
      'opponent_pose',
      'opponent_vel',
      'model_root_pos',
      'model_root_vel',
      'muscle_length',
      'muscle_velocity',
      'muscle_force',
      'hfield',
      'act',
      'task'
    ]

    # obs_dict = rc.get_obsdict()
    obs_dict = env.get_obs_dict(env.sim)
    # add new features here that can be computed from obs_dict
    # obs_dict['qpos_without_xy'] = np.array(obs_dict['internal_qpos'][2:35].copy())
    if virtual_traj: # both cases virtual
        new_target = get_target_pos(obs["model_root_pos"], obs["opponent_pose"])[:].copy() 
        obs_dict['opponent_pose'] = new_target
    elif obs_dict['task'] == 'EVADE': # always for evade
        new_target = get_target_pos(obs["model_root_pos"], obs["opponent_pose"])[:].copy() 
        obs_dict['opponent_pose'] = new_target 
    return obsdict2obsvec(obs_dict, obs_keys)



################################################
## A -replace with your trained policy.
## HERE an example from a previously trained policy with deprl is shown (see https://github.com/facebookresearch/myosuite/blob/main/docs/source/tutorials/4a_deprl.ipynb)
## additional dependencies such as gym and deprl might be needed
class Agent:
    def __init__(self, model, env_norm):
        self.model = model
        self.env_norm = env_norm
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        self.arena_size = 6

        
    def reset(self):
        self.episode_starts = np.ones((1,), dtype=bool)
        self.lstm_states = None
        
    
    def get_action(self, obs):
        action, self.lstm_states = self.model.predict(
                self.env_norm.normalize_obs(obs),
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=True,
            )
        self.episode_starts = False
        return action



EXPERIMENT_PATH_VT = os.path.join(ROOT_DIR, "output/training/2023-11-04/CustomChaseTag_seed_15_x_-5.5_5.5_y_-5.5_5.5_dist_0.001_hip_0.0_period_100.0_alive_0.0_solved_1000.0_early_solved_0.0_joints_0.0_lose_-1.0_ref_0.01_heel_0.0_gait_l_0.9_gait_c_0.01_fix_0.1_ran_0.1_mov_0.8_traj_virtual_trajmyo-train-ad-p2-7-2-2")
CHECKPOINT_NUM_VT = 1180000000

EXPERIMENT_PATH_C = os.path.join(ROOT_DIR, "output/training/2023-11-04/CustomChaseTag_seed_13_x_-4.0_4.0_y_-6.0_1.0_dist_0.2_hip_0.0_period_100.0_alive_0.0_solved_0.0_early_solved_1.0_joints_0.0_lose_-10.0_ref_0.02_heel_0.0_gait_l_0.8_gait_c_0.01_fix_0.1_ran_0.45_mov_0.45_job_177")
CHECKPOINT_NUM_C = 552000000

EXPERIMENT_PATH_E = os.path.join(ROOT_DIR, "output/training/2023-11-04/CustomChaseTag_seed_15_x_-5.5_5.5_y_-5.5_5.5_dist_0.01_hip_0.0_period_100.0_alive_0.0_solved_1.0_early_solved_0.0_joints_0.0_lose_-10.0_ref_0.01_heel_0.0_gait_l_0.9_gait_c_0.01_fix_0.1_ran_0.45_mov_0.45_traj_virtual_trajmyo-train-ad-p2-6")
CHECKPOINT_NUM_E = 1192000000


if __name__ == "__main__":
    
    virtual_traj = True # if True use same model for chase and evade
    
    if virtual_traj:
        print('Loaded this: ',EXPERIMENT_PATH_VT,CHECKPOINT_NUM_VT)
        model = load_model(EXPERIMENT_PATH_VT, CHECKPOINT_NUM_VT)
        config_path = os.path.join(EXPERIMENT_PATH_VT, "env_config.json")
        env_config = json.load(open(config_path, "r"))
        base_env = EnvironmentFactory.create(**env_config)
        envs = load_vecnormalize(EXPERIMENT_PATH_VT, CHECKPOINT_NUM_VT, base_env)
        envs.training = False
        pi = Agent(model, envs)
    else: # two different
        print('Loaded this: ',EXPERIMENT_PATH_C,CHECKPOINT_NUM_C)
        modelC = load_model(EXPERIMENT_PATH_C, CHECKPOINT_NUM_C)
        config_path = os.path.join(EXPERIMENT_PATH_C, "env_config.json")
        env_config = json.load(open(config_path, "r"))
        base_env = EnvironmentFactory.create(**env_config)
        envsC = load_vecnormalize(EXPERIMENT_PATH_C, CHECKPOINT_NUM_C, base_env)
        envsC.training = False
        piChase = Agent(modelC, envsC)

        print('Loaded this: ',EXPERIMENT_PATH_E,CHECKPOINT_NUM_E)
        modelE = load_model(EXPERIMENT_PATH_E, CHECKPOINT_NUM_E)
        config_path = os.path.join(EXPERIMENT_PATH_E, "env_config.json")
        env_config = json.load(open(config_path, "r"))
        base_env = EnvironmentFactory.create(**env_config)
        envsE = load_vecnormalize(EXPERIMENT_PATH_E, CHECKPOINT_NUM_E, base_env)
        envsE.training = False
        piEvade = Agent(modelE, envsE) 

    # print('Loaded this: ',EXPERIMENT_PATH,CHECKPOINT_NUM)
    # model = load_model(EXPERIMENT_PATH, CHECKPOINT_NUM)
    
    # config_path = os.path.join(EXPERIMENT_PATH, "env_config.json")
    # env_config = json.load(open(config_path, "r"))
    # norm_env = EnvironmentFactory.create(**env_config)

    env_config_base= {"env_name":"ChaseTagEnvPhase2", "seed":0}
    base_env = EnvironmentFactory.create(**env_config_base)
 
    # envs = load_vecnormalize(EXPERIMENT_PATH, CHECKPOINT_NUM, norm_env)
    # envs.training = False

    # pi = Agent(model, envs)
    ################################################
    flag_completed = None # this flag will detect then the whole eval is finished
    repetition = 0
    episodes = 0
    perfs = []
    solved = []
    while not flag_completed:
        flag_trial = None # this flag will detect the end of an episode/trial
        counter = 0
        cum_reward = 0
        repetition +=1
        while not flag_trial :

            if counter == 0:
                print('RELOCATE: Trial #'+str(repetition)+'Start Resetting the environment and get 1st obs')
                # obs = rc.reset()
                base_env.reset()

            ################################################
            ### B - HERE the action is obtained from the policy and passed to the remote environment
            obs = get_custom_observation(base_env)
            # breakpoint()
            task = obs[-1]
            obs = obs[:-1]

            if not virtual_traj:
                if task == 'CHASE':
                    # use chase environement 
                    pi = piChase
                elif task == 'EVADE':
                    # use evade enc
                    pi = piEvade

            if counter == 0:
                pi.reset()
                
            action = pi.get_action(obs)
            ################################################

            ## gets info from the environment
            obs, rewards, done, info = base_env.step(action)

            flag_trial = done #base["feedback"][2]
            counter +=1
            cum_reward += rewards
        
        print(info["rwd_dict"])
        print('Solved? ', info["rwd_dict"]['solved'])
        episodes+= 1
        perfs.append(cum_reward)
        if info["rwd_dict"]['solved'] == 1:
            solved.append(info["rwd_dict"]['solved'])
        else:
            solved.append(0)

        if (episodes) % 10 == 0:
            perf_error = np.std(perfs) / np.sqrt(episodes + 1)
            solved_error = np.std(solved) / np.sqrt(episodes + 1)

            print(f"\nEpisode {episodes+1}")
            print(f"Average rew: {np.mean(perfs):.2f} +/- {perf_error:.2f}\n")
            print(f"Average solved: {np.sum(solved)/(episodes):.2f}\n")