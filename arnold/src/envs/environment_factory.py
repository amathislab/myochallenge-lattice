import gym
# import pybullet_envs
# from .isaacgym_envs.envs.shadow_hand_reach import ShadowHandReachEnv
# from .isaacgym_envs.envs.shadow_hand_finger_reach import ShadowHandFingerReachEnv
# from .isaacgym_envs.envs.shadow_hand_reorient import ShadowHandReorientEnv

class EnvironmentFactory:
    """Static factory to instantiate and register gym environments by name."""

    @staticmethod
    def create(env_name, **kwargs):
        """Creates an environment given its name as a string, and forwards the kwargs
        to its __init__ function.

        Args:
            env_name (str): name of the environment

        Raises:
            ValueError: if the name of the environment is unknown

        Returns:
            gym.env: the selected environment
        """
        # make myosuite envs
        if env_name == "MyoFingerPoseFixed":
            return gym.make("myoFingerPoseFixed-v0")
        elif env_name == "MyoFingerPoseRandom":
            return gym.make("myoFingerPoseRandom-v0")
        elif env_name == "MyoFingerReachFixed":
            return gym.make("myoFingerReachFixed-v0")
        elif env_name == "MyoFingerReachRandom":
            return gym.make("myoFingerReachRandom-v0")
        elif env_name == "MyoHandPoseFixed":
            return gym.make("myoHandPoseFixed-v0")
        elif env_name == "MyoHandPoseRandom":
            return gym.make("myoHandPoseRandom-v0")
        elif env_name == "MyoHandReachFixed":
            return gym.make("myoHandReachFixed-v0")
        elif env_name == "MyoHandReachRandom":
            return gym.make("myoHandReachRandom-v0")
        elif env_name == "MyoElbowPoseFixed":
            return gym.make("myoElbowPose1D6MFixed-v0")
        elif env_name == "MyoElbowPoseRandom":
            return gym.make("myoElbowPose1D6MRandom-v0")
        elif env_name == "MyoHandKeyTurnFixed":
            return gym.make("myoHandKeyTurnFixed-v0")
        elif env_name == "MyoHandKeyTurnRandom":
            return gym.make("myoHandKeyTurnRandom-v0")
        elif env_name == "MyoBaodingBallsP1":
            return gym.make("myoChallengeBaodingP1-v1")
        elif env_name == "CustomMyoBaodingBallsP1":
            return gym.make("CustomMyoChallengeBaodingP1-v1", **kwargs)
        elif env_name == "CustomMyoReorientP1":
            return gym.make("CustomMyoChallengeDieReorientP1-v0", **kwargs)
        elif env_name == "CustomMyoReorientP2":
            return gym.make("CustomMyoChallengeDieReorientP2-v0", **kwargs)
        elif env_name == "MyoBaodingBallsP2":
            return gym.make("myoChallengeBaodingP2-v1", **kwargs)
        elif env_name == "CustomMyoBaodingBallsP2":
            return gym.make("CustomMyoChallengeBaodingP2-v1", **kwargs)
        elif env_name == "MixtureModelBaodingEnv":
            return gym.make("MixtureModelBaoding-v1", **kwargs)
        elif env_name == "CustomMyoElbowPoseFixed":
            return gym.make("CustomMyoElbowPoseFixed-v0", **kwargs)
        elif env_name == "CustomMyoElbowPoseRandom":
            return gym.make("CustomMyoElbowPoseRandom-v0", **kwargs)
        elif env_name == "CustomMyoFingerPoseFixed":
            return gym.make("CustomMyoFingerPoseFixed-v0", **kwargs)
        elif env_name == "CustomMyoFingerPoseRandom":
            return gym.make("CustomMyoFingerPoseRandom-v0", **kwargs)
        elif env_name == "CustomMyoHandPoseFixed":
            return gym.make("CustomMyoHandPoseFixed-v0", **kwargs)
        elif env_name == "CustomMyoHandPoseRandom":
            return gym.make("CustomMyoHandPoseRandom-v0", **kwargs)
        elif env_name == "CustomMyoPenTwirlRandom":
            return gym.make("CustomMyoHandPenTwirlRandom-v0", **kwargs)
        elif env_name == "CustomChaseTag":
            return gym.make("CustomChaseTagEnv-v0", **kwargs)
            
        # Muscle environments
        elif env_name == "MuscleElbowPoseFixed":
            return gym.make("MuscleElbowPoseFixed-v0", **kwargs)
        elif env_name == "MuscleElbowPoseRandom":
            return gym.make("MuscleElbowPoseRandom-v0", **kwargs)
        elif env_name == "MuscleFingerPoseFixed":
            return gym.make("MuscleFingerPoseFixed-v0", **kwargs)
        elif env_name == "MuscleFingerPoseRandom":
            return gym.make("MuscleFingerPoseRandom-v0", **kwargs)
        elif env_name == "MuscleHandPoseFixed":
            return gym.make("MuscleHandPoseFixed-v0", **kwargs)
        elif env_name == "MuscleHandPoseRandom":
            return gym.make("MuscleHandPoseRandom-v0", **kwargs)
        elif env_name == "MuscleHandPoseRandomHalfRange":
            return gym.make("MuscleHandPoseRandomHalfRange-v0", **kwargs)
        elif env_name == "MuscleFingerReachFixed":
            return gym.make("MuscleFingerReachFixed-v0", **kwargs)
        elif env_name == "MuscleFingerReachRandom":
            return gym.make("MuscleFingerReachRandom-v0", **kwargs)
        elif env_name == "MuscleHandReachFixed":
            return gym.make("MuscleHandReachFixed-v0", **kwargs)
        elif env_name == "MuscleHandReachRandom":
            return gym.make("MuscleHandReachRandom-v0", **kwargs)
        elif env_name == "MuscleBaodingEnvP0":
            return gym.make("MuscleBaodingP0-v1", **kwargs)
        elif env_name == "MuscleBaodingEnvP1":
            return gym.make("MuscleBaodingP1-v1", **kwargs)
        elif env_name == "MuscleBaodingEnvP2":
            return gym.make("MuscleBaodingP2-v1", **kwargs)
        elif env_name == "MuscleBaodingEnvP3":
            return gym.make("MuscleBaodingP3-v1", **kwargs)
        elif env_name == "MuscleReorientEnvP0":
            return gym.make("MuscleDieReorientP0-v0", **kwargs)
        elif env_name == "MuscleReorientEnvP1":
            return gym.make("MuscleDieReorientP1-v0", **kwargs)
        elif env_name == "MuscleReorientEnvP2":
            return gym.make("MuscleDieReorientP2-v0", **kwargs)
        elif env_name == "MuscleLegsStandEnv":
            return gym.make("MuscleLegDemo-v0", **kwargs)
        elif env_name == "MuscleLegsWalkEnv":
            return gym.make("MuscleLegWalk-v0", **kwargs)
        
        # Isaacgym environments
        elif env_name == "IsaacgymHandReach" :
            return ShadowHandReachEnv(**kwargs)
        elif env_name == "IsaacgymFingerReach" :
            return ShadowHandFingerReachEnv(**kwargs)
        elif env_name == "IsaacgymReorient" :
            return ShadowHandReorientEnv(**kwargs)

        # PyBullet environments
        elif env_name == "WalkerBulletEnv":
            return gym.make("Walker2DBulletEnv-v0", **kwargs)
        elif env_name == "HalfCheetahBulletEnv":
            return gym.make("HalfCheetahBulletEnv-v0", **kwargs)
        elif env_name == "AntBulletEnv":
            return gym.make("AntBulletEnv-v0", **kwargs)
        elif env_name == "HopperBulletEnv":
            return gym.make("HopperBulletEnv-v0", **kwargs)
        elif env_name == "HumanoidBulletEnv":
            return gym.make("HumanoidBulletEnv-v0", **kwargs)
        elif env_name == "HumanoidFlagrunBulletEnv":
            return gym.make("HumanoidFlagrunBulletEnv-v0", **kwargs)
        elif env_name == "HumanoidFlagrunHarderBulletEnv":
            return gym.make("HumanoidFlagrunHarderBulletEnv-v0", **kwargs)
        else:
            raise ValueError("Environment name not recognized:", env_name)