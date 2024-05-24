from agents.envs.env_utils import NoisytargetEnv
from agents.envs.env_utils import TimeLimit, NormalizedEnv
from agents.envs.pendulum.Pendulum import PendulumEnv
from agents.envs.friction_mujoco.create_friction_mujoco import get_friction_mujoco
from agents.envs.broken_joint_mujoco.create_broken_joint_mujoco import get_broken_joint_mujoco
from agents.envs.highly_crippled_mujoco.create_highly_crippled_mujoco import get_highly_crippled_mujoco
from agents.envs.foot_friction_mujoco.create_foot_friction_mujoco import get_foot_friction_mujoco




def get_gravity_pendulum(mode, source_max_episode_steps=200, target_max_episode_steps=500, gravity=10):
    if mode == 'target':
        env = TimeLimit(NormalizedEnv(PendulumEnv(g=gravity, early_done=True)),
                        max_episode_steps=target_max_episode_steps)
    elif mode == 'evaluation':
        env = TimeLimit(NormalizedEnv(PendulumEnv(g=gravity)),
                        max_episode_steps=target_max_episode_steps)
    else:
        env = TimeLimit(NormalizedEnv(PendulumEnv()),
                        max_episode_steps=source_max_episode_steps)
    return env


def get_classic_minitaur_env(mode):
    import gym
    from gym.wrappers.clip_action import ClipAction
    if mode == 'source':
        env = gym.make('MinitaurLinearBulletEnv-v0')
    else:
        env = gym.make('MinitaurBulletEnv-v0')
    return ClipAction(env)


def get_heavy_env(mode):
    import gym
    if mode == 'source':
        env = gym.make('HalfCheetah-v2')
    else:
        env = gym.make('heavy_HalfCheetah-v2')
    return env

def get_env(env_name,
            mode,
            source_max_episode_steps=200,
            target_max_episode_steps=300,
            broken_joint=0,
            broken_joints=(0, 1),
            orig_cwd='./',
            noisy_env=False,
            noise_std=0.0,
            **kwargs
            ):
    """
    Get environment, depending if it is source or target env
    """
    if env_name == 'gravity_pendulum':
        env = get_gravity_pendulum(mode,
                                   source_max_episode_steps=source_max_episode_steps,
                                   target_max_episode_steps=target_max_episode_steps,
                                   gravity=kwargs['gravity'])
    elif env_name == 'MinitaurLinear':
        env = get_classic_minitaur_env(mode=mode)
    elif env_name in ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Ant-v2']:
        import gym
        env = gym.make(env_name)
    elif env_name.startswith("broken_joint"):
        base_name = env_name.split("broken_joint_")[1]
        env = get_broken_joint_mujoco(env_name=base_name,
                                      mode=mode,
                                      broken_joint=broken_joint)
    elif env_name.startswith("broken_leg"):
        base_name = env_name.split("broken_leg_")[1]
        env = get_highly_crippled_mujoco(env_name=base_name,
                                         mode=mode,
                                         broken_joints=broken_joints)
    elif env_name.startswith("foot_fric"):
        env = get_foot_friction_mujoco(env_name=env_name,
                                       mode=mode,
                                       foot_fric=1)
    elif env_name.startswith("heavy"):
        env = get_heavy_env(mode=mode)
    else:
        raise NotImplementedError("Unknown environment: %s" % env_name)

    if noisy_env:
        assert noise_std > 0
        return NoisytargetEnv(env, noise_value=noise_std)

    return env
