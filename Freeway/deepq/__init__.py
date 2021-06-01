from deepq.policies import MlpPolicy, CnnPolicy, LnMlpPolicy, LnCnnPolicy
from deepq.build_graph import build_act, build_train  # noqa
from deepq.dqn import DQN
from utils.common.buffers import ReplayBuffer, PrioritizedReplayBuffer  # noqa


def wrap_atari_dqn(env):
    """
    wrap the environment in atari wrappers for DQN

    :param env: (Gym Environment) the environment
    :return: (Gym Environment) the wrapped environment
    """
    from utils.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
