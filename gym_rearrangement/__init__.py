import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='maze-v0',
    entry_point='gym_rearrangement.envs:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='maze-sample-5x5-v0',
    entry_point='gym_rearrangement.envs:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='maze-random-5x5-v0',
    entry_point='gym_rearrangement.envs:MazeEnvRandom5x5',
    max_episode_steps=2000,
    nondeterministic=True,
)

register(
    id='maze-sample-10x10-v0',
    entry_point='gym_rearrangement.envs:MazeEnvSample10x10',
    max_episode_steps=10000,
)

register(
    id='maze-random-10x10-v0',
    entry_point='gym_rearrangement.envs:MazeEnvRandom10x10',
    max_episode_steps=10000,
    nondeterministic=True,
)

register(
    id='maze-sample-3x3-v0',
    entry_point='gym_rearrangement.envs:MazeEnvSample3x3',
    max_episode_steps=1000,
)

register(
    id='maze-random-3x3-v0',
    entry_point='gym_rearrangement.envs:MazeEnvRandom3x3',
    max_episode_steps=1000,
    nondeterministic=True,
)

register(
    id='maze-sample-100x100-v0',
    entry_point='gym_rearrangement.envs:MazeEnvSample100x100',
    max_episode_steps=1000000,
)

register(
    id='maze-random-100x100-v0',
    entry_point='gym_rearrangement.envs:MazeEnvRandom100x100',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-10x10-plus-v0',
    entry_point='gym_rearrangement.envs:MazeEnvRandom10x10Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-20x20-plus-v0',
    entry_point='gym_rearrangement.envs:MazeEnvRandom20x20Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-30x30-plus-v0',
    entry_point='gym_rearrangement.envs:MazeEnvRandom30x30Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)


# ================robotics==========================

def _merge(a, b):
    a.update(b)
    return a


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    # Fetch
    register(
        id='FetchSlide{}-v2'.format(suffix),
        entry_point='gym_rearrangement.envs:FetchSlideEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchReach{}-v2'.format(suffix),
        entry_point='gym_rearrangement.envs:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPush{}-v2'.format(suffix),
        entry_point='gym_rearrangement.envs:FetchPushEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickPlace{}-v2'.format(suffix),
        entry_point='gym_rearrangement.envs:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
