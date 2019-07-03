import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# ================robotics==========================

def _merge(a, b):
    a.update(b)
    return a


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    # Fetch Pick and Place, mirrior of the gym fetch pick and place
    register(
        id='FetchPickAndPlace{}-v2'.format(suffix),
        entry_point='gym_rearrangement.envs:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=150,
    )

    # Fetch rearrangement
    for n_object in [1, 2, 3, 4, 6]:
        kwargs = {
            'reward_type': reward_type,
            'n_object': n_object,
        }
        register(
            id='FetchRearrangement{}{}-v1'.format(n_object, suffix),
            entry_point='gym_rearrangement.envs:Rearrangement',
            kwargs=kwargs,
            max_episode_steps=400,
        )
