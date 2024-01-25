"""
Credits to https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

from typing import Tuple

import gym
import numpy as np
from PIL import Image
import homegrid

tasks_dict = {'find the bottle': 0, 'move the papers to the dining room': 1, 'open the recycling bin': 2,
              'find the recycling bin': 3, 'find the papers': 4, 'get the papers': 5,
              'move the bottle to the dining room': 6, 'open the trash bin': 7,
              'put the papers in the trash bin': 8, 'get the plates': 9, 'move the bottle to the kitchen': 10,
              'move the papers to the living room': 11, 'put the fruit in the compost bin': 12,
              'get the fruit': 13, 'move the plates to the kitchen': 14, 'find the fruit': 15,
              'get the bottle': 16, 'find the compost bin': 17, 'move the fruit to the dining room': 18,
              'move the papers to the kitchen': 19, 'put the plates in the recycling bin': 20,
              'find the plates': 21, 'put the bottle in the compost bin': 22, 'open the compost bin': 23,
              'move the bottle to the living room': 24, 'put the fruit in the trash bin': 25,
              'put the bottle in the trash bin': 26, 'put the papers in the compost bin': 27,
              'put the plates in the compost bin': 28, 'move the plates to the living room': 29,
              'move the plates to the dining room': 30, 'move the fruit to the kitchen': 31,
              'find the trash bin': 32, 'put the papers in the recycling bin': 33,
              'put the bottle in the recycling bin': 34, 'put the fruit in the recycling bin': 35,
              'put the plates in the trash bin': 36}
def make_atari(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=4, done_on_life_loss=False, clip_reward=False):
    # print('id', id)
    from gym.envs.registration import register

    sym = {'step': 0, 'agent': {'pos': (5, 9), 'room': 'K', 'dir': 1, 'carrying': None}, 'objects': [
        {'name': 'recycling bin', 'type': 'Storage', 'pos': (12, 10), 'room': 'D', 'state': 'open', 'action': 'lift',
         'invisible': None, 'contains': []},
        {'name': 'trash bin', 'type': 'Storage', 'pos': (11, 1), 'room': 'L', 'state': 'closed', 'action': 'lift',
         'invisible': None, 'contains': []},
        {'name': 'fruit', 'type': 'Pickable', 'pos': (8, 2), 'room': 'L', 'state': None, 'action': None,
         'invisible': False, 'contains': None},
        {'name': 'bottle', 'type': 'Pickable', 'pos': (12, 2), 'room': 'L', 'state': None, 'action': None,
         'invisible': False, 'contains': None}], 'front_obj': None, 'unsafe': {'name': None, 'poss': {}, 'end': -1}}

    register(
        id="homegrid-fix",
        entry_point="homegrid:HomeGrid",
        kwargs={"lang_types": ["task"],
                'fixed_state': sym},
    )

    if id == "homegrid-task":
        env = gym.make(id)
    elif id == "homegrid-fix":
        print(id)
        env = gym.make(id)

    else:
        env = gym.make(id)
    # print('using env in make_atari', id)
    # print('the info of the env is ', env)
    # print('spec.id ', env.spec.id)
    # assert 'NoFrameskip' in env.spec.id or 'Frameskip' not in env.spec
    env = ResizeObsWrapper(env, (size, size))
    # print('after resize, the env info is as: ', env)
    if clip_reward:
        env = RewardClippingWrapper(env)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    # env = MaxAndSkipEnv(env, skip=frame_skip) # todo wrong delete
    if done_on_life_loss:
        env = EpisodicLifeEnv(env)
    return env


class ResizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]) -> None:
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
        self.unwrapped.original_obs = None
        self.unwrapped.original_token = None
        self.latent_dict = {474: 0, 8: 1, 5778: 2, 16: 3, 17883: 4, 2701: 5, 0: 6, 888: 7, 2728: 8, 12: 9, 1228: 10, 129: 11, 539: 12, 11667: 13, 2851: 14, 562: 15, 253: 16, 7623: 17, 9319: 18, 4782:19, 840:20}
        self.hist_cnt = len(self.latent_dict)

    def resize(self, obs: np.ndarray):
        if isinstance(obs, dict):
            obs_t = obs['image']
        img = Image.fromarray(obs_t)
        img = img.resize(self.size, Image.BILINEAR)
        # return {'image':np.array(img),  'token': obs['token']}
        return np.array(img)
    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.unwrapped.original_obs = observation['image']
        self.unwrapped.original_token = observation['token']
        # print('---- used observation')
        # using key to get the image
        # print('observation token in wrapper ', observation['token'])


        tmp = observation['log_language_info']
        if tmp in tasks_dict.keys():
            tok = tasks_dict[tmp]
            self.history_id = tok
        else:
            tok = self.history_id

        if tok == None:
            tok = 38 # no task
        print(tok)
        return {'image': self.resize(observation), 'token':tok}
        # return self.resize(observation)

class RewardClippingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        # assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            tmp = self.env.step(self.noop_action)
            # print(tmp)
            # for i in tmp:
            #     print('-------')
            #     print(i)
            # print(len(tmp), '------------------')
            obs, _, done, trun, _ = tmp
            done = done or trun
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        # tmp = self.env.step(action)
        # for i in tmp:
        #     print(tmp)
        # print('tmp_len', len(tmp))
        obs, reward, done, trun,  info = self.env.step(action)
        done = trun or done
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        # lives = self.env.unwrapped.ale.lives()
        # if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            # done = True
        # self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, _= self.env.step(0)
        # self.lives = self.env.unwrapped.ale.lives()
        # self.lives = self.env.unwrapped
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        assert skip > 0
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        # print('self._obs_buffer ', self._obs_buffer.shape)
        # print('env.observation_space.shape', env.observation_space.shape)
        self._obs_token_buffer = [None, None]
        self._skip = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done,trun, info = self.env.step(action)
            done = done or trun
            if i == self._skip - 2:
                self._obs_buffer[0] = obs['image']
                self._obs_token_buffer[0] = obs['token']
            if i == self._skip - 1:
                self._obs_buffer[1] = obs['image']
                self._obs_token_buffer[1] = obs['token']
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        # self.max_frame = self._obs_buffer.max(axis=0)
        indices = np.argmax(self._obs_buffer, axis=0)
        # print(indices)
        return {'image': self._obs_buffer[indices], 'token':self._obs_token_buffer[indices]}, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
