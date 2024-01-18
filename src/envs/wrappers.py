"""
Credits to https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

from typing import Tuple

import gym
import numpy as np
from PIL import Image
import homegrid


def make_atari(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=4, done_on_life_loss=False, clip_reward=False):
    print('id', id)
    if id == "homegrid-task":
        env = gym.make(id)
    else:
        env = gym.make(id)
    print('using env in make_atari', id)
    print('the info of the env is ', env)
    print('spec.id ', env.spec.id)
    # assert 'NoFrameskip' in env.spec.id or 'Frameskip' not in env.spec
    env = ResizeObsWrapper(env, (size, size))
    print('after resize, the env info is as: ', env)
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
        return {'image': self.resize(observation), 'token':observation['token']}
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
        print(indices)
        return {'image': self._obs_buffer[indices], 'token':self._obs_token_buffer[indices]}, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
