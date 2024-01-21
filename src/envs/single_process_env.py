from typing import Any, Tuple

import numpy as np

from .done_tracker import DoneTrackerEnv


class SingleProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn):
        super().__init__(num_envs=1)
        self.env = env_fn()
        self.num_actions = self.env.action_space.n

    def should_reset(self) -> bool:
        return self.num_envs_done == 1

    def reset(self) -> dict:
        self.reset_done_tracker()
        obs = self.env.reset()
        img = obs['image']
        img = img[None, ...]
        tok = obs['token']
        return {'image':img, 'token':tok}

    def step(self, action) -> Tuple[dict, np.ndarray, np.ndarray, Any]:
        tmp = self.env.step(action[0])  # action is supposed to be ndarray (1,)
        if len(tmp ) == 4:
            obs, reward, done, _ = tmp
        else:
            obs,reward, done, trun, _ = tmp
            done = done or trun
        done = np.array([done])
        self.update_done_tracker(done)
        img = obs['image']
        img = img[None, ...]
        tok = obs['token']
        # print('token in single_process_env', tok)
        return {'image':img, 'token':tok}, np.array([reward]), done, None

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()
