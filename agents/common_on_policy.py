import numpy as np
import time
from agents.common_all import Agent

class OnPolicyAgent(Agent):
    def __init__(
            self,
            env,
            batch_size,
            max_path_frames,
            render_interval=10,
    ):
        super(OnPolicyAgent, self).__init__(
            env,
        )
        self.render_interval = render_interval
        self.min_frames_per_iter = batch_size
        self.max_path_frames = max_path_frames

        # utilities for collect_rollouts()
        self._paths = []
        self._frame = 0

        # logging
        self._start = time.time()
        self._itr_start_time = self._start
        self._frames_last_iter = 0

    def get_episode_lengths(self):
        return [len(path["rewards"]) for path in self._paths]

    def get_episode_returns(self):
        return [path["rewards"].sum() for path in self._paths]

    def get_paths(self):
        """
        :return: dict where each entry is a list (len: num_episodes) 
        of numpy arrays (shape: [num_timesteps, dim])
        """
        data = {}
        for key in self._paths[0].keys():
            data[key] = [path[key] for path in self._paths]
        return data

    def log_progress(self):
        returns = self.get_episode_returns()
        ep_lengths = self.get_episode_lengths()
        stats = {
            'Time': (time.time() - self._start) / 60.,
            'TimestepsSoFar': self._frame,
            'TimestepsThisBatch': self._frames_last_iter,
            'fps': (self._frames_last_iter / (time.time() - self._itr_start_time)),
            'AverageReturn': np.mean(returns),
            'StdReturn': np.std(returns),
            'MaxReturn': np.max(returns),
            'MinReturn': np.min(returns),
            'EpLenMean': np.mean(ep_lengths),
            'EpLenStd': np.std(ep_lengths),
        }
        self._itr_start_time = time.time()
        return stats      

    def collect_rollouts(self, itr, render=False):
        frames_this_iter = 0
        paths = []

        while True:
            render = (frames_this_iter == 0 and (itr % self.render_interval == 0) and render)
            path = self._collect_episode(render)    # collects one episode
            paths.append(path)
            pathlength = len(path[list(path)[0]])
            frames_this_iter += pathlength
            if frames_this_iter > self.min_frames_per_iter:
                break

        self._paths = paths
        self._frame += frames_this_iter
        self._frames_last_iter = frames_this_iter

    def _collect_episode(self, render, ):
        if render and self.env.is_bullet:
            self.env.render(mode="human", close=True)
        ob = self.env.reset()
        obs, actions, rewards = [], [], []
        episode_frames = 0

        while True:
            if render and not self.env.is_bullet:
                self.env.render()
                # time.sleep(0.1)
            obs.append(ob)
            ac = self.act(ob)  # TODO add noise
            actions.append(ac)
            ob, rew, done, _ = self.env.step(ac)
            rewards.append(rew)
            episode_frames += 1
            if done or episode_frames > self.max_path_frames - 1:
                break

        path = {"observations" : np.array(obs, dtype=np.float32), 
                "rewards" : np.array(rewards, dtype=np.float32), 
                "actions" : np.array(actions)}
        if render and not self.env.is_bullet:
            self.env.close()        
        return path

        



        