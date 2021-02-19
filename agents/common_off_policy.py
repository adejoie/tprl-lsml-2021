import time
import numpy as np
import torch
from agents.common_all import Agent
from agents.replay import ReplayBuffer

class OffPolicyAgent(Agent):
	def __init__(
			self,
			env,
			batch_size,
			learning_starts,	# start learning after x frames
			learning_freq,		# update model every x frames
			replay_buffer_size,
			max_path_frames,
	):
		super(OffPolicyAgent, self).__init__(
            env,
        )
		self.batch_size = batch_size
		self.learning_starts = learning_starts
		self.learning_freq = learning_freq
		self.replay_buffer = ReplayBuffer(replay_buffer_size)
		self.max_path_frames = max_path_frames

		# utilities for collect_rollouts()
		self._last_ob = self.env.reset()
		self._episode_frame = 0
		self._episode_rewards = []	# list of rewards at each timestep during an episode
		self._frame = 0

		# logging
		self._start = time.time()
		self._episode_returns = []  # list of returns (sum of rewards) for each episode
		self._episode_lengths = []  # list of lengths for each episode
		self._best_mean_episode_reward = - np.inf

	def get_episode_lengths(self):
		return self._episode_lengths

	def get_episode_returns(self):
		return self._episode_returns

	def sample_from_buffer(self):
		(states, next_states, actions, rewards, done_mask) = self.replay_buffer.sample(self.batch_size)
		return {
			'states': torch.FloatTensor(states),
			'next_states': torch.FloatTensor(next_states),
			'actions': torch.tensor(actions) if self.env.is_discrete else torch.FloatTensor(actions),
			'rewards': torch.FloatTensor(rewards),
			'done_mask': torch.tensor(done_mask),
		}

	def log_progress(self):
		mean_episode_reward = None
		if len(self._episode_returns) > 0:
			mean_episode_reward = np.mean(self._episode_returns[-100:])
			if len(self._episode_returns) > 100:
				self._best_mean_episode_reward = max(self._best_mean_episode_reward, mean_episode_reward)
		stats = {
			'Time': (time.time() - self._start) / 60.,
			'Timestep': self._frame,
			'MeanReward': mean_episode_reward,
			'BestMeanReward': self._best_mean_episode_reward,
			'Episodes': len(self._episode_returns),
		}
		return stats

	def collect_rollouts(self, itr, render=False):
		frames_this_iter = 0
		while True:
			ac = self.act(self._last_ob)
			ob, rew, done, _ = self.env.step(ac)
			self.replay_buffer.add((self._last_ob, ob, ac, rew, done)) 
			self._last_ob = ob
			self._episode_rewards.append(rew)
			frames_this_iter += 1
			self._episode_frame += 1 
			self._frame += 1	# must update here because self.act() might use it
			
			if done or self._episode_frame > self.max_path_frames - 1:
				self._last_ob = self.env.reset()
				self._episode_returns.append(sum(self._episode_rewards))
				self._episode_lengths.append(len(self._episode_rewards))
				self._episode_rewards = []
				self._episode_frame = 0	

			if self._frame > self.learning_starts - 1 and \
			   frames_this_iter > self.learning_freq - 1 and \
			   self.replay_buffer.can_sample(self.batch_size):
				break