import datetime
import io
import random
import traceback
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
	# subtract -1 because the dummy first transition
	return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
	with io.BytesIO() as bs:
		np.savez_compressed(bs, **episode)
		bs.seek(0)
		with fn.open('wb') as f:
			f.write(bs.read())


def load_episode(fn):
	with fn.open('rb') as f:
		episode = np.load(f)
		episode = {k: episode[k] for k in episode.keys()}
		return episode


def relable_episode(env, episode):   # relabel the reward function
	rewards = []
	reward_spec = env.reward_spec()
	states = episode['physics']
	for i in range(states.shape[0]):
		with env.physics.reset_context():
			env.physics.set_state(states[i])
		# Input: the current physics of env, then use env.task.get_reward to calculate the reward
		reward = env.task.get_reward(env.physics)
		reward = np.full(reward_spec.shape, reward, reward_spec.dtype)  # 改变shape和dtype
		rewards.append(reward)
	original_reward = np.mean(episode['reward'])
	episode['reward'] = np.array(rewards, dtype=reward_spec.dtype)
	# print("Reward difference after relabeling:", original_reward - episode['reward'].mean())
	return episode


class OfflineReplayBuffer(IterableDataset):
	# 用于 offline training 的 dataset
	def __init__(self, env, replay_dir_list, max_size, num_workers, discount, main_task, task_list):
		self._env = env
		self._size = 0
		self._max_size = max_size
		self._episode_fns = []
		self._episodes = dict()    # save as episode
		self._discount = discount
		self._loaded = False
		self._main_task = main_task
		self._task_list = task_list

	def _load(self, relable=True):
		print("load data", self._replay_dir_list)
		for i in range(len(self._replay_dir_list)):       # loop
			_replay_dir = self._replay_dir_list[i]
			_task_share = self._task_list[i]
			assert _task_share in str(_replay_dir)
			try:
				worker_id = torch.utils.data.get_worker_info().id
			except:
				worker_id = 0
			print(f'Loading data from {_replay_dir} and Relabel...', "worker_id:", worker_id)      # each worker will run this function
			print(f"Need relabeling: {relable and _task_share != self._main_task}")
			eps_fns = sorted(_replay_dir.glob('*.npz'))
			for eps_fn in eps_fns:
				if self._size > self._max_size:
					break
				eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
				if eps_idx % self._num_workers != worker_id:  # read the npz file of the worker
					continue
				# load a npz file to represent an episodic sample. The keys include 'observation', 'action', 'reward', 'discount', 'physics'
				episode = load_episode(eps_fn)

				if relable and _task_share != self._main_task:
					# print(f"relabel {_replay_dir} for {self._main_task} task")
					episode = self._relable_reward(episode)   # relabel
				data_flag = _task_share+str(eps_fn)+str(_task_share == self._main_task)
				self._episode_fns.append(data_flag)
				self._episodes[data_flag] = episode
				self._size += episode_len(episode)
				# if worker_id == 0:
				# 	print("data_flag:", data_flag)
		print("load done. Num of episodes", len(self._episode_fns)*self._num_workers)

	def _sample_episode(self):
		if not self._loaded:
			self._load()
			self._loaded = True
		eps_fn = random.choice(self._episode_fns)
		return self._episodes[eps_fn], eps_fn.endswith("True")   # whether is the main buffer

	def _relable_reward(self, episode):
		return relable_episode(self._env, episode)

	def _sample(self):
		episode, eps_flag = self._sample_episode()   # return the signal
		# add +1 for the first dummy transition
		idx = np.random.randint(0, episode_len(episode)) + 1
		obs = episode['observation'][idx - 1]
		action = episode['action'][idx]
		next_obs = episode['observation'][idx]
		reward = episode['reward'][idx]
		discount = episode['discount'][idx] * self._discount
		#not_done = (idx != episode_len(episode) -1)

		return (obs, action, reward, discount, next_obs, True, bool(eps_flag))

	def __iter__(self):
		while True:
			yield self._sample()


def _worker_init_fn(worker_id):
	seed = np.random.get_state()[1][0] + worker_id
	np.random.seed(seed)
	random.seed(seed)


def make_replay_loader(env, replay_dir_list, max_size, batch_size, num_workers, discount, main_task, task_list):
	max_size_per_worker = max_size // max(1, num_workers)

	iterable = OfflineReplayBuffer(env, replay_dir_list, max_size_per_worker,
								   num_workers, discount, main_task, task_list)      # task 表示主任务

	loader = torch.utils.data.DataLoader(iterable,
										 batch_size=batch_size,
										 num_workers=num_workers,
										 pin_memory=True,
										 worker_init_fn=_worker_init_fn)
	return loader

