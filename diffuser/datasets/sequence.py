from collections import namedtuple
import numpy as np
import torch
import pdb
import diffuser.utils as utils
from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset, load_dmc_dataset, load_metaworld_dataset, load_antmaze_dataset, load_maze2d_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
from scipy import stats
import os
Batch = namedtuple('Batch', 'trajectories conditions')
AugBatch = namedtuple('AugBatch', 'trajectories task')
TaskBatch = namedtuple('TaskBatch', 'trajectories conditions task value')
DTBatch = namedtuple('DTBatch', 'actions rtg observations timestep mask')
DT1Batch = namedtuple('DT1Batch', 'actions rtg observations timestep mask task')
PromptDTBatch = namedtuple('PromptDTBatch', 'DTBatch actions rtg observations timestep mask')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
MTValueBatch = namedtuple('MTValueBatch', 'trajectories conditions task values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
            #print("episode:",episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions', 'rewards']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        rewards = self.fields.normed_rewards[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([rewards, actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch
'''load offline DMC dataset'''
class DMCSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', replay_dir_list=[], task_list=[], horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=200000, termination_penalty=0, use_padding=True, seed=None, meta_world=False, maze2d=False, antmaze=False, optimal=True):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env
        self.replay_dir_list = replay_dir_list
        self.task_list = task_list
        self.reward_scale = 400.0
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.record_values = []
        if meta_world:
            itr = load_metaworld_dataset(self.replay_dir_list, self.task_list, optimal=optimal)
        elif maze2d:
            itr = load_maze2d_dataset(self.task_list)
        elif antmaze:
            itr = load_antmaze_dataset(self.task_list)
        else:
            itr = load_dmc_dataset(self.replay_dir_list, self.task_list)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        """task_counter = {
            'antmaze-umaze-diverse-v1': 0,
            'antmaze-medium-diverse-v1': 0,
            'antmaze-large-diverse-v1': 0,
        }
        task_prompts = {
            'antmaze-umaze-diverse-v1': [],
            'antmaze-medium-diverse-v1': [],
            'antmaze-large-diverse-v1': [],
        }"""
        for i, episode in enumerate(itr):
            if len(episode['rewards']) > self.max_path_length:
                # episode = {k: episode[k][:max_path_length] for k in episode.keys()}
                continue
            self.record_values.append(episode['rewards'].sum())
            data_dir = f"./dataset/f{episode['task']}"
            if not os.path.exists(data_dir):
                os.makedirs(self.data_dir)
            np.savez_compressed(os.path.join(data_dir, f'{i%2000}.npz'), **episode)
            #if len(episode['rewards']) < 1200 and task_counter[episode['task']] < 10:
            #    task_prompts[episode['task']].append(episode)
            #    task_counter[episode['task']] += 1
            """
            if i%4000>3989:
                prompt_data.append(episode)
            if i%4000==3999 and i!=0:
                np.save(os.path.join(f"./metaworld_prompts/{episode['task']}_prompt.npy"), np.array(prompt_data))
                prompt_data = []
            print(f"episode:{i}, task:{episode['task']}, Return:{episode['rewards'].sum()}")
            """
            #for key in task_prompts.keys():
            #    np.save(os.path.join(f"./antmaze_prompts/{key}_prompt.npy"), np.array(task_prompts[key]))
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions', 'rewards']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            #if i % 10: continue
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}
    def get_task_id(self,str):
        task_dict = {
            'quadruped_walk': np.array(0),
            'quadruped_jump': np.array(1),
            'quadruped_run': np.array(2),
            'quadruped_roll_fast': np.array(3)
            }
        return task_dict[str]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        rewards = self.fields.normed_rewards[path_ind, start:end]
        #observations = self.fields.observations[path_ind, start:end]
        #actions = self.fields.actions[path_ind, start:end]
        #rewards = self.fields.rewards[path_ind, start:end]
        task = np.array(self.task_list.index(self.fields.get_task(path_ind))).reshape(-1,1)#self.get_task_id(self.fields.get_task(path_ind))
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([rewards, actions, observations], axis=-1)
        #if np.any(np.isnan(trajectories)):
        #    print("True->>>")
        batch = TaskBatch(trajectories, conditions, task, 1)
        #print("Batch item load!")
        return batch
class RTGDataset(DMCSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
    def get_conditions_new(self, observations, rtg=None):
        '''
            condition on current observation for planning
        '''
        cond = {}
        for i in range(1):
            cond[i] = observations[:, i]
            #x = np.concatenate((rtg[:, 0], observations[:, 0]), axis=-1)
        #x = np.concatenate((rtg[:,0], observations[:,0]),axis=-1)
        return cond #{0: x}
    def get_conditions_v1(self, observations, actions):
        cond = {}
        for i in range(9):
            cond[i] = np.concatenate((actions[:,i],observations[:, i]),axis=-1)
        cond[9] = observations[:, 9]
        return cond #{0: x}
    def normalize(self, keys=['observations', 'actions']):#, 'observations', 'rewards'
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key == 'rewards':
                self.fields[f'normed_{key}'] = self.fields[key] #/ self.reward_scale
                continue
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def __len__(self):
        return len(self.indices)

    def discount_cumsum(self, x, gamma):
        x = x.squeeze(-1)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[:,-1] = x[:,-1]
        for t in reversed(range(x.shape[-1] - 1)):
            discount_cumsum[:,t] = x[:,t] + gamma * discount_cumsum[:,t + 1]
        return np.expand_dims(discount_cumsum, axis=-1)
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        path_inds = []
        interval = int(self.fields.n_episodes/len(self.task_list))
        for i in range(len(self.task_list)):
            path_inds.append((path_ind+interval*i)%self.fields.n_episodes)
            #print(self.fields.get_task((path_ind+2000*i)%self.fields.n_episodes))
        observations = self.fields.normed_observations[path_inds, start:end]
        actions = self.fields.normed_actions[path_inds, start:end]
        task = np.array([self.task_list.index(self.fields.get_task(path_ind)) for path_ind in path_inds])#self.get_task_id(self.fields.get_task(path_ind))
        rtg = self.discount_cumsum(self.fields['rewards'][path_inds, :], gamma=self.discount)[:, :end - start] / 400.
        conditions = self.get_conditions_new(observations) #TODO
        trajectories = np.concatenate([rtg, actions, observations], axis=-1)
        batch = TaskBatch(trajectories, conditions, task, rtg[:, 0])
        return batch
class RTGActDataset(DMCSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=True, seq_length=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.seq_length = seq_length
        self.draw()
    def normalize(self, keys=['observations', 'actions']):#, 'observations', 'rewards'
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key == 'rewards':
                self.fields[f'normed_{key}'] = self.fields[key] #/ self.reward_scale
                continue
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def __len__(self):
        return len(self.indices)
    def draw(self):
        V = np.array(self.record_values)
        print(V.shape)
        normed_V = (V - V.min()) / (V.max() - V.min())
        #normed_V = normed_V * 2 - 1
        sns.set_palette("hls")
        mpl.rc("figure", figsize=(9, 5))
        fig = sns.distplot(normed_V,bins=20)
        fig.set_xlabel("Normalized Return", fontsize=16)
        fig.set_ylabel("Density", fontsize=16)
        displot_fig = fig.get_figure()
        displot_fig.savefig('./sub-optimal.pdf', dpi = 400)
    def discount_cumsum(self, x, gamma):
        x = x.squeeze(-1)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[:,-1] = x[:,-1]
        for t in reversed(range(x.shape[-1] - 1)):
            discount_cumsum[:,t] = x[:,t] + gamma * discount_cumsum[:,t + 1]
        return np.expand_dims(discount_cumsum, axis=-1)
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        path_inds = []
        interval = int(self.fields.n_episodes/len(self.task_list))
        for i in range(len(self.task_list)):
            path_inds.append((path_ind+interval*i)%self.fields.n_episodes)
        """
        if start == 0:
            tmp = np.expand_dims(self.fields.normed_observations[path_inds, start], axis=1)
            observations = np.concatenate([np.zeros_like(tmp), tmp], axis=1)
        else:
            observations = self.fields.normed_observations[path_inds, start-1:start+1]
        """
        observations = np.zeros((len(path_inds), self.seq_length, self.observation_dim))
        count = start
        k = self.seq_length - 1
        while count >= 0 and k >= 0:
            observations[:, k, :] = self.fields.normed_observations[path_inds, count]
            k -= 1
            count -= 1
        actions = self.fields.normed_actions[path_inds, start:end]
        task = np.array([self.task_list.index(self.fields.get_task(path_ind)) for path_ind in path_inds])#self.get_task_id(self.fields.get_task(path_ind))
        rtg = self.discount_cumsum(self.fields['rewards'][path_inds, start:], gamma=self.discount)[:,:end-start] / (self.max_path_length-start)
        #rtg = self.discount_cumsum(self.fields['rewards'][path_inds, :], gamma=self.discount)[:, :end - start] / 400.
        batch = TaskBatch(actions, observations, task, rtg[:, 0])
        return batch


class MazeDataset(DMCSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=True, seq_length=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.seq_length = seq_length
        if normed:
            self.vmin, self.vmax = self._get_bounds()

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.get_value(i)
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        # self.draw()
        return vmin, vmax

    def get_value(self, idx):
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind].sum()
        return rewards

    def normalize(self, keys=['observations', 'actions']):  # , 'observations', 'rewards'
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def __len__(self):
        return len(self.indices)

    def discount_cumsum(self, x, gamma):
        x = x.squeeze(-1)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[:, -1] = x[:, -1]
        for t in reversed(range(x.shape[-1] - 1)):
            discount_cumsum[:, t] = x[:, t] + gamma * discount_cumsum[:, t + 1]
        return np.expand_dims(discount_cumsum, axis=-1)

    def normalize_return(self, rewards):
        return (np.sum(rewards, axis=1) - self.vmin) / (self.vmax - self.vmin)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        path_inds = []
        interval = int(self.fields.n_episodes / len(self.task_list))
        for i in range(len(self.task_list)):
            path_inds.append((path_ind + interval * i) % self.fields.n_episodes)
        observations = np.zeros((len(path_inds), self.seq_length, self.observation_dim))
        count = start
        k = self.seq_length - 1
        while count >= 0 and k >= 0:
            observations[:, k, :] = self.fields.normed_observations[path_inds, count]
            k -= 1
            count -= 1
        actions = self.fields.normed_actions[path_inds, start:end]
        task = np.array([self.task_list.index(self.fields.get_task(path_ind)) for path_ind in
                         path_inds])  # self.get_task_id(self.fields.get_task(path_ind))
        # rtg = self.discount_cumsum(self.fields['rewards'][path_inds, start:], gamma=self.discount)[:,:end-start] / (self.max_path_length-start)
        rtg = self.normalize_return(self.fields['rewards'][path_inds])
        # rtg = self.discount_cumsum(self.fields['rewards'][path_inds, :], gamma=self.discount)[:, :end - start] / 400.
        batch = TaskBatch(actions, observations, task, rtg)
        return batch
class PromptDTMazeDataset(DMCSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=1.0, normed=True, seq_length=5, **kwargs):
        super().__init__(*args, **kwargs)
        dic = {
            'maze2d-1': utils.MAZE_1,
            'maze2d-2': utils.MAZE_2,
            'maze2d-3': utils.MAZE_3,
            'maze2d-4': utils.MAZE_4,
            'maze2d-5': utils.MAZE_5,
            'maze2d-6': utils.MAZE_6,
            'maze2d-7': utils.MAZE_7,
            'maze2d-8': utils.MAZE_8,
        }
        self.prompt_trajectories = [utils.parse_maze(dic[task_id]) for task_id in self.task_list]
        #self.prompt_trajectories = [np.load(f"./metaworld_prompts/{self.task_list[ind]}_prompt.npy", allow_pickle=True) for ind in
        #                       range(len(self.task_list))]
        self.discount = discount
    def normalize(self, keys=['observations', 'actions', 'rewards']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key == 'rewards':
                self.fields[f'normed_{key}'] = self.fields[key] / self.reward_scale
                continue
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def __len__(self):
        return len(self.indices)

    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        timestep = np.arange(start, end)
        prompt_batch = self.prompt_trajectories[self.task_list.index(self.fields.get_task(path_ind))]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        rtg = self.discount_cumsum(self.fields.normed_rewards[path_ind, start:], gamma=self.discount)[:end-start]
        #trajectories = np.concatenate([actions, rtg, observations], axis=-1)
        mask = np.ones(end-start)
        batch = DTBatch(actions, rtg, observations, timestep, mask)
        #print("Batch item load!")
        return batch, prompt_batch
class AugDataset(DMCSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=True, seq_length=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.seq_length = seq_length
    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            #if i % 10: continue
            max_start = min(path_length - 2, self.max_path_length - horizon - 1)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def normalize(self, keys=['observations', 'actions', 'rewards',]):#, 'observations', 'rewards'
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        path_inds = []
        interval = int(self.fields.n_episodes/len(self.task_list))
        for i in range(len(self.task_list)):
            path_inds.append((path_ind+interval*i)%self.fields.n_episodes)
        actions = self.fields.normed_actions[path_inds, start:end]
        observations = self.fields.normed_observations[path_inds, start:end]
        rewards = self.fields.normed_rewards[path_inds, start:end].reshape(len(self.task_list), end-start, 1)
        next_observations = self.fields.normed_observations[path_inds, start+1:end+1]
        trajectories = np.concatenate([observations, actions, rewards, next_observations], axis=-1)
        task = np.array([self.task_list.index(self.fields.get_task(path_ind)) for path_ind in path_inds])#self.get_task_id(self.fields.get_task(path_ind))
        #print(task.shape)
        batch = AugBatch(trajectories, task)
        return batch
class DTDataset(DMCSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
    def normalize(self, keys=['observations', 'actions', 'rewards']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key == 'rewards':
                self.fields[f'normed_{key}'] = self.fields[key] / self.reward_scale
                continue
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def __len__(self):
        return len(self.indices)

    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        timestep = np.arange(start, end)

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        task = np.eye(len(self.task_list))[self.task_list.index(self.fields.get_task(path_ind))] \
            .reshape(1, -1).repeat(end - start, axis=0)  # self.get_task_id(self.fields.get_task(path_ind))
        observations = np.concatenate([observations, task], axis=-1)
        rtg = self.discount_cumsum(self.fields.normed_rewards[path_ind, start:], gamma=self.discount)[:end - start]
        # trajectories = np.concatenate([actions, rtg, observations], axis=-1)
        mask = np.ones(end - start)
        batch = DTBatch(actions, rtg, observations, timestep, mask)
        # print("Batch item load!")
        return batch
class PromptDTDataset(DMCSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=1.0, normed=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_trajectories = [np.load(f"./metaworld_prompts/{self.task_list[ind]}_prompt.npy", allow_pickle=True) for ind in
                               range(len(self.task_list))]
        self.discount = discount
    def normalize(self, keys=['observations', 'actions', 'rewards']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key == 'rewards':
                self.fields[f'normed_{key}'] = self.fields[key] / self.reward_scale
                continue
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def __len__(self):
        return len(self.indices)

    def get_prompt(self, inds, num_episodes=1, num_steps=20, is_meta=True):
        prompt = self.prompt_trajectories[inds]
        trj_ids = np.random.choice(
            np.arange(len(prompt)),
            size=num_episodes,
            replace=False,
        )
        T = 220 if is_meta else 1000
        start_steps = np.random.choice(
            np.arange(0, T - 20),
            size=num_episodes,
            replace=True,
        )
        obs = np.array([prompt[trj_ids[i]]['observations'][start_steps[i]:start_steps[i] + num_steps]
                              for i in range(num_episodes)]).reshape(num_episodes * num_steps, -1)
        actions = np.array([prompt[trj_ids[i]]['actions'][start_steps[i]:start_steps[i] + num_steps]
                                  for i in range(num_episodes)]).reshape(num_episodes * num_steps, -1)
        rtg = np.array([self.discount_cumsum(prompt[trj_ids[i]]['rewards'][start_steps[i]:], gamma=self.discount)[:num_steps]
                                 for i in range(num_episodes)]).reshape(num_episodes * num_steps, -1)
        timestep = np.array([np.arange(start_steps[i], start_steps[i] + num_steps)
                             for i in range(num_episodes)]).reshape(num_episodes * num_steps)
        mask = np.ones(num_steps)
        return DTBatch(actions, rtg, obs, timestep, mask)

    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        timestep = np.arange(start, end)
        prompt_batch = self.get_prompt(self.task_list.index(self.fields.get_task(path_ind)))
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        rtg = self.discount_cumsum(self.fields.normed_rewards[path_ind, start:], gamma=self.discount)[:end-start]
        #trajectories = np.concatenate([actions, rtg, observations], axis=-1)
        mask = np.ones(end-start)
        batch = DTBatch(actions, rtg, observations, timestep, mask)
        #print("Batch item load!")
        return batch, prompt_batch
class MTValueDataset(DMCSequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_values = []
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = True
        self.reward_scale = 1000.0
        ##if normed:
        ##   self.vmin, self.vmax = self._get_bounds()
        ##   self.normed = True
        self.record_values = []
    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
    def normalize(self, keys=['observations', 'actions', 'rewards']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key == 'rewards':
                self.fields[f'normed_{key}'] = self.fields[key] / self.reward_scale
                continue
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.get_value(i)
            vmin = min(value, vmin)
            vmax = max(value, vmax)
            self.record_values.append(value)
        print('✓')
        #self.draw()
        return vmin, vmax
    def draw(self):
        V = np.array(self.record_values)
        print(V.shape)
        normed_V = (V - V.min()) / (V.max() - V.min())
        normed_V = normed_V * 2 - 1
        sns.set_palette("hls")
        mpl.rc("figure", figsize=(9, 5))
        fig = sns.distplot(normed_V,bins=20)
        displot_fig = fig.get_figure()
        displot_fig.savefig('./pic/metaworld-1.png', dpi = 400)
    def get_value(self, idx):
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        self.record_values.append(value)
        return value
    

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        #path_ind, start, end = self.indices[idx]
        #rewards = self.fields['rewards'][path_ind, start:]
        #rtg = self.discount_cumsum(self.fields.normed_rewards[path_ind, start:], gamma=1.).sum()
        ##discounts = self.discounts[:len(rewards)]
        ##value = (discounts * rewards).sum()
        ##if self.normed:
        ##    value = self.normalize_value(value)
        #with open('./output/quadruped_roll_fast_norm_return.txt', 'a') as f:
         #   f.writelines(str(value)+'\n')
        ##value = np.array([value], dtype=np.float32)
        value = np.array([rtg], dtype=np.float32)
        value_batch = MTValueBatch(*batch, value)
        return value_batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }


class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_values = []
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.get_value(i)
            vmin = min(value, vmin)
            vmax = max(value, vmax)
            self.record_values.append(value)
        print('✓')
        #self.draw()
        return vmin, vmax
    def draw(self):
        V = np.array(self.record_values)
        print(V.shape)
        normed_V = (V - V.min()) / (V.max() - V.min())
        normed_V = normed_V * 2 - 1
        sns.set_palette("hls")
        mpl.rc("figure", figsize=(9, 5))
        fig = sns.distplot(normed_V,bins=20)
        displot_fig = fig.get_figure()
        displot_fig.savefig('./pic/d4rl_hopper_norm_value.png', dpi = 400)
    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed
    def get_value(self, idx):
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        return value

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = MTValueBatch(*batch, np.array(-1), value)
        return value_batch
class LatentDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', replay_dir_list=[], task_list=[], horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=200000, termination_penalty=0, use_padding=True, seed=None, meta_world=False, maze2d=False, antmaze=False, optimal=True):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env
        self.replay_dir_list = replay_dir_list
        self.task_list = task_list
        self.reward_scale = 400.0
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.record_values = []
        itr = load_latent_dataaset(self.replay_dir_list, self.task_list)
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        """task_counter = {
            'antmaze-umaze-diverse-v1': 0,
            'antmaze-medium-diverse-v1': 0,
            'antmaze-large-diverse-v1': 0,
        }
        task_prompts = {
            'antmaze-umaze-diverse-v1': [],
            'antmaze-medium-diverse-v1': [],
            'antmaze-large-diverse-v1': [],
        }"""
        for i, episode in enumerate(itr):
            if len(episode['rewards']) > self.max_path_length:
                # episode = {k: episode[k][:max_path_length] for k in episode.keys()}
                continue
            self.record_values.append(episode['rewards'].sum())
            data_dir = f"./dataset/f{episode['task']}"
            if not os.path.exists(data_dir):
                os.makedirs(self.data_dir)
            np.savez_compressed(os.path.join(data_dir, f'{i%2000}.npz'), **episode)
            #if len(episode['rewards']) < 1200 and task_counter[episode['task']] < 10:
            #    task_prompts[episode['task']].append(episode)
            #    task_counter[episode['task']] += 1
            """
            if i%4000>3989:
                prompt_data.append(episode)
            if i%4000==3999 and i!=0:
                np.save(os.path.join(f"./metaworld_prompts/{episode['task']}_prompt.npy"), np.array(prompt_data))
                prompt_data = []
            print(f"episode:{i}, task:{episode['task']}, Return:{episode['rewards'].sum()}")
            """
            #for key in task_prompts.keys():
            #    np.save(os.path.join(f"./antmaze_prompts/{key}_prompt.npy"), np.array(task_prompts[key]))
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions', 'rewards']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            #if i % 10: continue
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}
    def get_task_id(self,str):
        task_dict = {
            'quadruped_walk': np.array(0),
            'quadruped_jump': np.array(1),
            'quadruped_run': np.array(2),
            'quadruped_roll_fast': np.array(3)
            }
        return task_dict[str]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        rewards = self.fields.normed_rewards[path_ind, start:end]
        #observations = self.fields.observations[path_ind, start:end]
        #actions = self.fields.actions[path_ind, start:end]
        #rewards = self.fields.rewards[path_ind, start:end]
        task = np.array(self.task_list.index(self.fields.get_task(path_ind))).reshape(-1,1)#self.get_task_id(self.fields.get_task(path_ind))
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([rewards, actions, observations], axis=-1)
        #if np.any(np.isnan(trajectories)):
        #    print("True->>>")
        batch = TaskBatch(trajectories, conditions, task, 1)
        #print("Batch item load!")
        return batch
