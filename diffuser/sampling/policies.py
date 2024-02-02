from collections import namedtuple
import torch
import einops
import pdb
import numpy as np
import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', ' rewards actions observations values')
NewTrajectories = namedtuple('NewTrajectories', ' actions observations values')
class CondPolicy:

    def __init__(self, task, value, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.task= utils.to_torch(task, dtype=torch.float32, device='cuda:0')
        self.value = utils.to_torch(value, dtype=torch.float32, device='cuda:0')
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs
        print("normalizer: ",self.normalizer)

    def __call__(self, conditions, batch_size=1, verbose=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        #print(self.task.shape,self.value.shape)
        samples = self.diffusion_model(conditions, task=self.task, value=self.value, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)
        #print(trajectories.shape)
        ##rewards = trajectories[:, :, :1]
        #print("normalize rewards:", rewards)
        ##rewards = self.normalizer.unnormalize(rewards, 'rewards')
        #print("unnormalized rewards:",rewards)
        ##actions = trajectories[:, :, 1:self.action_dim+1]
        actions = trajectories[:, :, :self.action_dim]
        #print(actions.shape)
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        ##normed_observations = trajectories[:, :, 1 + self.action_dim:]
        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = NewTrajectories(actions, observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

class GuidedPolicy:

    def __init__(self, task, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.task=task
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, task=self.task, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)
        #print(trajectories.shape)
        rewards = trajectories[:, :, :1]
        rewards = self.normalizer.unnormalize(rewards, 'rewards')
        actions = trajectories[:, :, 1:self.action_dim+1]
        #print(actions.shape)
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, 1 + self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(rewards, actions, observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
class Policy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        rewards = trajectories[:, :, :1]
        rewards = self.normalizer.unnormalize(rewards, 'rewards')
        actions = trajectories[:, :, 1:self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, 1+self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(rewards, actions, observations, samples.values)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
