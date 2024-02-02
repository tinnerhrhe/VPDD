import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, init_w=1e-3):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),  # 新增
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())

        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        f = self.policy(obs)
        mu = self.fc_mu(f)
        log_std = self.fc_logstd(f)
        mu = torch.clamp(mu, -5, 5)

        std = log_std.clamp(-5, 2).exp()
        dist = utils.SquashedNormal2(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, init_w=1e-3):
        super().__init__()

        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.q1_last = nn.Linear(hidden_dim, 1)

        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.q2_last = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q1 = self.q1_last(q1)

        q2 = self.q2_net(obs_action)
        q2 = self.q2_last(q2)

        return q1, q2


class CQLAgent:
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 device,
                 actor_lr,
                 critic_lr,
                 hidden_dim,
                 critic_target_tau,
                 nstep,
                 batch_size,
                 use_tb,
                 alpha,
                 n_samples,
                 target_cql_penalty,
                 use_critic_lagrange,
                 num_expl_steps,
                 has_next_action=False):
        self.num_expl_steps = num_expl_steps
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.use_critic_lagrange = use_critic_lagrange
        self.target_cql_penalty = target_cql_penalty

        self.alpha = alpha
        self.n_samples = n_samples

        state_dim = obs_shape[0]
        action_dim = action_shape[0]

        # models
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # lagrange multipliers
        self.target_entropy = -self.action_dim
        self.log_actor_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.log_critic_alpha = torch.zeros(1, requires_grad=True, device=device)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_alpha_opt = torch.optim.Adam([self.log_actor_alpha], lr=actor_lr)
        self.critic_alpha_opt = torch.optim.Adam([self.log_critic_alpha], lr=actor_lr)

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        policy = self.actor(obs)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def _repeated_critic_apply(self, obs, actions):
        """
			obs is (batch_size, obs_dim)
			actions is (n_samples, batch_size, action_dim)

			output tensors are (n_samples, batch_size, 1)
		"""
        batch_size = obs.shape[0]
        n_samples = actions.shape[0]

        reshaped_actions = actions.reshape((n_samples * batch_size, -1))
        repeated_obs = obs.unsqueeze(0).repeat((n_samples, 1, 1))
        repeated_obs = repeated_obs.reshape((n_samples * batch_size, -1))

        Q1, Q2 = self.critic(repeated_obs, reshaped_actions)
        Q1 = Q1.reshape((n_samples, batch_size, 1))
        Q2 = Q2.reshape((n_samples, batch_size, 1))

        return Q1, Q2

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # Compute standard SAC loss
        with torch.no_grad():
            dist = self.actor(next_obs)  # SquashedNormal distribution
            sampled_next_action = dist.sample()  # (1024, act_dim)
            # print("sampled_next_action:", sampled_next_action.shape)
            target_Q1, target_Q2 = self.critic_target(next_obs, sampled_next_action)  # (1024,1), (1024,1)
            target_V = torch.min(target_Q1, target_Q2)  # (1024,1)
            target_Q = reward + (discount * target_V)  # (1024,1)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)  # scalar

        # Add CQL penalty
        with torch.no_grad():
            random_actions = torch.FloatTensor(self.n_samples, Q1.shape[0],
                                               action.shape[-1]).uniform_(-1, 1).to(
                self.device)  # (n_samples, 1024, act_dim)
            sampled_actions = self.actor(obs).sample(
                sample_shape=(self.n_samples,))  # (n_samples, 1024, act_dim)
            # print("sampled_actions:", sampled_actions.shape)
            next_sampled_actions = self.actor(next_obs).sample(
                sample_shape=(self.n_samples,))  # (n_samples, 1024, act_dim)
        # print("next_sampled_actions:", next_sampled_actions.shape)

        rand_Q1, rand_Q2 = self._repeated_critic_apply(obs, random_actions)  # (n_samples, 1024, 1)
        sampled_Q1, sampled_Q2 = self._repeated_critic_apply(  # (n_samples, 1024, 1)
            obs, sampled_actions)
        next_sampled_Q1, next_sampled_Q2 = self._repeated_critic_apply(  # (n_samples, 1024, 1)
            obs, next_sampled_actions)

        # situation 1
        cat_Q1 = torch.cat([rand_Q1, sampled_Q1, next_sampled_Q1,
                            Q1.unsqueeze(0)], dim=0)  # (1+3*n_samples, 1024, 1)
        cat_Q2 = torch.cat([rand_Q2, sampled_Q2, next_sampled_Q2,
                            Q2.unsqueeze(0)], dim=0)  # (1+3*n_samples, 1024, 1)

        # situation 2
        # cat_Q1 = torch.cat([rand_Q1 - np.log(0.5 ** self.action_dim),
        # 	sampled_Q1 - self.actor(obs).log_prob(sampled_actions).unsqueeze(-1).detach(),
        # 	next_sampled_Q1 - self.actor(obs).log_prob(next_sampled_actions).unsqueeze(-1).detach()], dim=0)         # (1+3*n_samples, 1024, 1)
        # cat_Q2 = torch.cat([rand_Q2 - np.log(0.5 ** self.action_dim),
        # 	sampled_Q2 - self.actor(obs).log_prob(sampled_actions).unsqueeze(-1).detach(),
        # 	next_sampled_Q2 - self.actor(obs).log_prob(next_sampled_actions).unsqueeze(-1).detach()], dim=0)         # (1+3*n_samples, 1024, 1)

        # cat_Q1 = torch.cat([rand_Q1 - np.log(0.5 ** self.action_dim),
        # 	sampled_Q1 - self.actor(obs).log_prob(sampled_actions).sum(-1, keepdims=True).detach(),
        # 	next_sampled_Q1 - self.actor(obs).log_prob(next_sampled_actions).sum(-1, keepdims=True).detach()], dim=0)         # (1+3*n_samples, 1024, 1)
        # cat_Q2 = torch.cat([rand_Q2 - np.log(0.5 ** self.action_dim),
        # 	sampled_Q2 - self.actor(obs).log_prob(sampled_actions).sum(-1, keepdims=True).detach(),
        # 	next_sampled_Q2 - self.actor(obs).log_prob(next_sampled_actions).sum(-1, keepdims=True).detach()], dim=0)         # (1+3*n_samples, 1024, 1)

        # print("cat Q1:", cat_Q1.shape, cat_Q2.shape)
        # if torch.isnan(cat_Q1).any():
        # 	print("in cql 1:", torch.isnan(rand_Q1).any(), torch.isnan(sampled_Q1).any(), torch.isnan(rand_Q1).any(), torch.isnan(next_sampled_Q1).any())
        # 	print("in cql 2:", torch.isnan(self.actor(obs).log_prob(sampled_actions)).any())
        # 	print("in cql 3:", torch.isnan(self.actor(obs).log_prob(next_sampled_actions)).any())

        assert (not torch.isnan(cat_Q1).any()) and (not torch.isnan(cat_Q2).any())

        cql_logsumexp1 = torch.logsumexp(cat_Q1, dim=0).mean()
        cql_logsumexp2 = torch.logsumexp(cat_Q2, dim=0).mean()
        cql_logsumexp = cql_logsumexp1 + cql_logsumexp2
        # print("train 1:", rand_Q1.mean(), np.log(0.5 ** self.action_dim))
        # dist_test = self.actor(obs)
        # print("train 2:", dist_test.loc.shape, dist_test.scale.shape, torch.isnan(dist_test.loc).any(), torch.isnan(dist_test.scale).any())
        # print("train 3:", sampled_Q1.mean(), self.actor(obs).log_prob(sampled_actions).detach().mean())
        # print("train 4:", next_sampled_Q1.mean(), self.actor(obs).log_prob(next_sampled_actions).detach().mean())

        cql_penalty = cql_logsumexp - (Q1 + Q2).mean()  # 标量

        # Update lagrange multiplier
        if self.use_critic_lagrange:
            alpha = torch.clamp(self.log_critic_alpha.exp(), min=0.0, max=1000000.0)
            alpha_loss = -0.5 * alpha * (cql_penalty - self.target_cql_penalty)

            self.critic_alpha_opt.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.critic_alpha_opt.step()
            alpha = torch.clamp(self.log_critic_alpha.exp(),
                                min=0.0,
                                max=1000000.0).detach()
        else:
            alpha = self.alpha

        # Combine losses
        critic_loss = critic_loss + alpha * cql_penalty

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 50)
        critic_grad_norm = utils.grad_norm(self.critic.parameters())
        self.critic_opt.step()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            # metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['critic_cql'] = cql_penalty.item()
            metrics['critic_cql_logsum1'] = cql_logsumexp1.item()
            # metrics['critic_cql_logsum2'] = cql_logsumexp1.item()
            metrics['rand_Q1'] = rand_Q1.mean().item()
            # metrics['rand_Q2'] = rand_Q2.mean().item()
            metrics['sampled_Q1'] = sampled_Q1.mean().item()
            # metrics['sampled_Q2'] = sampled_Q2.mean().item()
            metrics['critic_grad_norm'] = critic_grad_norm

        return metrics

    def update_actor(self, obs, action, step):
        metrics = dict()

        policy = self.actor(obs)
        sampled_action = policy.rsample()  # (1024, 6)
        # print("sampled_action:", sampled_action.shape)
        log_pi = policy.log_prob(sampled_action)  # (1024, 6)

        # update lagrange multiplier
        alpha_loss = -(self.log_actor_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.actor_alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.actor_alpha_opt.step()
        alpha = self.log_actor_alpha.exp().detach()

        # optimize actor
        Q1, Q2 = self.critic(obs, sampled_action)  # (1024, 1)
        Q = torch.min(Q1, Q2)  # (1024, 1)
        actor_loss = (alpha * log_pi - Q).mean()  # 标量
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        actor_grad_norm = utils.grad_norm(self.actor.parameters())
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = -log_pi.mean().item()
            metrics['actor_alpha'] = alpha.item()
            metrics['actor_alpha_loss'] = alpha_loss.item()
            metrics['actor_mean'] = policy.loc.mean().item()
            metrics['actor_std'] = policy.scale.mean().item()
            metrics['actor_action'] = sampled_action.mean().item()
            metrics['actor_atanh_action'] = utils.atanh(sampled_action).mean().item()
            metrics['actor_grad_norm'] = actor_grad_norm

        return metrics

    def update(self, replay_iter, step, total_step):
        metrics = dict()

        batch = next(replay_iter)
        # obs.shape=(1024,obs_dim), action.shape=(1024,1), reward.shape=(1024,1), discount.shape=(1024,1)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(
            batch, self.device)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs, action, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
