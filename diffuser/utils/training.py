import os
import copy
import numpy as np
import torch
import einops
import pdb
import random
from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
#import dmc
import time
import gym
import einops
import math
import torch.nn.functional as F
#import d4rl
import statistics
DTYPE = torch.float
from collections import namedtuple
import diffuser.utils as utils
DTBatch = namedtuple('DTBatch', 'actions rtg observations timestep mask')
DEVICE = 'cuda'
from enum import Enum
from helpers.clip.core.clip import build_model, load_clip, tokenize
from metaworld.envs.mujoco.env_dict import MT50_V2, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from torch.utils.tensorboard import SummaryWriter
def cycle(dl):
    while True:
        for data in dl:
            yield data

def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
        # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        envs=[],
        task_list=[],
        is_unet=False,
        trainer_device=None,
        horizon=32,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = horizon
        self.task_list = task_list
        self.is_unet=is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.reset_parameters()
        self.step = 0
        self.inverse_model = ARInvModel(hidden_dim=256, action_dim=self.dataset.action_dim,
                                        observation_dim=self.dataset.observation_dim).to(self.device)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        #l1, l2 = [], []
        best_score, prev_label = 0, 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                #print("START!")
                batch = next(self.dataloader)
                #print("BATCH LOAD!")
                batch = batch_to_device(batch,self.device)
                #print("training step:", self.step, "gradient:", i)
                loss, infos = self.model.loss(*batch)
                #l1.append(loss)
                #for key, val in infos.items():
                #    l2.append(val)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                '''
                for name, parms in self.model.named_parameters(): 
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, 
                          ' -->grad_value:',parms.grad)
                '''
            #print("training step:", self.step)
            self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #self.writer.add_scalar('Inv_Loss', infos['inv_loss'], global_step=self.step)
            self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
                #score, success_rate = self.state_evaluate_dmc(self.device)
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                if self.step != 0:
                    if self.is_unet:
                        score, success_rate = self.evaluate_dmc(self.device)
                    else:
                        score, success_rate = self.evaluate_dmc(self.device)#self.evaluate_dmc_dt(self.device)
                    label = str(label) + '_' + str(score) + '_' + str(success_rate)
                    if score > best_score:
                        self.save(label)
                        best_score = score
                else:
                    #score, success_rate = self.evaluate_dmc(self.device)#self.evaluate_dmc_dt(self.device)
                    self.save(label)
                #self.save(label)


            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
                #l1,l2=torch.tensor(l1,device=loss.device),torch.tensor(l2,device=loss.device)
                #print(f'max:{l1.max()},Triquarter:{l1.quantile(0.75)},median:{l1.median()},mean:{l1.mean()},quarter:{l1.quantile(0.25)},min:{l1.min()}')
                #print(f'max:{l2.max()},Triquarter:{l2.quantile(0.75)},median:{l2.median()},mean:{l2.mean()},quarter:{l2.quantile(0.25)},min:{l2.min()}')
                #l1=[]
                #l2=[]
            '''
            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()
            '''
            self.step += 1
    def state_evaluate_dmc_unet(self,device):
        #inverse_path = "./logs/inverse-dynamic/-Feb25_05-08-54/state_19000000.pt"
        #self.inverse_model.load_state_dict(torch.load(inverse_path)['model'])
        #self.task_list = ['walker_run','walker_walk','walker_flip','walker_stand']
        #env_list = [dmc2gym.make(domain_name='walker', task_name=task, seed=0) for task in self.task_list]
        env_list = [dmc.make(task, seed=0) for task in self.task_list]
        num_eval = 1
        score = 0
        for i in range(len(env_list)):
            dones = 0
            max_episode_length = 1000
            eval_return = []
            while dones < num_eval:
                timestep = env_list[i].reset()
                observation = timestep.observation
                total_reward = 0
                reward = 0
                rtg = torch.ones((1,), device=device) * 950
                condition_step = 0
                conditions = {}
                condition_value = 950.0
                for j in range(max_episode_length):
                    obs = self.dataset.normalizer.normalize(observation, 'observations')
                    rtg = rtg - reward
                    """
                    if condition_step >= 20:
                        for key in sorted(conditions):
                            if key == 19:
                                conditions[key] = to_torch(obs, device=device).unsqueeze(0)
                            else:
                                conditions[key] = conditions[key+1]
                    #conditions = {0: to_torch(obs, device=device)}
                    else:
                        conditions[condition_step] = to_torch(obs, device=device).unsqueeze(0)
                    """
                    conditions = {0: torch.cat((rtg / 100.0, to_torch(obs, device=device)), dim=-1).unsqueeze(0)}
                    condition_value = condition_value - reward
                    samples = self.ema_model.conditional_sample(conditions, task=torch.tensor(i, device=device),
                                                                value=torch.tensor([condition_value], device=device, dtype=conditions[0].dtype),
                                                                verbose=False, horizon=self.horizon, guidance=1.2)
                    #obs_comb = torch.cat([samples.trajectories[0, 0, :], samples.trajectories[0, 1, :]], dim=-1)
                    obs_comb = torch.cat([samples.trajectories[0, 0, 1:],
                                          samples.trajectories[0, 1, 1:]], dim=-1)
                    obs_comb = obs_comb.reshape(-1, 2 * self.dataset.observation_dim)
                    #action = self.inverse_model.forward0(torch.tensor(obs_comb, device=self.device), torch.tensor([i], device=device))
                    action = self.ema_model.inv_model.forward0(torch.tensor(obs_comb, device=self.device),
                                                         torch.tensor([i], device=device))
                    #print(action.shape)
                    action = action.reshape(self.dataset.action_dim)
                    #print(action.shape)
                    action = to_np(action)
                    action = self.dataset.normalizer.unnormalize(action, 'actions')
                    #next_observation, reward, terminal, info = env_list[i].step(action)
                    timestep = env_list[i].step(action)
                    next_observation, reward = timestep.observation, timestep.reward
                    total_reward += reward
                    observation = next_observation
                    #condition_step += 1
                dones += 1
                eval_return.append(total_reward)
            env_score = sum(eval_return) / len(eval_return)
            #env_success_rate.append(success / num_eval)
            print(f"task:{self.task_list[i]},episodic return:{env_score}")
            score += env_score
        return score, 1
    def evaluate_dmc_dt(self,device):
        #inverse_path = "./logs/inverse-dynamic/-Feb25_05-08-54/state_19000000.pt"
        #self.inverse_model.load_state_dict(torch.load(f'quadruped_inv_model.pt', map_location=device)['model'], strict=True)
        #self.task_list = ['walker_run','walker_walk','walker_flip','walker_stand']
        #env_list = [dmc2gym.make(domain_name='walker', task_name=task, seed=0) for task in self.task_list]
        num_eval = 5
        env_list = [dmc.make(task, seed=i) for i in range(num_eval) for task in self.task_list]
        score = 0
        dones = 0
        # max_episode_length = 150
        max_episode_length = 1000
        eval_return = []
        episode_rewards = [0 for i in env_list]
        obs_list = [env.reset().observation[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.task_list))], device=device).reshape(-1,)
        while dones < 1:
            conditions = {}
            for k in range(10):
                conditions[k] = torch.zeros((obs.shape[0], obs.shape[1]+self.dataset.action_dim), device=device)
                #conditions[k] = torch.zeros(obs.shape, device=device)
            # condition_step = 0
            rtg = torch.ones((len(env_list),), device=device) * 0.9
            total_reward = 0
            # reward = 0
            for j in range(max_episode_length):
                obs = self.dataset.normalizer.normalize(obs, 'observations')
                # rtg = rtg - reward
                # if condition_step >= 10:
                for key in sorted(conditions):
                    if key == 9:
                        conditions[key] = to_torch(obs, device=device)
                    else:
                        conditions[key] = conditions[key + 1]
                samples = self.ema_model.conditional_sample_v1(conditions, task=cond_task,
                                                            value=rtg,
                                                            verbose=False, horizon=self.horizon, guidance=1.6)
                action =samples.trajectories[:, 9, :self.dataset.action_dim]
                conditions[9] = torch.cat((action, conditions[9]), dim=-1).to(device=device)
                # print(action.shape)
                action = to_np(action)
                action = self.dataset.normalizer.unnormalize(action, 'actions')
                # next_observation, reward, terminal, info = env_list[i].step(action)
                obs_list = []
                for i in range(len(env_list)):
                    timestep = env_list[i].step(action[i])
                    next_observation, reward = timestep.observation, timestep.reward
                    obs_list.append(next_observation[None])
                    episode_rewards[i] += reward
                obs = np.concatenate(obs_list, axis=0)
            dones += 1

        for i in range(len(self.task_list)):
            tmp = []
            for j in range(num_eval):
                tmp.append(episode_rewards[i+j*4])
            this_score = statistics.mean(tmp)
            score += this_score
            print(f"task:{self.task_list[i]},mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        return score, 1
    def state_evaluate_dmc_v1(self,device):
        #inverse_path = "./logs/inverse-dynamic/-Feb25_05-08-54/state_19000000.pt"
        self.inverse_model.load_state_dict(torch.load(f'quadruped_inv_model_v5.pt', map_location=device)['model'], strict=True)
        #self.task_list = ['walker_run','walker_walk','walker_flip','walker_stand']
        #env_list = [dmc2gym.make(domain_name='walker', task_name=task, seed=0) for task in self.task_list]
        num_eval = 5
        env_list = [dmc.make(task, seed=i) for i in range(num_eval) for task in self.task_list]
        score = 0
        dones = 0
        # max_episode_length = 150
        max_episode_length = 1000
        eval_return = []
        episode_rewards = [0 for i in env_list]
        obs_list = [env.reset().observation[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.task_list))], device=device).reshape(-1,)
        while dones < 1:
            conditions = {}
            for k in range(10):
                #conditions[k] = torch.zeros((obs.shape[0], obs.shape[1]+self.dataset.action_dim), device=device)
                conditions[k] = torch.zeros(obs.shape, device=device)
            # condition_step = 0
            rtg = torch.ones((len(env_list),), device=device) * 0.95
            total_reward = 0
            # reward = 0
            for j in range(max_episode_length):
                obs = self.dataset.normalizer.normalize(obs, 'observations')
                # rtg = rtg - reward
                # if condition_step >= 10:
                for key in sorted(conditions):
                    if key == 9:
                        conditions[key] = to_torch(obs, device=device)
                    else:
                        conditions[key] = conditions[key + 1]
                samples = self.ema_model.conditional_sample_v1(conditions, task=cond_task,
                                                            value=rtg,
                                                            verbose=False, horizon=self.horizon, guidance=1.2)
                # obs_comb = torch.cat([samples.trajectories[0, 0, :], samples.trajectories[0, 1, :]], dim=-1)
                obs_comb = torch.cat([samples.trajectories[:, 9, :],
                                      samples.trajectories[:, 10, :]], dim=-1)
                obs_comb = obs_comb.reshape(-1, 2 * self.dataset.observation_dim)
                action = self.inverse_model.forward0(obs_comb.clone().to(self.device), cond_task.unsqueeze(1))
                # action = self.ema_model.inv_model.forward0(obs_comb.clone().to(self.device),
                #                                     torch.tensor([i], device=device))
                # print(action.shape)
                action = action.reshape(-1, self.dataset.action_dim)
                #conditions[9] = torch.cat((action, conditions[9]), dim=-1).to(device=device)
                # print(action.shape)
                action = to_np(action)
                action = self.dataset.normalizer.unnormalize(action, 'actions')
                # next_observation, reward, terminal, info = env_list[i].step(action)
                obs_list = []
                for i in range(len(env_list)):
                    timestep = env_list[i].step(action[i])
                    next_observation, reward = timestep.observation, timestep.reward
                    obs_list.append(next_observation[None])
                    episode_rewards[i] += reward
                obs = np.concatenate(obs_list, axis=0)
            dones += 1

        for i in range(len(self.task_list)):
            tmp = []
            for j in range(num_eval):
                tmp.append(episode_rewards[i+j*4])
            this_score = statistics.mean(tmp)
            score += this_score
            print(f"task:{self.task_list[i]},mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        return score, 1
    def state_evaluate_dmc(self,device):
        #inverse_path = "./logs/inverse-dynamic/-Feb25_05-08-54/state_19000000.pt"
        #self.inverse_model.load_state_dict(torch.load(inverse_path)['model'])
        #self.task_list = ['walker_run','walker_walk','walker_flip','walker_stand']
        #env_list = [dmc2gym.make(domain_name='walker', task_name=task, seed=0) for task in self.task_list]
        env_list = [dmc.make(task, seed=0) for task in self.task_list]
        num_eval = 1
        score = 0
        for i in range(len(env_list)):
            dones = 0
            #max_episode_length = 150
            max_episode_length = 1000
            eval_return = []
            while dones < num_eval:
                #observation = env_list[i].reset()
                timestep = env_list[i].reset()
                observation = timestep.observation
                rtg = torch.ones((1,), device=device) * 8.5
                total_reward = 0
                reward = 0
                for j in range(max_episode_length):
                    obs = self.dataset.normalizer.normalize(observation, 'observations')
                    #rtg = rtg - reward
                    """
                    if condition_step >= 20:
                        for key in sorted(conditions):
                            if key == 19:
                                conditions[key] = torch.cat((rtg, to_torch(obs, device=device)), dim=-1).unsqueeze(0)
                            else:
                                conditions[key] = conditions[key+1]
                    #conditions = {0: to_torch(obs, device=device)}
                    else:
                        conditions[condition_step] = torch.cat((rtg, to_torch(obs, device=device)), dim=-1).unsqueeze(0)
                    """
                    conditions = {0: torch.cat((rtg, to_torch(obs, device=device)), dim=-1).unsqueeze(0)}
                    samples = self.ema_model.conditional_sample(conditions, task=torch.tensor(i, device=device),
                                                                value=torch.tensor([0.9], device=device),
                                                                verbose=False, horizon=self.horizon, guidance=1.6)
                    #obs_comb = torch.cat([samples.trajectories[0, 0, :], samples.trajectories[0, 1, :]], dim=-1)
                    obs_comb = torch.cat([samples.trajectories[0, 0, 1:],
                                          samples.trajectories[0, 1, 1:]], dim=-1)
                    obs_comb = obs_comb.reshape(-1, 2 * self.dataset.observation_dim)
                    #action = self.inverse_model.forward0(torch.tensor(obs_comb, device=self.device), torch.tensor([i], device=device))
                    action = self.ema_model.inv_model.forward0(torch.tensor(obs_comb, device=self.device),
                                                         torch.tensor([i], device=device))
                    #print(action.shape)
                    action = action.reshape(self.dataset.action_dim)
                    #print(action.shape)
                    action = to_np(action)
                    action = self.dataset.normalizer.unnormalize(action, 'actions')
                    #next_observation, reward, terminal, info = env_list[i].step(action)
                    timestep = env_list[i].step(action)
                    next_observation, reward = timestep.observation, timestep.reward
                    total_reward += reward
                    observation = next_observation
                    #condition_step += 1
                dones += 1
                eval_return.append(total_reward)
            env_score = sum(eval_return) / len(eval_return)
            #env_success_rate.append(success / num_eval)
            print(f"task:{self.task_list[i]},episodic return:{env_score}")
            score += env_score
        return score, 1
    def state_evaluate(self,device):
        #inverse_path = "./logs/inverse-dynamic/-Feb25_05-08-54/state_19000000.pt"
        #self.inverse_model.load_state_dict(torch.load(inverse_path)['model'])
        task = [metaworld.MT1(env).train_tasks[0] for env in self.envs]
        mt1 = [metaworld.MT1(env) for env in self.envs]
        env_list = [mt1[i].train_classes[self.envs[i]]() for i in range(len(self.envs))]
        #env_list = [dmc.make()]
        num_eval = 2
        score = 0
        env_success_rate = []
        for i in range(len(self.env_list)):
            seed = random.randint(0, 100000)
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)
            success = 0.0
            dones = 0
            #max_episode_length = 150
            max_episode_length = 1000
            eval_return = []
            while dones < num_eval:
                observation = env_list[i].reset()
                rtg = torch.ones((observation.shape[0],1),device=device) * 900
                total_reward = 0
                reward = 0
                for j in range(max_episode_length):
                    obs = self.dataset.normalizer.normalize(observation, 'observations')
                    rtg = rtg - reward
                    #conditions = {0: to_torch(obs, device=device)}
                    conditions = {0: torch.cat((rtg, to_torch(obs, device=device)), dim=-1)}
                    samples = self.ema_model.conditional_sample(conditions, task=torch.tensor(i, device=device),
                                                                value=torch.tensor([0.9], device=device),
                                                                verbose=False, horizon=32, guidance=1.6)
                    #obs_comb = torch.cat([samples.trajectories[0, 0, :], samples.trajectories[0, 1, :]], dim=-1)
                    obs_comb = torch.cat([samples.trajectories[0, 0, self.dataset.action_dim:],
                                          samples.trajectories[0, 1, self.dataset.action_dim:]], dim=-1)
                    obs_comb = obs_comb.reshape(-1, 2 * self.dataset.observation_dim)
                    #action = self.inverse_model.forward0(torch.tensor(obs_comb, device=self.device), torch.tensor([i], device=device))
                    action = self.ema_model.inv_model.forward0(torch.tensor(obs_comb, device=self.device),
                                                         torch.tensor([i], device=device))
                    #print(action.shape)
                    action = action.reshape(self.dataset.action_dim)
                    #print(action.shape)
                    action = to_np(action)
                    action = self.dataset.normalizer.unnormalize(action, 'actions')
                    next_observation, reward, terminal, info = env_list[i].step(action)
                    total_reward += reward
                    # print(f'reward:{reward}')
                    if info['success'] > 1e-8:
                        success += 1
                        break
                    observation = next_observation
                dones += 1
                eval_return.append(total_reward)
            env_score = sum(eval_return) / len(eval_return)
            env_success_rate.append(success / num_eval)
            print(f"task:{self.envs[i]},episodic return:{env_score},success rate:{success / num_eval}")
            score += env_score
        return score, sum(env_success_rate) / len(env_success_rate)

    def evaluate_dmc(self, device):
        num_eval = 5
        env_list = [dmc.make(task, seed=i) for i in range(num_eval) for task in self.task_list]
        score = 0
        dones = 0
        # max_episode_length = 150
        max_episode_length = 1000
        eval_return = []
        episode_rewards = [0 for i in env_list]
        obs_list = [env.reset().observation[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.task_list))],
                                 device=device).reshape(-1, )
        while dones < 1:
            conditions = torch.zeros([obs.shape[0], 5, obs.shape[-1]], device=device)
            rtg = torch.ones((len(env_list),), device=device) * 0.95
            # reward = 0
            for j in range(max_episode_length):
                obs = self.dataset.normalizer.normalize(obs, 'observations')
                conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=device).unsqueeze(1)], dim=1)
                samples = self.ema_model.conditional_sample(conditions, task=cond_task,
                                                               value=rtg,
                                                               verbose=False, horizon=self.horizon, guidance=1.2)
                action = samples.trajectories[:, 0, :]
                action = action.reshape(-1, self.dataset.action_dim)
                action = to_np(action)
                action = self.dataset.normalizer.unnormalize(action, 'actions')
                obs_list = []
                for i in range(len(env_list)):
                    timestep = env_list[i].step(action[i])
                    next_observation, reward = timestep.observation, timestep.reward
                    obs_list.append(next_observation[None])
                    episode_rewards[i] += reward
                obs = np.concatenate(obs_list, axis=0)
            dones += 1

        for i in range(len(self.task_list)):
            tmp = []
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * 4])
            this_score = statistics.mean(tmp)
            score += this_score
            print(f"task:{self.task_list[i]},mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        return score, 1
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    #print(video.shape)
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    #video = video.float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    video -= 0.5

    return video
class DTTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        envs=[],
        task_list=[],
        is_unet=False,
        trainer_device=None,
        horizon=32,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = horizon
        self.task_list = task_list
        self.is_unet=is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.reset_parameters()
        self.step = 0
        # self.inverse_model = ARInvModel(hidden_dim=256, action_dim=self.dataset.action_dim,
        #                                 observation_dim=self.dataset.observation_dim, act='relu').to(self.device)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        #l1, l2 = [], []
        best_score, prev_label = 0, 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                #print("START!")
                batch = next(self.dataloader)
                #print("BATCH LOAD!")
                batch = batch_to_device(batch,self.device)
                #print("training step:", self.step, "gradient:", i)
                loss, infos = self.model.loss(*batch)
                #l1.append(loss)
                #for key, val in infos.items():
                #    l2.append(val)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                '''
                for name, parms in self.model.named_parameters(): 
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, 
                          ' -->grad_value:',parms.grad)
                '''
            #print("training step:", self.step)
            self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #self.writer.add_scalar('Inv_Loss', infos['inv_loss'], global_step=self.step)
            self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
                #score, success_rate = self.state_evaluate_dmc(self.device)
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                if self.step != 0:
                    if self.is_unet:
                        score, success_rate = self.evaluate(self.device)
                    else:
                        score, success_rate = self.evaluate(self.device)#self.evaluate_dmc_dt(self.device)
                    label = str(label) + '_' + str(score) + '_' + str(success_rate)
                    if score > best_score:
                        self.save(label)
                        best_score = score
                else:
                    #score, success_rate = self.evaluate(self.device)#self.evaluate_dmc_dt(self.device)
                    self.save(label)
                #self.save(label)


            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
                #l1,l2=torch.tensor(l1,device=loss.device),torch.tensor(l2,device=loss.device)
                #print(f'max:{l1.max()},Triquarter:{l1.quantile(0.75)},median:{l1.median()},mean:{l1.mean()},quarter:{l1.quantile(0.25)},min:{l1.min()}')
                #print(f'max:{l2.max()},Triquarter:{l2.quantile(0.75)},median:{l2.median()},mean:{l2.mean()},quarter:{l2.quantile(0.25)},min:{l2.min()}')
                #l1=[]
                #l2=[]
            '''
            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()
            '''
            self.step += 1
    def create_env_and_policy(self, env_name, seed=None):
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']
        env = env_cls(seed=seed, render_mode='rgb_array')
        env.camera_name="corner2"
        return env
    def evaluate(self, device):
        from tqdm import tqdm
        description = "Dunk the basketball into the basket"
        max_episode_length = 200
        num_evals = 20
        env_ = "basketball-v2"
        #vqvae = diffusion.model.traj_model.vqvae
        #clip_model = dataset.clip_model
        #gripper_list = np.concatenate([np.array([-1.0, 0.0, 0.10000000149011612, 0.5, 0.6000000238418579, 0.6499999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]), np.zeros(48)])
        tokens = tokenize(description).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        lang_feats, lang_embs = self.dataset.clip_model.encode_text_with_embeddings(token_tensor)
        env_list = [self.create_env_and_policy(env_, seed=idx) for idx in range(num_evals)]
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        for each in env_list:
            each.render()
        obs_list = [env.reset()[0][None] for env in env_list]
        condition = torch.zeros([num_evals, 4, 260, 260, 3], device=device)
        #import pdb;pdb.set_trace()
        for i, env in enumerate(env_list):
            condition[i,-1] = torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(device)
        
        #imgs = [np.rot90(env.render(), 2)[110:370,110:370] for env in env_list]
        # obs = np.concatenate(obs_list, axis=0)
        for step in tqdm(range(0, max_episode_length),desc="Episode timestep ", total=max_episode_length):
            x_condition = self.ema_model.model.vqvae.encode(torch.stack([preprocess(condition[ind], 96) for ind in range(condition.shape[0])])).unsqueeze(1)
            #import pdb;pdb.set_trace()
            samples = self.ema_model.conditional_sample(x_condition, task=lang_feats.unsqueeze(0).repeat(num_evals,1,1).to(dtype=torch.float32), x_condition=x_condition, horizon=4, guidance=1.2)
            
            action = samples.trajectories[:, 0, :]
            action = action.detach().cpu().numpy()
            for i in range(len(env_list)):
                obs, reward, done, truncate, info = env_list[i].step(action[i])
                condition[i] = torch.cat([condition[i,1:], torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(device).unsqueeze(0)],dim=0)
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
        print(f"task:{env_},success rate:{sum(env_success_rate)/len(env_success_rate)}, mean episodic return:{sum(episode_rewards)/len(episode_rewards)}")
        return sum(env_success_rate)/len(env_success_rate), sum(episode_rewards)/len(episode_rewards)
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

class ContinuousTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        envs=[],
        task_list=[],
        is_unet=False,
        trainer_device=None,
        horizon=32,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = horizon
        self.task_list = task_list
        self.is_unet=is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.reset_parameters()
        self.step = 0
        # self.inverse_model = ARInvModel(hidden_dim=256, action_dim=self.dataset.action_dim,
        #                                 observation_dim=self.dataset.observation_dim, act='relu').to(self.device)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        #l1, l2 = [], []
        best_score, prev_label = 0, 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                #print("START!")
                batch = next(self.dataloader)
                #print("BATCH LOAD!")
                batch = batch_to_device(batch,self.device)
                #print("training step:", self.step, "gradient:", i)
                loss, infos = self.model.loss(*batch)
                #l1.append(loss)
                #for key, val in infos.items():
                #    l2.append(val)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                '''
                for name, parms in self.model.named_parameters(): 
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, 
                          ' -->grad_value:',parms.grad)
                '''
            #print("training step:", self.step)
            self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #self.writer.add_scalar('Inv_Loss', infos['inv_loss'], global_step=self.step)
            self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
                #score, success_rate = self.state_evaluate_dmc(self.device)
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                if self.step != 0:
                    if self.is_unet:
                        score, success_rate = self.evaluate(self.device)
                    else:
                        score, success_rate = self.evaluate(self.device)#self.evaluate_dmc_dt(self.device)
                    label = str(label) + '_' + str(score) + '_' + str(success_rate)
                    if score > best_score:
                        self.save(label)
                        best_score = score
                else:
                    #score, success_rate = self.evaluate(self.device)#self.evaluate_dmc_dt(self.device)
                    self.save(label)
                #self.save(label)


            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
                #l1,l2=torch.tensor(l1,device=loss.device),torch.tensor(l2,device=loss.device)
                #print(f'max:{l1.max()},Triquarter:{l1.quantile(0.75)},median:{l1.median()},mean:{l1.mean()},quarter:{l1.quantile(0.25)},min:{l1.min()}')
                #print(f'max:{l2.max()},Triquarter:{l2.quantile(0.75)},median:{l2.median()},mean:{l2.mean()},quarter:{l2.quantile(0.25)},min:{l2.min()}')
                #l1=[]
                #l2=[]
            '''
            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()
            '''
            self.step += 1
    def create_env_and_policy(self, env_name, seed=None):
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']
        env = env_cls(seed=seed, render_mode='rgb_array')
        env.camera_name="corner2"
        return env
    def evaluate(self, device):
        from tqdm import tqdm
        description = "Dunk the basketball into the basket"
        max_episode_length = 200
        num_evals = 20
        env_ = "basketball-v2"
        #vqvae = diffusion.model.traj_model.vqvae
        #clip_model = dataset.clip_model
        #gripper_list = np.concatenate([np.array([-1.0, 0.0, 0.10000000149011612, 0.5, 0.6000000238418579, 0.6499999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]), np.zeros(48)])
        tokens = tokenize(description).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        lang_feats, lang_embs = self.dataset.clip_model.encode_text_with_embeddings(token_tensor)
        env_list = [self.create_env_and_policy(env_, seed=idx) for idx in range(num_evals)]
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        for each in env_list:
            each.render()
        obs_list = [env.reset()[0][None] for env in env_list]
        condition = torch.zeros([num_evals, 1, 260, 260, 3], device=device)
        #import pdb;pdb.set_trace()
        for i, env in enumerate(env_list):
            condition[i,-1] = torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(device)
        
        #imgs = [np.rot90(env.render(), 2)[110:370,110:370] for env in env_list]
        # obs = np.concatenate(obs_list, axis=0)
        for step in tqdm(range(0, max_episode_length),desc="Episode timestep ", total=max_episode_length):
            x_condition = torch.stack([condition[ind] for ind in range(condition.shape[0])])
            #import pdb;pdb.set_trace()
            samples = self.ema_model.conditional_sample(x_condition, task=lang_feats.unsqueeze(0).repeat(num_evals,1,1).to(dtype=torch.float32), x_condition=x_condition, horizon=4, guidance=1.2)
            
            action = samples.trajectories[:, 0, :]
            action = action.detach().cpu().numpy()
            for i in range(len(env_list)):
                obs, reward, done, truncate, info = env_list[i].step(action[i])
                condition[i, 0] = torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(device)
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
        print(f"task:{env_},success rate:{sum(env_success_rate)/len(env_success_rate)}, mean episodic return:{sum(episode_rewards)/len(episode_rewards)}")
        return sum(env_success_rate)/len(env_success_rate), sum(episode_rewards)/len(episode_rewards)
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    
class R3MTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        trainer_device=None,
        horizon=32,
        distributed=False,
        gpuid=None,
        pretrain=True,
    ):
        super().__init__()
        
        self.model = diffusion_model
        
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        #self.model.cuda()
        #self.device = self.model.device
        self.device = trainer_device
        self.horizon = horizon

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.distributed = distributed
        if distributed:
            print('Distributed, begin DDP the model...')
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(diffusion_model, device_ids=[None], find_unused_parameters=False)
            print('Distributed, DDP model done!')
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=True)
            self.dataloader = cycle(torch.utils.data.DataLoader(self.dataset, 
                                               batch_size=train_batch_size, 
                                               shuffle=(train_sampler is None),
                                               num_workers=1, 
                                               pin_memory=True, 
                                               sampler=train_sampler, 
                                               drop_last=True,
                                               persistent_workers=True))
            
        else:
            self.dataloader = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
                ))
        self.dataloader_vis = None
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.reset_parameters()
        self.step = 0
        if distributed:
            self.gpuid=gpuid
            if self.gpuid == 0:
                self.writer = SummaryWriter(self.logdir)
        else:
            self.writer = SummaryWriter(self.logdir)

    def reset_parameters(self):
        if self.distributed:
            self.ema_model.load_state_dict(self.model.module.state_dict())
        else:
            self.ema_model.load_state_dict(self.model.state_dict())
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        best_score=0
        #l1, l2 = [], []
        best_score, prev_label = 0, 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                #print("START!")
                batch = next(self.dataloader)
                #print("BATCH LOAD!")
                batch = batch_to_device(batch, self.device)
                #print("training step:", self.step, "gradient:", i)
                if self.distributed: loss, infos = self.model.module.loss(*batch)
                else: loss, infos = self.model.loss(*batch)
                #print("loss computed")
                #l1.append(loss)
                #for key, val in infos.items():
                #    l2.append(val)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                '''
                for name, parms in self.model.named_parameters(): 
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, 
                          ' -->grad_value:',parms.grad)
                '''
            #print("loss backward")
            #print("training step:", self.step)
            # if self.distributed and self.gpuid==0:
            #     self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #     self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
            #     self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
            #     self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            # else:
            #     self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #     self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
            #     self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
            #     self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.distributed:
                if self.gpuid==0:
                    if self.step % self.update_ema_every == 0:
                        self.step_ema()
            else:
                if self.step % self.update_ema_every == 0:
                        self.step_ema()
                #score, success_rate = self.state_evaluate_dmc(self.device)
            

            if self.distributed:
                # infos['a0_loss'] = torch.distributed.all_reduce([torch.tensor(infos['a0_loss'])], op=torch.distributed.ReduceOp.SUM)
                # infos['loss_state'] = torch.distributed.all_reduce([torch.tensor(infos['loss_state'])], op=torch.distributed.ReduceOp.SUM)
                # infos['loss_action'] = torch.distributed.all_reduce([torch.tensor(infos['loss_action'])], op=torch.distributed.ReduceOp.SUM)
                # loss = torch.distributed.all_reduce([loss], op=torch.distributed.ReduceOp.SUM)
                if self.gpuid == 0:
                    self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
                    if 'loss_state' in infos:
                        self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
                        self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
                    self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
                    if self.step % self.save_freq == 0:
                        label = self.step // self.label_freq * self.label_freq
                        suc_rate, score = self.state_evaluate(self.device)
                        label = str(label) + '_' + str(score) + '_' + str(success_rate)
                        if score > best_score:
                            self.save(label)
                            best_score = score
                        #self.save(label)
                    if self.step % self.log_freq == 0:
                        infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                        print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            else:
                if self.step % self.save_freq == 0 and self.step != 0:
                    label = self.step // self.label_freq * self.label_freq
                    suc_rate, score = self.state_evaluate(self.device)
                    label = str(label) + '_' + str(score) + '_' + str(suc_rate)
                    if score > best_score:
                        self.save(label)
                        best_score = score
                    #self.save(label)
                self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
                if 'loss_state' in infos:
                    self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
                    self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
                self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
                if self.step % self.log_freq == 0:
                    infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                    print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
                #l1,l2=torch.tensor(l1,device=loss.device),torch.tensor(l2,device=loss.device)
                #print(f'max:{l1.max()},Triquarter:{l1.quantile(0.75)},median:{l1.median()},mean:{l1.mean()},quarter:{l1.quantile(0.25)},min:{l1.min()}')
                #print(f'max:{l2.max()},Triquarter:{l2.quantile(0.75)},median:{l2.median()},mean:{l2.mean()},quarter:{l2.quantile(0.25)},min:{l2.min()}')
                #l1=[]
                #l2=[]
            '''
            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()
            '''
            self.step += 1
    def create_env_and_policy(self, env_name, seed=None):
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']
        env = env_cls(seed=seed, render_mode='rgb_array')
        env.camera_name="corner2"
        return env
    def state_evaluate(self,device):
        from tqdm import tqdm
        description = "Dunk the basketball into the basket"
        max_episode_length = 150
        num_evals = 20
        env_ = "basketball-v2"
        #vqvae = diffusion.model.traj_model.vqvae
        #clip_model = dataset.clip_model
        gripper_list = np.concatenate([np.array([-1.0, 0.0, 0.10000000149011612, 0.5, 0.6000000238418579, 0.6499999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]), np.zeros(48)])
        tokens = tokenize(description).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        lang_feats, lang_embs = self.dataset.clip_model.encode_text_with_embeddings(token_tensor)
        env_list = [self.create_env_and_policy(env_, seed=idx) for idx in range(num_evals)]
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        for each in env_list:
            each.render()
        obs_list = [env.reset()[0][None] for env in env_list]
        condition = torch.zeros([num_evals, 1, 260, 260, 3], device=device)
        #import pdb;pdb.set_trace()
        for i, env in enumerate(env_list):
            condition[i,-1] = torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(device)
        
        #imgs = [np.rot90(env.render(), 2)[110:370,110:370] for env in env_list]
        # obs = np.concatenate(obs_list, axis=0)
        for step in tqdm(range(0, max_episode_length),desc="Episode timestep ", total=max_episode_length):
            x_condition = torch.stack([condition[ind] for ind in range(condition.shape[0])])
            #import pdb;pdb.set_trace()
            act = self.ema_model.sample_mask(num_evals, task=lang_feats.unsqueeze(0).repeat(num_evals,1,1).to(dtype=torch.float32), x_condition=x_condition)
            
            trans = self.dataset.discretizer.reconstruct(act[:,0,:3])
            grippers = gripper_list[(act[:,0,-1:]).cpu().numpy()]
            action = np.concatenate([trans, grippers], axis=-1)
            for i in range(len(env_list)):
                obs, reward, done, truncate, info = env_list[i].step(action[i])
                condition[i, 0] = torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(device)
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
        print(f"task:{env_},success rate:{sum(env_success_rate)/len(env_success_rate)}, mean episodic return:{sum(episode_rewards)/len(episode_rewards)}")
        return sum(env_success_rate)/len(env_success_rate), sum(episode_rewards)/len(episode_rewards)

    def evaluate(self, device):
        num_eval = 10
        task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in self.envs]
        mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in self.envs]
        env_list = [mt1[i].train_classes[self.envs[i]]() for j in range(num_eval) for i in range(len(self.envs))]
        seed = 0
        for i in range(len(env_list)):
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 500
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.envs))],
                                 device=device).reshape(-1, )
        conditions = torch.zeros([obs.shape[0], 2, obs.shape[-1]], device=device)
        rtg = torch.ones((len(env_list),), device=device) * 9
        for j in range(max_episode_length):
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=device).unsqueeze(1)], dim=1)
            samples = self.ema_model.conditional_sample(conditions, task=cond_task,
                                                           value=rtg,
                                                           verbose=False, horizon=self.horizon, guidance=1.2)
            action = samples.trajectories[:, 0, :]
            action = action.reshape(-1, self.dataset.action_dim)
            action = to_np(action)
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            for i in range(len(env_list)):
                next_observation, reward, terminal, info = env_list[i].step(action[i])
                obs_list.append(next_observation[None])
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
            obs = np.concatenate(obs_list, axis=0)
        for i in range(len(self.envs)):
            tmp = []
            tmp_suc = 0
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * len(self.envs)])
                tmp_suc += env_success_rate[i + j * len(self.envs)]
            this_score = statistics.mean(tmp)
            success = tmp_suc / num_eval
            total_success += success
            score += this_score
            print(f"task:{self.envs[i]},success rate:{success}, mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        print('Total success rate:', total_success / len(self.envs))
        return score, total_success / len(self.envs)
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            #'model': self.model.state_dict(), 
            #TODO,13/11/2023
            'model': self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'],strict=False)
        self.ema_model.load_state_dict(data['ema'],strict=False)

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
class SodaTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        trainer_device=None,
        horizon=32,
        distributed=False,
        gpuid=None,
        pretrain=True,
    ):
        super().__init__()
        
        self.model = diffusion_model
        self.pretrain=pretrain
        if not pretrain:
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/test/-Dec01_10-22-52", 'state_200000.pt')
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/meta/-Dec28_14-16-13", 'state_0.pt')
            loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/meta/-Jan19_23-23-04", 'encoder.pt') #metaworld
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/meta/-Dec28_14-16-13", 'state_600000.pt')
            
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/meta/-Dec29_07-49-13", 'traj_model.pt')
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/test/-Dec29_13-10-17", 'state_200000.pt') #metaworld
        
            data = torch.load(loadpath,map_location=trainer_device)
            #import pdb;pdb.set_trace()
            self.model.model.encoder.load_state_dict(data, strict=True)

            # self.model.model.traj_model.load_state_dict(data['traj_model'], strict=True)
            #import pdb;pdb.set_trace()
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        #self.model.cuda()
        #self.device = self.model.device
        self.device = trainer_device
        self.horizon = horizon

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.distributed = distributed
        if distributed:
            print('Distributed, begin DDP the model...')
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(diffusion_model, device_ids=[None], find_unused_parameters=False)
            print('Distributed, DDP model done!')
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=True)
            self.dataloader = cycle(torch.utils.data.DataLoader(self.dataset, 
                                               batch_size=train_batch_size, 
                                               shuffle=(train_sampler is None),
                                               num_workers=1, 
                                               pin_memory=True, 
                                               sampler=train_sampler, 
                                               drop_last=True,
                                               persistent_workers=True))
            
        else:
            self.dataloader = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
                ))
        self.dataloader_vis = None
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.reset_parameters()
        self.step = 0
        if distributed:
            self.gpuid=gpuid
            if self.gpuid == 0:
                self.writer = SummaryWriter(self.logdir)
        else:
            self.writer = SummaryWriter(self.logdir)

    def reset_parameters(self):
        if self.distributed:
            self.ema_model.load_state_dict(self.model.module.state_dict())
        else:
            self.ema_model.load_state_dict(self.model.state_dict())
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        best_score=0
        #l1, l2 = [], []
        best_score, prev_label = 0, 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                #print("START!")
                batch = next(self.dataloader)
                #print("BATCH LOAD!")
                batch = batch_to_device(batch, self.device)
                #print("training step:", self.step, "gradient:", i)
                if self.distributed: loss, infos = self.model.module.loss(*batch)
                else: loss, infos = self.model.loss(*batch)
                #print("loss computed")
                #l1.append(loss)
                #for key, val in infos.items():
                #    l2.append(val)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                '''
                for name, parms in self.model.named_parameters(): 
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, 
                          ' -->grad_value:',parms.grad)
                '''
            #print("loss backward")
            #print("training step:", self.step)
            # if self.distributed and self.gpuid==0:
            #     self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #     self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
            #     self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
            #     self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            # else:
            #     self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #     self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
            #     self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
            #     self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.distributed:
                if self.gpuid==0:
                    if self.step % self.update_ema_every == 0:
                        self.step_ema()
            else:
                if self.step % self.update_ema_every == 0:
                        self.step_ema()
                #score, success_rate = self.state_evaluate_dmc(self.device)
            

            if self.distributed:
                # infos['a0_loss'] = torch.distributed.all_reduce([torch.tensor(infos['a0_loss'])], op=torch.distributed.ReduceOp.SUM)
                # infos['loss_state'] = torch.distributed.all_reduce([torch.tensor(infos['loss_state'])], op=torch.distributed.ReduceOp.SUM)
                # infos['loss_action'] = torch.distributed.all_reduce([torch.tensor(infos['loss_action'])], op=torch.distributed.ReduceOp.SUM)
                # loss = torch.distributed.all_reduce([loss], op=torch.distributed.ReduceOp.SUM)
                if self.gpuid == 0:
                    self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
                    if 'loss_state' in infos:
                        self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
                        self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
                    self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
                    if self.step % self.save_freq == 0:
                        label = self.step // self.label_freq * self.label_freq
                        if not self.pretrain:
                            suc_rate, score = self.state_evaluate(self.device)
                            label = str(label) + '_' + str(score) + '_' + str(success_rate)
                            if score > best_score:
                                self.save(label)
                                best_score = score
                        else:
                            self.save(label)
                        #self.save(label)
                    if self.step % self.log_freq == 0:
                        infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                        print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            else:
                if self.step % self.save_freq == 0 and self.step != 0:
                    label = self.step // self.label_freq * self.label_freq
                    if not self.pretrain:
                        suc_rate, score = self.state_evaluate(self.device)
                        label = str(label) + '_' + str(score) + '_' + str(suc_rate)
                        if score > best_score:
                            self.save(label)
                            best_score = score
                    else:
                        self.save(label)
                    #self.save(label)
                self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
                if 'loss_state' in infos:
                    self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
                    self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
                self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
                if self.step % self.log_freq == 0:
                    infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                    print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
                #l1,l2=torch.tensor(l1,device=loss.device),torch.tensor(l2,device=loss.device)
                #print(f'max:{l1.max()},Triquarter:{l1.quantile(0.75)},median:{l1.median()},mean:{l1.mean()},quarter:{l1.quantile(0.25)},min:{l1.min()}')
                #print(f'max:{l2.max()},Triquarter:{l2.quantile(0.75)},median:{l2.median()},mean:{l2.mean()},quarter:{l2.quantile(0.25)},min:{l2.min()}')
                #l1=[]
                #l2=[]
            '''
            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()
            '''
            self.step += 1
    def create_env_and_policy(self, env_name, seed=None):
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']
        env = env_cls(seed=seed, render_mode='rgb_array')
        env.camera_name="corner2"
        return env
    def state_evaluate(self,device):
        from tqdm import tqdm
        description = "Dunk the basketball into the basket"
        max_episode_length = 150
        num_evals = 20
        env_ = "basketball-v2"
        #vqvae = diffusion.model.traj_model.vqvae
        #clip_model = dataset.clip_model
        gripper_list = np.concatenate([np.array([-1.0, 0.0, 0.10000000149011612, 0.5, 0.6000000238418579, 0.6499999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]), np.zeros(256)])
        tokens = tokenize(description).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        lang_feats, lang_embs = self.dataset.clip_model.encode_text_with_embeddings(token_tensor)
        env_list = [self.create_env_and_policy(env_, seed=idx) for idx in range(num_evals)]
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        for each in env_list:
            each.render()
        obs_list = [env.reset()[0][None] for env in env_list]
        condition = torch.zeros([num_evals, 4, 260, 260, 3], device=device)
        #import pdb;pdb.set_trace()
        for i, env in enumerate(env_list):
            condition[i,-1] = torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(device)
        
        #imgs = [np.rot90(env.render(), 2)[110:370,110:370] for env in env_list]
        # obs = np.concatenate(obs_list, axis=0)
        for step in tqdm(range(0, max_episode_length),desc="Episode timestep ", total=max_episode_length):
            x_condition = self.ema_model.model.vqvae.encode(torch.stack([preprocess(condition[ind], 96) for ind in range(condition.shape[0])])).unsqueeze(1)
            #import pdb;pdb.set_trace()
            act = self.ema_model.sample_mask(num_evals, task=lang_feats.unsqueeze(0).repeat(num_evals,1,1).to(dtype=torch.float32), x_condition=x_condition)
            
            trans = self.dataset.discretizer.reconstruct(act[:,0,:3])
            grippers = gripper_list[(act[:,0,-1:]).cpu().numpy()]
            action = np.concatenate([trans, grippers], axis=-1)
            for i in range(len(env_list)):
                obs, reward, done, truncate, info = env_list[i].step(action[i])
                condition[i] = torch.cat([condition[i,1:], torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(device).unsqueeze(0)],dim=0)
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
        print(f"task:{env_},success rate:{sum(env_success_rate)/len(env_success_rate)}, mean episodic return:{sum(episode_rewards)/len(episode_rewards)}")
        return sum(env_success_rate)/len(env_success_rate), sum(episode_rewards)/len(episode_rewards)

    def evaluate(self, device):
        num_eval = 10
        task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in self.envs]
        mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in self.envs]
        env_list = [mt1[i].train_classes[self.envs[i]]() for j in range(num_eval) for i in range(len(self.envs))]
        seed = 0
        for i in range(len(env_list)):
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 500
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.envs))],
                                 device=device).reshape(-1, )
        conditions = torch.zeros([obs.shape[0], 2, obs.shape[-1]], device=device)
        rtg = torch.ones((len(env_list),), device=device) * 9
        for j in range(max_episode_length):
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=device).unsqueeze(1)], dim=1)
            samples = self.ema_model.conditional_sample(conditions, task=cond_task,
                                                           value=rtg,
                                                           verbose=False, horizon=self.horizon, guidance=1.2)
            action = samples.trajectories[:, 0, :]
            action = action.reshape(-1, self.dataset.action_dim)
            action = to_np(action)
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            for i in range(len(env_list)):
                next_observation, reward, terminal, info = env_list[i].step(action[i])
                obs_list.append(next_observation[None])
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
            obs = np.concatenate(obs_list, axis=0)
        for i in range(len(self.envs)):
            tmp = []
            tmp_suc = 0
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * len(self.envs)])
                tmp_suc += env_success_rate[i + j * len(self.envs)]
            this_score = statistics.mean(tmp)
            success = tmp_suc / num_eval
            total_success += success
            score += this_score
            print(f"task:{self.envs[i]},success rate:{success}, mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        print('Total success rate:', total_success / len(self.envs))
        return score, total_success / len(self.envs)
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            #'model': self.model.state_dict(), 
            #TODO,13/11/2023
            'model': self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'],strict=False)
        self.ema_model.load_state_dict(data['ema'],strict=False)
       

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)

class MetaworldTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        trainer_device=None,
        horizon=32,
        distributed=False,
        gpuid=None,
        pretrain=True,
    ):
        super().__init__()
        
        self.model = diffusion_model
        self.pretrain=pretrain
        if not pretrain:
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/test/-Dec01_10-22-52", 'state_200000.pt')
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/meta/-Dec28_14-16-13", 'state_0.pt')
            loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/meta/-Dec29_07-49-13", 'state_600000.pt') #metaworld
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/meta/-Dec28_14-16-13", 'state_600000.pt')
            
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/meta/-Dec29_07-49-13", 'traj_model.pt')
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/test/-Dec29_13-10-17", 'state_200000.pt') #metaworld
            data = torch.load(loadpath)
            self.model.load_state_dict(data['ema'], strict=False)
            # self.model.model.traj_model.load_state_dict(data['traj_model'], strict=True)
            #import pdb;pdb.set_trace()
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        #self.model.cuda()
        #self.device = self.model.device
        self.device = trainer_device
        self.horizon = horizon

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.distributed = distributed
        if distributed:
            print('Distributed, begin DDP the model...')
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(diffusion_model, device_ids=[None], find_unused_parameters=False)
            print('Distributed, DDP model done!')
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=True)
            self.dataloader = cycle(torch.utils.data.DataLoader(self.dataset, 
                                               batch_size=train_batch_size, 
                                               shuffle=(train_sampler is None),
                                               num_workers=1, 
                                               pin_memory=True, 
                                               sampler=train_sampler, 
                                               drop_last=True,
                                               persistent_workers=True))
            
        else:
            self.dataloader = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
                ))
        self.dataloader_vis = None
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.reset_parameters()
        self.step = 0
        if distributed:
            self.gpuid=gpuid
            if self.gpuid == 0:
                self.writer = SummaryWriter(self.logdir)
        else:
            self.writer = SummaryWriter(self.logdir)

    def reset_parameters(self):
        if self.distributed:
            self.ema_model.load_state_dict(self.model.module.state_dict())
        else:
            self.ema_model.load_state_dict(self.model.state_dict())
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        best_score=0
        #l1, l2 = [], []
        best_score, prev_label = 0, 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                #print("START!")
                batch = next(self.dataloader)
                #print("BATCH LOAD!")
                batch = batch_to_device(batch, self.device)
                #print("training step:", self.step, "gradient:", i)
                if self.distributed: loss, infos = self.model.module.loss(*batch)
                else: loss, infos = self.model.loss(*batch)
                #print("loss computed")
                #l1.append(loss)
                #for key, val in infos.items():
                #    l2.append(val)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                '''
                for name, parms in self.model.named_parameters(): 
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, 
                          ' -->grad_value:',parms.grad)
                '''
            #print("loss backward")
            #print("training step:", self.step)
            # if self.distributed and self.gpuid==0:
            #     self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #     self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
            #     self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
            #     self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            # else:
            #     self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #     self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
            #     self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
            #     self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.distributed:
                if self.gpuid==0:
                    if self.step % self.update_ema_every == 0:
                        self.step_ema()
            else:
                if self.step % self.update_ema_every == 0:
                        self.step_ema()
                #score, success_rate = self.state_evaluate_dmc(self.device)
            

            if self.distributed:
                # infos['a0_loss'] = torch.distributed.all_reduce([torch.tensor(infos['a0_loss'])], op=torch.distributed.ReduceOp.SUM)
                # infos['loss_state'] = torch.distributed.all_reduce([torch.tensor(infos['loss_state'])], op=torch.distributed.ReduceOp.SUM)
                # infos['loss_action'] = torch.distributed.all_reduce([torch.tensor(infos['loss_action'])], op=torch.distributed.ReduceOp.SUM)
                # loss = torch.distributed.all_reduce([loss], op=torch.distributed.ReduceOp.SUM)
                if self.gpuid == 0:
                    self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
                    if 'loss_state' in infos:
                        self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
                        self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
                    self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
                    if self.step % self.save_freq == 0:
                        label = self.step // self.label_freq * self.label_freq
                        if not self.pretrain:
                            suc_rate, score = self.state_evaluate(self.device)
                            label = str(label) + '_' + str(score) + '_' + str(success_rate)
                            if score > best_score:
                                self.save(label)
                                best_score = score
                        else:
                            self.save(label)
                        #self.save(label)
                    if self.step % self.log_freq == 0:
                        infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                        print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            else:
                if self.step % self.save_freq == 0 and self.step != 0:
                    label = self.step // self.label_freq * self.label_freq
                    if not self.pretrain:
                        suc_rate, score = self.state_evaluate(self.device)
                        label = str(label) + '_' + str(score) + '_' + str(success_rate)
                        if score > best_score:
                            self.save(label)
                            best_score = score
                    else:
                        self.save(label)
                    #self.save(label)
                self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
                if 'loss_state' in infos:
                    self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
                    self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
                self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
                if self.step % self.log_freq == 0:
                    infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                    print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
                #l1,l2=torch.tensor(l1,device=loss.device),torch.tensor(l2,device=loss.device)
                #print(f'max:{l1.max()},Triquarter:{l1.quantile(0.75)},median:{l1.median()},mean:{l1.mean()},quarter:{l1.quantile(0.25)},min:{l1.min()}')
                #print(f'max:{l2.max()},Triquarter:{l2.quantile(0.75)},median:{l2.median()},mean:{l2.mean()},quarter:{l2.quantile(0.25)},min:{l2.min()}')
                #l1=[]
                #l2=[]
            '''
            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()
            '''
            self.step += 1
    def create_env_and_policy(self, env_name, seed=None):
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']
        env = env_cls(seed=seed, render_mode='rgb_array')
        env.camera_name="corner2"
        return env
    def state_evaluate(self,device):
        from tqdm import tqdm
        description = "Dunk the basketball into the basket"
        max_episode_length = 150
        num_evals = 20
        env_ = "basketball-v2"
        #vqvae = diffusion.model.traj_model.vqvae
        #clip_model = dataset.clip_model
        gripper_list = np.concatenate([np.array([-1.0, 0.0, 0.10000000149011612, 0.5, 0.6000000238418579, 0.6499999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]), np.zeros(256)])
        tokens = tokenize(description).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        lang_feats, lang_embs = self.dataset.clip_model.encode_text_with_embeddings(token_tensor)
        env_list = [self.create_env_and_policy(env_, seed=idx) for idx in range(num_evals)]
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        for each in env_list:
            each.render()
        obs_list = [env.reset()[0][None] for env in env_list]
        condition = torch.zeros([num_evals, 4, 260, 260, 3], device=device)
        #import pdb;pdb.set_trace()
        for i, env in enumerate(env_list):
            condition[i,-1] = torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(device)
        
        #imgs = [np.rot90(env.render(), 2)[110:370,110:370] for env in env_list]
        # obs = np.concatenate(obs_list, axis=0)
        for step in tqdm(range(0, max_episode_length),desc="Episode timestep ", total=max_episode_length):
            x_condition = self.ema_model.model.traj_model.vqvae.encode(torch.stack([preprocess(condition[ind], 96) for ind in range(condition.shape[0])])).unsqueeze(1)
            #import pdb;pdb.set_trace()
            _, act = self.ema_model.sample_mask(num_evals, task=lang_feats.unsqueeze(0).repeat(num_evals,1,1).to(dtype=torch.float32), x_condition=x_condition.unsqueeze(0))
            
            trans = self.dataset.discretizer.reconstruct(act[:,0,:3]-2048)
            grippers = gripper_list[(act[:,0,-1:]-2048).cpu().numpy()]
            action = np.concatenate([trans, grippers], axis=-1)
            for i in range(len(env_list)):
                obs, reward, done, truncate, info = env_list[i].step(action[i])
                condition[i] = torch.cat([condition[i,1:], torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(device).unsqueeze(0)],dim=0)
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
        print(f"task:{env_},success rate:{sum(env_success_rate)/len(env_success_rate)}, mean episodic return:{sum(episode_rewards)/len(episode_rewards)}")
        return sum(env_success_rate)/len(env_success_rate), sum(episode_rewards)/len(episode_rewards)

    def evaluate(self, device):
        num_eval = 10
        task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in self.envs]
        mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in self.envs]
        env_list = [mt1[i].train_classes[self.envs[i]]() for j in range(num_eval) for i in range(len(self.envs))]
        seed = 0
        for i in range(len(env_list)):
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 500
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.envs))],
                                 device=device).reshape(-1, )
        conditions = torch.zeros([obs.shape[0], 2, obs.shape[-1]], device=device)
        rtg = torch.ones((len(env_list),), device=device) * 9
        for j in range(max_episode_length):
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=device).unsqueeze(1)], dim=1)
            samples = self.ema_model.conditional_sample(conditions, task=cond_task,
                                                           value=rtg,
                                                           verbose=False, horizon=self.horizon, guidance=1.2)
            action = samples.trajectories[:, 0, :]
            action = action.reshape(-1, self.dataset.action_dim)
            action = to_np(action)
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            for i in range(len(env_list)):
                next_observation, reward, terminal, info = env_list[i].step(action[i])
                obs_list.append(next_observation[None])
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
            obs = np.concatenate(obs_list, axis=0)
        for i in range(len(self.envs)):
            tmp = []
            tmp_suc = 0
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * len(self.envs)])
                tmp_suc += env_success_rate[i + j * len(self.envs)]
            this_score = statistics.mean(tmp)
            success = tmp_suc / num_eval
            total_success += success
            score += this_score
            print(f"task:{self.envs[i]},success rate:{success}, mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        print('Total success rate:', total_success / len(self.envs))
        return score, total_success / len(self.envs)
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            #'model': self.model.state_dict(), 
            #TODO,13/11/2023
            'model': self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'],strict=False)
        self.ema_model.load_state_dict(data['ema'],strict=False)

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
class MultiviewTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        trainer_device=None,
        horizon=32,
        distributed=False,
        gpuid=None,
        pretrain=True,
    ):
        super().__init__()
        
        self.model = diffusion_model
        if not pretrain:
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/test/-Dec01_10-22-52", 'state_200000.pt')
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/meta/-Dec28_14-16-13", 'state_0.pt')
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/meta/-Dec29_07-49-13", 'state_200000.pt') #metaworld
            loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/test/-Dec31_15-00-07", 'state_800000.pt') #metaworld/-Jan10_11-32-43
            #loadpath = os.path.join("/mnt/data/optimal/hehaoran/video_diff/logs/test/-Jan10_11-32-43", 'state_0.pt')
            data = torch.load(loadpath)
            self.model.load_state_dict(data['model'], strict=False)
            

        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        #self.model.cuda()
        #self.device = self.model.device
        self.device = trainer_device
        self.horizon = horizon

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.distributed = distributed
        if distributed:
            print('Distributed, begin DDP the model...')
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(diffusion_model, device_ids=[None], find_unused_parameters=False)
            print('Distributed, DDP model done!')
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=True)
            self.dataloader = cycle(torch.utils.data.DataLoader(self.dataset, 
                                               batch_size=train_batch_size, 
                                               shuffle=(train_sampler is None),
                                               num_workers=1, 
                                               pin_memory=True, 
                                               sampler=train_sampler, 
                                               drop_last=True,
                                               persistent_workers=True))
            
        else:
            self.dataloader = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
                ))
        self.dataloader_vis = None
        self.renderer = renderer
        if pretrain:
            self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        else:
            # self.optimizer = torch.optim.Adam([
            #     {'params':diffusion_model.model.traj_model.parameters(), 'lr':train_lr/5},
            #     {'params':diffusion_model.model.act_model.parameters(), 'lr':train_lr}],
            #     lr=train_lr)
            self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.reset_parameters()
        self.step = 0
        if distributed:
            self.gpuid=gpuid
            if self.gpuid == 0:
                self.writer = SummaryWriter(self.logdir)
        else:
            self.writer = SummaryWriter(self.logdir)

    def reset_parameters(self):
        if self.distributed:
            self.ema_model.load_state_dict(self.model.module.state_dict())
        else:
            self.ema_model.load_state_dict(self.model.state_dict())
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        #l1, l2 = [], []
        best_score, prev_label = 0, 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                #print("START!")
                batch = next(self.dataloader)
                #print("BATCH LOAD!")
                batch = batch_to_device(batch, self.device)
                #print("training step:", self.step, "gradient:", i)
                if self.distributed: loss, infos = self.model.module.loss(*batch)
                else: loss, infos = self.model.loss(*batch)
                #print("loss computed")
                #l1.append(loss)
                #for key, val in infos.items():
                #    l2.append(val)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                '''
                for name, parms in self.model.named_parameters(): 
                    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, 
                          ' -->grad_value:',parms.grad)
                '''
            #print("loss backward")
            #print("training step:", self.step)
            # if self.distributed and self.gpuid==0:
            #     self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #     self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
            #     self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
            #     self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            # else:
            #     self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            #     self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
            #     self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
            #     self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.distributed:
                if self.gpuid==0:
                    if self.step % self.update_ema_every == 0:
                        self.step_ema()
            else:
                if self.step % self.update_ema_every == 0:
                        self.step_ema()
                #score, success_rate = self.state_evaluate_dmc(self.device)
            

            if self.distributed:
                # infos['a0_loss'] = torch.distributed.all_reduce([torch.tensor(infos['a0_loss'])], op=torch.distributed.ReduceOp.SUM)
                # infos['loss_state'] = torch.distributed.all_reduce([torch.tensor(infos['loss_state'])], op=torch.distributed.ReduceOp.SUM)
                # infos['loss_action'] = torch.distributed.all_reduce([torch.tensor(infos['loss_action'])], op=torch.distributed.ReduceOp.SUM)
                # loss = torch.distributed.all_reduce([loss], op=torch.distributed.ReduceOp.SUM)
                if self.gpuid == 0:
                    self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
                    if 'loss_state' in infos:
                        self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
                        self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
                    self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
                    if self.step % self.save_freq == 0:
                        label = self.step #// self.label_freq * self.label_freq
                        self.save(label)
                    if self.step % self.log_freq == 0:
                        infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                        print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            else:
                if self.step % self.save_freq == 0:
                    label = self.step // self.label_freq * self.label_freq
                    self.save(label)
                self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
                if 'loss_state' in infos:
                    self.writer.add_scalar('Loss_state', infos['loss_state'], global_step=self.step)
                    self.writer.add_scalar('Loss_action', infos['loss_action'], global_step=self.step)
                self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
                if self.step % self.log_freq == 0:
                    infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                    print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
                #l1,l2=torch.tensor(l1,device=loss.device),torch.tensor(l2,device=loss.device)
                #print(f'max:{l1.max()},Triquarter:{l1.quantile(0.75)},median:{l1.median()},mean:{l1.mean()},quarter:{l1.quantile(0.25)},min:{l1.min()}')
                #print(f'max:{l2.max()},Triquarter:{l2.quantile(0.75)},median:{l2.median()},mean:{l2.mean()},quarter:{l2.quantile(0.25)},min:{l2.min()}')
                #l1=[]
                #l2=[]
            '''
            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()
            '''
            self.step += 1
    def state_evaluate(self,device):
        inverse_path = "./logs/inverse-dynamic/-Feb25_05-08-54/state_19000000.pt"
        self.inverse_model.load_state_dict(torch.load(inverse_path)['model'])
        num_eval = 10
        task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in self.envs]
        mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in self.envs]
        env_list = [mt1[i].train_classes[self.envs[i]]() for j in range(num_eval) for i in range(len(self.envs))]
        seed = 0
        for i in range(len(env_list)):
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 150
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.envs))],
                                 device=device).reshape(-1, )
        conditions = {}
        for k in range(10):
            conditions[k] = torch.zeros(obs.shape, device=device)
        rtg = torch.ones((len(env_list),), device=device) * 9
        for j in range(max_episode_length):
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            for key in sorted(conditions):
                if key == 9:
                    conditions[key] = to_torch(obs, device=device)
                else:
                    conditions[key] = conditions[key + 1]
            samples = self.ema_model.conditional_sample_v1(conditions, task=cond_task,
                                                            value=rtg,
                                                            verbose=False, horizon=self.horizon, guidance=1.2)
            obs_comb = torch.cat([samples.trajectories[:, 9, :],
                                  samples.trajectories[:, 10, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, 2 * self.dataset.observation_dim)
            action = self.inverse_model.forward0(obs_comb.clone().to(self.device), cond_task.unsqueeze(1))
            # action = self.ema_model.inv_model.forward0(obs_comb.clone().to(self.device),
            #                                     torch.tensor([i], device=device))
            # print(action.shape)
            action = action.reshape(-1, self.dataset.action_dim)
            # conditions[9] = torch.cat((action, conditions[9]), dim=-1).to(device=device)
            # print(action.shape)
            action = to_np(action)
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            for i in range(len(env_list)):
                next_observation, reward, terminal, info = env_list[i].step(action[i])
                obs_list.append(next_observation[None])
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
            obs = np.concatenate(obs_list, axis=0)
        for i in range(len(self.envs)):
            tmp = []
            tmp_suc = 0
            for j in range(num_eval):
                tmp.append(episode_rewards[i+j*50])
                tmp_suc += env_success_rate[i+j*50]
            this_score = statistics.mean(tmp)
            success = tmp_suc / num_eval
            total_success += success
            score += this_score
            print(f"task:{self.task_list[i]},success rate:{success}, mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        print('Total success rate:', total_success / len(self.envs))
        return score, total_success / len(self.envs)

    def evaluate(self, device):
        num_eval = 10
        task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in self.envs]
        mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in self.envs]
        env_list = [mt1[i].train_classes[self.envs[i]]() for j in range(num_eval) for i in range(len(self.envs))]
        seed = 0
        for i in range(len(env_list)):
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 500
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.envs))],
                                 device=device).reshape(-1, )
        conditions = torch.zeros([obs.shape[0], 2, obs.shape[-1]], device=device)
        rtg = torch.ones((len(env_list),), device=device) * 9
        for j in range(max_episode_length):
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=device).unsqueeze(1)], dim=1)
            samples = self.ema_model.conditional_sample(conditions, task=cond_task,
                                                           value=rtg,
                                                           verbose=False, horizon=self.horizon, guidance=1.2)
            action = samples.trajectories[:, 0, :]
            action = action.reshape(-1, self.dataset.action_dim)
            action = to_np(action)
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            for i in range(len(env_list)):
                next_observation, reward, terminal, info = env_list[i].step(action[i])
                obs_list.append(next_observation[None])
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
            obs = np.concatenate(obs_list, axis=0)
        for i in range(len(self.envs)):
            tmp = []
            tmp_suc = 0
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * len(self.envs)])
                tmp_suc += env_success_rate[i + j * len(self.envs)]
            this_score = statistics.mean(tmp)
            success = tmp_suc / num_eval
            total_success += success
            score += this_score
            print(f"task:{self.envs[i]},success rate:{success}, mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        print('Total success rate:', total_success / len(self.envs))
        return score, total_success / len(self.envs)
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            #'model': self.model.state_dict(), 
            #TODO,13/11/2023
            'model': self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'],strict=False)
        self.ema_model.load_state_dict(data['ema'],strict=False)

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
class InvTrainer(object):
    def __init__(
        self,
        inv_model,
        dataset,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        trainer_device=None,
    ):
        super().__init__()
        self.model = inv_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.device = trainer_device

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.optimizer = torch.optim.Adam(inv_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch,self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer.add_scalar('Loss', loss, global_step=self.step)
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)
            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
            self.step += 1
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

class MTDT_Trainer(object):
    def __init__(
        self,
        model,
        dataset,
        train_batch_size=32,
        train_lr=1e-4,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        task_list=[],
        trainer_device=None,
        horizon=32,
        envs=None,
    ):
        super().__init__()
        self.device = trainer_device
        self.task_list = task_list[:2]
        self.horizon = horizon
        self.envs = envs[:2]
        self.model = model

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.loss_fn=torch.nn.MSELoss()
        self.step = 0


    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        best_score = 0
        for step in range(n_train_steps):
            batch = next(self.dataloader)
            batch = batch_to_device(batch, self.device)
            states, actions, rtg, timesteps, attention_mask = batch.observations, batch.actions, batch.rtg, \
                                                              batch.timestep, batch.mask
            action_target = torch.clone(actions)
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, None, rtg, timesteps, attention_mask=attention_mask,
            )
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            loss = self.loss_fn(action_preds, action_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer.add_scalar('Loss', loss, global_step=self.step)
            if self.step % self.save_freq == 0:
                torch.cuda.empty_cache()
                label = self.step // self.label_freq * self.label_freq
                score, success_rate = self.evaluate_meta(self.device)
                label = str(label) + '_' + str(score) + '_' + str(success_rate)
                if score > best_score:
                    self.save(label)
                    best_score = score
            if self.step % self.log_freq == 0:
                print(f'{self.step}: {loss:8.4f} | t: {timer():8.4f}', flush=True)
            self.step += 1
    def evaluate_meta(self,device):
        num_eval = 1
        task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in self.envs]
        mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in self.envs]
        env_list = [mt1[i].train_classes[self.envs[i]]() for j in range(num_eval) for i in range(len(self.envs))]
        seed = 0
        for i in range(len(env_list)):
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 500
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        ids = np.eye(50)
        obs = self.dataset.normalizer.normalize(obs, 'observations')
        cond_task = torch.tensor([ids[i] for j in range(num_eval) for i in range(len(self.envs))],
                                 device=device).reshape(-1, 50)
        observation = torch.cat([to_torch(obs, device=device), cond_task], dim=-1).unsqueeze(1)
        actions = torch.zeros((len(self.envs)*num_eval, 0, self.dataset.action_dim), device=device, dtype=torch.float32)
        timesteps = torch.zeros((len(env_list), 1), device=device, dtype=torch.long)
        rtg = torch.ones((len(env_list), 1, 1), device=device) * 9
        for j in range(max_episode_length):
            torch.cuda.empty_cache()
            actions = torch.cat([actions, torch.zeros((len(self.envs)*num_eval, 1, self.dataset.action_dim), device=device)], dim=1)
            action = self.model.get_action(
                observation.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                None,
                rtg.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                batch_size=num_eval*len(self.envs)
            )
            actions[:, -1, :] = action
            action = action.detach().cpu().numpy()
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            reward_list = []
            for i in range(len(env_list)):
                next_observation, reward, terminal, info = env_list[i].step(action[i])
                obs_list.append(next_observation[None])
                reward_list.append(reward)
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
            obs = np.concatenate(obs_list, axis=0)
            reward_list = torch.tensor(reward_list, device=device)
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            current_observation = torch.cat([to_torch(obs, device=device), cond_task], dim=-1).unsqueeze(1)
            observation = torch.cat([observation, current_observation], dim=1)
            pred_return = rtg[:, -1, 0] - (reward_list / 400)
            rtg = torch.cat(
                [rtg, pred_return.reshape(-1, 1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((len(env_list), 1), device=device, dtype=torch.long) * (j + 1)], dim=-1)
        for i in range(len(self.envs)):
            tmp = []
            tmp_suc = 0
            for j in range(num_eval):
                tmp.append(episode_rewards[i+j*50])
                tmp_suc += env_success_rate[i+j*50]
            this_score = statistics.mean(tmp)
            success = tmp_suc / num_eval
            total_success += success
            score += this_score
            print(f"task:{self.task_list[i]},success rate:{success}, mean episodic return:{this_score}, "
                  )#f"std:{statistics.stdev(tmp)}"
        print('Total success rate:', total_success / len(self.envs))
        return score, total_success / len(self.envs)
    def evaluate_dmc(self,device):
        env_list = [dmc.make(task, seed=0) for task in self.task_list]
        num_eval = 1
        score = 0
        scale = 100
        ids = np.eye(len(self.task_list))
        for i in range(len(env_list)):
            dones = 0
            eval_return = []
            while dones < num_eval:
                timestep = env_list[i].reset()
                obs = timestep.observation
                obs = self.dataset.normalizer.normalize(obs, 'observations')
                observation = np.concatenate((obs, ids[i]), axis=-1).reshape(1, -1)
                observation = to_torch(observation, device=device)
                actions = torch.zeros((0, self.dataset.action_dim), device=device, dtype=torch.float32)
                rtg = torch.tensor(900, device=device, dtype=torch.long).reshape(1, 1)
                timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
                total_reward, reward, t = 0, 0, 0
                while not timestep.last():
                    actions = torch.cat([actions, torch.zeros((1, self.dataset.action_dim), device=device)], dim=0)
                    action = self.model.get_action(
                        observation.to(dtype=torch.float32),
                        actions.to(dtype=torch.float32),
                        None,
                        rtg.to(dtype=torch.float32),
                        timesteps.to(dtype=torch.long),
                    )
                    actions[-1] = action
                    action = action.detach().cpu().numpy()
                    timestep = env_list[i].step(action)
                    next_obs, reward = timestep.observation, timestep.reward
                    total_reward += reward
                    obs = next_obs
                    obs = self.dataset.normalizer.normalize(obs, 'observations')
                    obs = np.concatenate((obs, ids[i]), axis=-1).reshape(1, -1)
                    current_observation = to_torch(obs, device=device)
                    observation = torch.cat([observation, current_observation], dim=0)
                    pred_return = rtg[0, -1] - (reward / scale)
                    rtg = torch.cat(
                        [rtg, pred_return.reshape(1, 1)], dim=1)
                    timesteps = torch.cat(
                        [timesteps,
                         torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
                dones += 1
                eval_return.append(total_reward)
            env_score = sum(eval_return) / len(eval_return)
            # env_success_rate.append(success / num_eval)
            print(f"task:{self.task_list[i]},episodic return:{env_score}")
            score += env_score
        return score, 1
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=True)

class PromptDT_Trainer(object):
    def __init__(
        self,
        model,
        dataset,
        train_batch_size=32,
        train_lr=1e-4,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        task_list=[],
        trainer_device=None,
        horizon=32,
        envs=None,
        prompt=None,
    ):
        super().__init__()
        self.model = model
        self.device = trainer_device
        self.task_list = task_list[:2]
        self.horizon = horizon
        self.envs = envs[:2]

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)
        self.prompt = [np.load(f"./metaworld_prompts/{self.envs[ind]}_prompt.npy", allow_pickle=True) for ind in range(len(self.envs))]
        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.loss_fn=torch.nn.MSELoss()
        self.step = 0
        self.envs = self.envs[:10]


    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        best_score = 0
        for step in range(n_train_steps):
            batch, prompt_batch = next(self.dataloader)
            batch, prompt_batch = batch_to_device(batch, self.device), batch_to_device(prompt_batch, self.device)
            states, actions, rtg, timesteps, attention_mask = batch.observations, batch.actions, batch.rtg,\
                                                              batch.timestep, batch.mask
            action_target = torch.clone(actions)
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, None, rtg, timesteps, attention_mask=attention_mask, prompt=prompt_batch
            )
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            loss = self.loss_fn(action_preds, action_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer.add_scalar('Loss', loss, global_step=self.step)
            if self.step % self.save_freq == 0:
                torch.cuda.empty_cache()
                label = self.step // self.label_freq * self.label_freq
                score, success_rate = self.evaluate_meta(self.device)
                label = str(label) + '_' + str(score) + '_' + str(success_rate)
                if score > best_score:
                    self.save(label)
                    best_score = score
            if self.step % self.log_freq == 0:
                print(f'{self.step}: {loss:8.4f} | t: {timer():8.4f}', flush=True)
            self.step += 1
    def get_prompt(self, prompt):
        start_steps = np.random.choice(
            np.arange(0, 200),
            size=1,
            replace=True,
        )
        obs = prompt['observations'][start_steps[0]:start_steps[0] + 20].reshape(20, -1)
        actions = prompt['actions'][start_steps[0]:start_steps[0] + 20].reshape(20, -1)
        rtg = self.discount_cumsum(prompt['rewards'][start_steps[0]:], gamma=1)[:20].reshape(20, -1)
        timestep = np.arange(start_steps[0], start_steps[0] + 20).reshape(20, -1)
        mask = np.ones(20)
        return DTBatch(actions, rtg, obs, timestep, mask)
    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
    def evaluate_meta(self, device):
        num_eval = 1

        task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in self.envs]
        mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in self.envs]
        env_list = [mt1[i].train_classes[self.envs[i]]() for j in range(num_eval) for i in range(len(self.envs))]
        start_steps = np.random.choice(
            np.arange(0, 200),
            size=1,
            replace=True,
        )
        p_actions = torch.tensor([self.prompt[i][0]['actions'][start_steps[0]:start_steps[0] + 20].reshape(20, -1)
                    for j in range(num_eval) for i in range(len(self.envs))])
        p_obs = torch.tensor([self.prompt[i][0]['observations'][start_steps[0]:start_steps[0] + 20].reshape(20, -1)
                 for j in range(num_eval) for i in range(len(self.envs))])
        p_rtg = torch.tensor([self.discount_cumsum(self.prompt[i][0]['rewards'][start_steps[0]:], gamma=1)[:20].reshape(20, -1)
                 for j in range(num_eval) for i in range(len(self.envs))])
        p_timestep = torch.tensor([np.arange(start_steps[0], start_steps[0] + 20).reshape(20, )
                      for j in range(num_eval) for i in range(len(self.envs))])
        p_mask = torch.tensor([np.ones(20) for j in range(num_eval) for i in range(len(self.envs))])
        prompt = DTBatch(p_actions, p_rtg, p_obs, p_timestep, p_mask)
        seed = 0
        for i in range(len(env_list)):
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 500
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        obs = self.dataset.normalizer.normalize(obs, 'observations')
        observation = to_torch(obs, device=device).unsqueeze(1)
        actions = torch.zeros((len(self.envs) * num_eval, 0, self.dataset.action_dim), device=device,
                              dtype=torch.float32)
        timesteps = torch.zeros((len(env_list), 1), device=device, dtype=torch.long)
        rtg = torch.ones((len(env_list), 1, 1), device=device) * 9
        for j in range(max_episode_length):
            torch.cuda.empty_cache()
            actions = torch.cat(
                [actions, torch.zeros((len(self.envs) * num_eval, 1, self.dataset.action_dim), device=device)], dim=1)
            action = self.model.get_action(
                observation.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                None,
                rtg.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                prompt=batch_to_device(prompt, device=device),
                batch_size=len(env_list)
            )
            actions[:, -1, :] = action
            action = action.detach().cpu().numpy()
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            reward_list = []
            for i in range(len(env_list)):
                next_observation, reward, terminal, info = env_list[i].step(action[i])
                obs_list.append(next_observation[None])
                reward_list.append(reward)
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
            obs = np.concatenate(obs_list, axis=0)
            reward_list = torch.tensor(reward_list, device=device)
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            current_observation = to_torch(obs, device=device).unsqueeze(1)
            observation = torch.cat([observation, current_observation], dim=1)
            pred_return = rtg[:, -1, 0] - (reward_list / 400)
            rtg = torch.cat(
                [rtg, pred_return.reshape(-1, 1, 1)], dim=-1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((len(env_list), 1), device=device, dtype=torch.long) * (j + 1)], dim=-1)
        for i in range(len(self.envs)):
            tmp = []
            tmp_suc = 0
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * 50])
                tmp_suc += env_success_rate[i + j * 50]
            this_score = statistics.mean(tmp)
            success = tmp_suc / num_eval
            total_success += success
            score += this_score
            print(f"task:{self.task_list[i]},success rate:{success}, mean episodic return:{this_score}, "
                  )#f"std:{statistics.stdev(tmp)}")
        print('Total success rate:', total_success / len(self.envs))
        return score, total_success / len(self.envs)
    def evaluate_maze(self, device):
        num_eval = 1

        task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in self.envs]
        mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in self.envs]
        env_list = [mt1[i].train_classes[self.envs[i]]() for j in range(num_eval) for i in range(len(self.envs))]
        start_steps = np.random.choice(
            np.arange(0, 200),
            size=1,
            replace=True,
        )
        p_actions = torch.tensor([self.prompt[i][0]['actions'][start_steps[0]:start_steps[0] + 20].reshape(20, -1)
                    for j in range(num_eval) for i in range(len(self.envs))])
        p_obs = torch.tensor([self.prompt[i][0]['observations'][start_steps[0]:start_steps[0] + 20].reshape(20, -1)
                 for j in range(num_eval) for i in range(len(self.envs))])
        p_rtg = torch.tensor([self.discount_cumsum(self.prompt[i][0]['rewards'][start_steps[0]:], gamma=1)[:20].reshape(20, -1)
                 for j in range(num_eval) for i in range(len(self.envs))])
        p_timestep = torch.tensor([np.arange(start_steps[0], start_steps[0] + 20).reshape(20, )
                      for j in range(num_eval) for i in range(len(self.envs))])
        p_mask = torch.tensor([np.ones(20) for j in range(num_eval) for i in range(len(self.envs))])
        prompt = DTBatch(p_actions, p_rtg, p_obs, p_timestep, p_mask)
        seed = 0
        for i in range(len(env_list)):
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 500
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        obs = self.dataset.normalizer.normalize(obs, 'observations')
        observation = to_torch(obs, device=device).unsqueeze(1)
        actions = torch.zeros((len(self.envs) * num_eval, 0, self.dataset.action_dim), device=device,
                              dtype=torch.float32)
        timesteps = torch.zeros((len(env_list), 1), device=device, dtype=torch.long)
        rtg = torch.ones((len(env_list), 1, 1), device=device) * 9
        for j in range(max_episode_length):
            torch.cuda.empty_cache()
            actions = torch.cat(
                [actions, torch.zeros((len(self.envs) * num_eval, 1, self.dataset.action_dim), device=device)], dim=1)
            action = self.model.get_action(
                observation.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                None,
                rtg.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                prompt=batch_to_device(prompt, device=device),
                batch_size=len(env_list)
            )
            actions[:, -1, :] = action
            action = action.detach().cpu().numpy()
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            reward_list = []
            for i in range(len(env_list)):
                next_observation, reward, terminal, info = env_list[i].step(action[i])
                obs_list.append(next_observation[None])
                reward_list.append(reward)
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
            obs = np.concatenate(obs_list, axis=0)
            reward_list = torch.tensor(reward_list, device=device)
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            current_observation = to_torch(obs, device=device).unsqueeze(1)
            observation = torch.cat([observation, current_observation], dim=1)
            pred_return = rtg[:, -1, 0] - (reward_list / 400)
            rtg = torch.cat(
                [rtg, pred_return.reshape(-1, 1, 1)], dim=-1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((len(env_list), 1), device=device, dtype=torch.long) * (j + 1)], dim=-1)
        for i in range(len(self.envs)):
            tmp = []
            tmp_suc = 0
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * 50])
                tmp_suc += env_success_rate[i + j * 50]
            this_score = statistics.mean(tmp)
            success = tmp_suc / num_eval
            total_success += success
            score += this_score
            print(f"task:{self.task_list[i]},success rate:{success}, mean episodic return:{this_score}, "
                  )#f"std:{statistics.stdev(tmp)}")
        print('Total success rate:', total_success / len(self.envs))
        return score, total_success / len(self.envs)
    def evaluate_dmc(self,device):
        env_list = [dmc.make(task, seed=0) for task in self.task_list]
        num_eval = 1
        score = 0
        scale = 100
        ids = np.eye(len(self.task_list))
        for i in range(len(env_list)):
            dones = 0
            eval_return = []
            while dones < num_eval:
                timestep = env_list[i].reset()
                obs = timestep.observation
                obs = self.dataset.normalizer.normalize(obs, 'observations')
                observation = np.concatenate((obs, ids[i]), axis=-1).reshape(1, -1)
                observation = to_torch(observation, device=device)
                actions = torch.zeros((0, self.dataset.action_dim), device=device, dtype=torch.float32)
                rtg = torch.tensor(900, device=device, dtype=torch.long).reshape(1, 1)
                timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
                total_reward, reward, t = 0, 0, 0
                while not timestep.last():
                    actions = torch.cat([actions, torch.zeros((1, self.dataset.action_dim), device=device)], dim=0)
                    action = self.model.get_action(
                        observation.to(dtype=torch.float32),
                        actions.to(dtype=torch.float32),
                        None,
                        rtg.to(dtype=torch.float32),
                        timesteps.to(dtype=torch.long),
                    )
                    actions[-1] = action
                    action = action.detach().cpu().numpy()
                    timestep = env_list[i].step(action)
                    next_obs, reward = timestep.observation, timestep.reward
                    total_reward += reward
                    obs = next_obs
                    obs = self.dataset.normalizer.normalize(obs, 'observations')
                    obs = np.concatenate((obs, ids[i]), axis=-1).reshape(1, -1)
                    current_observation = to_torch(obs, device=device)
                    observation = torch.cat([observation, current_observation], dim=0)
                    pred_return = rtg[0, -1] - (reward / scale)
                    rtg = torch.cat(
                        [rtg, pred_return.reshape(1, 1)], dim=1)
                    timesteps = torch.cat(
                        [timesteps,
                         torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
                dones += 1
                eval_return.append(total_reward)
            env_score = sum(eval_return) / len(eval_return)
            # env_success_rate.append(success / num_eval)
            print(f"task:{self.task_list[i]},episodic return:{env_score}")
            score += env_score
        return score, 1
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
class PromptDTMaze_Trainer(object):
    def __init__(
        self,
        model,
        dataset,
        train_batch_size=32,
        train_lr=1e-4,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        task_list=[],
        trainer_device=None,
        horizon=32,
        envs=None,
        prompt=None,
    ):
        super().__init__()
        self.model = model
        self.device = trainer_device
        self.task_list = task_list[:2]
        self.horizon = horizon
        self.envs = envs[:2]

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)
        #self.prompt = [np.load(f"./metaworld_prompts/{self.envs[ind]}_prompt.npy", allow_pickle=True) for ind in range(len(self.envs))]
        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.loss_fn=torch.nn.MSELoss()
        self.step = 0
        #self.envs = self.envs[:10]


    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        best_score = 0
        for step in range(n_train_steps):
            batch, prompt_batch = next(self.dataloader)
            batch, prompt_batch = batch_to_device(batch, self.device), to_torch(prompt_batch, device=self.device)
            states, actions, rtg, timesteps, attention_mask = batch.observations, batch.actions, batch.rtg,\
                                                              batch.timestep, batch.mask
            action_target = torch.clone(actions)
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, None, rtg, timesteps, attention_mask=attention_mask, prompt=prompt_batch
            )
            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            loss = self.loss_fn(action_preds, action_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer.add_scalar('Loss', loss, global_step=self.step)
            if self.step % self.save_freq == 0:
                torch.cuda.empty_cache()
                label = self.step // self.label_freq * self.label_freq
                score, success_rate = self.evaluate_maze(self.device)
                label = str(label) + '_' + str(score)
                if score > best_score:
                    self.save(label)
                    best_score = score
            if self.step % self.log_freq == 0:
                print(f'{self.step}: {loss:8.4f} | t: {timer():8.4f}', flush=True)
            self.step += 1
    def get_prompt(self, prompt):
        start_steps = np.random.choice(
            np.arange(0, 200),
            size=1,
            replace=True,
        )
        obs = prompt['observations'][start_steps[0]:start_steps[0] + 20].reshape(20, -1)
        actions = prompt['actions'][start_steps[0]:start_steps[0] + 20].reshape(20, -1)
        rtg = self.discount_cumsum(prompt['rewards'][start_steps[0]:], gamma=1)[:20].reshape(20, -1)
        timestep = np.arange(start_steps[0], start_steps[0] + 20).reshape(20, -1)
        mask = np.ones(20)
        return DTBatch(actions, rtg, obs, timestep, mask)
    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
    def evaluate_maze(self, device):
        num_eval = 1
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
        env_list = [gym.make(self.envs[i]) for j in range(num_eval) for i in range(len(self.envs))]
        prompt = [utils.parse_maze(dic[task_id]) for j in range(num_eval) for task_id in self.envs]
        dones = [False for j in range(num_eval) for i in range(len(self.envs))]
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 600
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        obs = self.dataset.normalizer.normalize(obs, 'observations')
        observation = to_torch(obs, device=device).unsqueeze(1)
        actions = torch.zeros((len(self.envs) * num_eval, 0, self.dataset.action_dim), device=device,
                              dtype=torch.float32)
        timesteps = torch.zeros((len(env_list), 1), device=device, dtype=torch.long)
        rtg = torch.ones((len(env_list), 1, 1), device=device) * 9
        j = 0
        while False in dones:
            torch.cuda.empty_cache()
            actions = torch.cat(
                [actions, torch.zeros((len(self.envs) * num_eval, 1, self.dataset.action_dim), device=device)], dim=1)
            action = self.model.get_action(
                observation.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                None,
                rtg.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                prompt=to_torch(prompt, device=device),
                batch_size=len(env_list)
            )
            actions[:, -1, :] = action
            action = action.detach().cpu().numpy()
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            reward_list = []
            for i in range(len(env_list)):
                if not dones[i]:
                    next_observation, reward, dones[i], info = env_list[i].step(action[i])
                    obs_list.append(next_observation[None])
                    reward_list.append(reward)
                    episode_rewards[i] += reward
                else:
                    obs_list.append(torch.zeros(1, self.dataset.observation_dim))
                    reward_list.append(0.)
            obs = np.concatenate(obs_list, axis=0)
            reward_list = torch.tensor(reward_list, device=device)
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            current_observation = to_torch(obs, device=device).unsqueeze(1)
            observation = torch.cat([observation, current_observation], dim=1)
            pred_return = rtg[:, -1, 0] - (reward_list / 400)
            rtg = torch.cat(
                [rtg, pred_return.reshape(-1, 1, 1)], dim=-1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((len(env_list), 1), device=device, dtype=torch.long) * (j + 1)], dim=-1)
            j += 1
        for i in range(len(self.envs)):
            tmp = []
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * len(self.envs)])
            this_score = statistics.mean(tmp)
            score += this_score
            print(f"task:{self.envs[i]},mean episodic return:{this_score}, ")
                  #f"std:{statistics.stdev(tmp)}")
        return score, 1.
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
class AugTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        envs=[],
        task_list=[],
        is_unet=False,
        trainer_device=None,
        horizon=32,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = horizon
        self.task_list = task_list
        self.is_unet=is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.reset_parameters()
        self.step = 0
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                #print("BATCH LOAD!")
                batch = batch_to_device(batch, self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            self.writer.add_scalar('Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)


            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            self.step += 1
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
class MazeTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        envs=[],
        task_list=[],
        is_unet=False,
        trainer_device=None,
        horizon=32,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = horizon
        self.task_list = task_list
        self.is_unet=is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.reset_parameters()
        self.step = 0
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        best_score = 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                #print("BATCH LOAD!")
                batch = batch_to_device(batch, self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            self.writer.add_scalar('Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                if self.step != 0:
                    score, success_rate = self.evaluate(self.device)#self.evaluate_dmc_dt(self.device)
                    label = str(label) + '_' + str(score) + '_' + str(success_rate)
                    if score > best_score:
                        self.save(label)
                        best_score = score
                else:
                    score, success_rate = self.evaluate(self.device)#self.evaluate_dmc_dt(self.device)
                    self.save(label)
                #label = self.step // self.label_freq * self.label_freq
                #self.save(label)


            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            self.step += 1
    def evaluate(self, device):
        num_eval = 10
        env_list = [gym.make(self.envs[i]) for j in range(num_eval) for i in range(len(self.envs))]
        score = 0
        dones = [False for j in range(num_eval) for i in range(len(self.envs))]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 600
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.envs))],
                                 device=device).reshape(-1, )
        conditions = torch.zeros([obs.shape[0], 5, obs.shape[-1]], device=device)
        rtg = torch.ones((len(env_list),), device=device) * 0.95
        while False in dones:#for _ in range(max_episode_length):#
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=device).unsqueeze(1)], dim=1)
            samples = self.ema_model.conditional_sample(conditions, task=cond_task,
                                                           value=rtg,
                                                           verbose=False, horizon=self.horizon, guidance=1.2)
            action = samples.trajectories[:, 0, :]
            action = action.reshape(-1, self.dataset.action_dim)
            action = to_np(action)
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            for i in range(len(env_list)):
                if not dones[i]:
                    next_observation, reward, dones[i], info = env_list[i].step(action[i])
                    obs_list.append(next_observation[None])
                    episode_rewards[i] += reward
                else:
                    obs_list.append(torch.zeros(1, self.dataset.observation_dim))
            obs = np.concatenate(obs_list, axis=0)
        for i in range(len(self.envs)):
            tmp = []
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * len(self.envs)])
            this_score = statistics.mean(tmp)
            score += this_score
            print(f"task:{self.envs[i]},mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        return score, 1.
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])