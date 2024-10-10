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


    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,

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
        Encoder_PATH = "Model path here"
        if not pretrain:

            loadpath = os.path.join(f"{Encoder_PATH}") #metaworld

        
            data = torch.load(loadpath,map_location=trainer_device)

            self.model.model.encoder.load_state_dict(data, strict=True)


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
        PATH = "Model Path Here"
        if not pretrain:

            loadpath = os.path.join(f"{PATH}") #metaworld

            data = torch.load(loadpath)
            self.model.load_state_dict(data['ema'], strict=False)

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
        PATH ='Your saved path here'
        if not pretrain:

            loadpath = os.path.join(PATH)

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
