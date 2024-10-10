from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
import math
import einops
import time
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import diffuser.utils as utils
from torch.cuda.amp import autocast
from .helpers import (
    extract,
    vp_beta_schedule,
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    apply_conditioning_v1,
    one_hot_dict,
    Losses,
    SinusoidalPosEmb,
)
from videogpt import VQVAE
import os
ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}

Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, task, value,  context_mask, t, **sample_kwargs):
    b, *_, device = *x.shape, x.device
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, task=task, value=value, context_mask=context_mask, t=t)
    #model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = 0.5*torch.randn_like(x)
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) #TODO:add nonzero_mask
    #noise = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    #noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t
#SECTION : auxiliary function for discrete diffusion models
def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    # assert x.max().item() < num_classes, \
    #     f'Error: {x.max().item()} >= {num_classes}'
    #print("###",x.max(),x.min())
    x_onehot = F.one_hot(x, num_classes)
    #print(">>>",x_onehot.argmax(dim=-1).max(),x_onehot.argmax(dim=-1).min())
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    #print(permute_order)
    x_onehot = x_onehot.permute(permute_order)
    #print(">>>",x_onehot.argmax(dim=1).max(),x_onehot.argmax(dim=1).min())
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    #print(log_x.min(),log_x.max(),log_x)
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)
def alpha_schedule(time_step, N_act=100, N_obs=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct
    bt_obs = (1-at-ct)/N_obs
    bt_act = (1-at-ct)/N_act
    att = np.concatenate((att[1:],[1]))
    ctt = np.concatenate((ctt[1:],[0]))
    btt_obs = (1-att-ctt)/N_obs
    btt_act = (1-att-ctt)/N_act
    return at, bt_obs, bt_act, ct, att, btt_obs, btt_act, ctt
class GaussianVideoDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, beta_schedule='vp',
        action_weight=1.0, loss_discount=1.0, loss_weights=None, drop_prob=0.25, obs_classes=2048, act_classes=256,
        auxiliary_loss_weight=5e-4, adaptive_auxiliary_loss=True, mask_weight=[1,1], pretrain=True,
        patch_size=2, hidden_dim=256, parametrization= 'x0', focal=False, force = False,
        alpha_init_type='alpha1', learnable_cf=False, vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
    ):
        super().__init__()
        self.drop_prob = torch.tensor(drop_prob)
        print(self.drop_prob)
        self.horizon = 4*horizon
        self.action_dim = action_dim
        self.parametrization = parametrization
        self.hidden_size = hidden_dim
        self.model = model
        self.pretrain = pretrain
        self.n_timesteps = int(n_timesteps)
        self.obs_classes = obs_classes
        self.act_classes = act_classes
        self.num_classes = self.obs_classes + self.act_classes + 1
        self.obs_length = 1*24*24
        self.gamma = 5
        #self.focal = True
        self.focal = focal
        self.force = force
        # if not pretrain:
        #     self.focal = False
        print("focal loss:",self.focal)
        ###SECTION : discrete diffusion model ###
        self.amp = False
        self.loss_type = 'vb_stochastic'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight
        if alpha_init_type == "alpha1":
            at, bt_obs, bt_act, ct, att, btt_obs, btt_act, ctt = alpha_schedule(self.n_timesteps, N_obs=self.obs_classes, N_act=self.act_classes)
        else:
            print("alpha_init_type is Wrong !! ")
        at = torch.tensor(at.astype('float64'))
        bt_obs= torch.tensor(bt_obs.astype('float64'))
        bt_act = torch.tensor(bt_act.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt_obs = torch.log(bt_obs)
        log_bt_act = torch.log(bt_act)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt_obs = torch.tensor(btt_obs.astype('float64'))
        btt_act = torch.tensor(btt_act.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt_obs = torch.log(btt_obs)
        log_cumprod_bt_act = torch.log(btt_act)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)
        
        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5 
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.n_timesteps
        self.diffusion_keep_list = [0] * self.n_timesteps
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt_obs', log_bt_obs.float())
        self.register_buffer('log_bt_act', log_bt_act.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt_obs', log_cumprod_bt_obs.float())
        self.register_buffer('log_cumprod_bt_act', log_cumprod_bt_act.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())
        
        self.register_buffer('Lt_history', torch.zeros(self.n_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.n_timesteps))
        self.step=0
    @property
    def device(self):
        return self.model.to_logits_act[-1].weight.device
    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl
    def q_pred_one_timestep(self, log_x_t, t):
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt_obs = extract(self.log_bt_obs, t, log_x_t.shape)             # bt
        log_bt_act = extract(self.log_bt_act, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:self.obs_classes,:]+log_at, log_bt_obs),
                log_add_exp(log_x_t[:,self.obs_classes:self.obs_classes+self.act_classes,:]+log_at, log_bt_act),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )
        #if self.pretrain: log_probs[:, :, self.obs_length:] = log_x_t[:, :, self.obs_length:]
        return log_probs
    def q_posterior(self, log_x_start, log_x_t, t, cond=None):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.n_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) 
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        if self.pretrain:
            log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
            #log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.obs_length)
            log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.shape[1])
        else:
            log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.shape[1])
            log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector
        

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)
    def q_pred_cond(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.n_timesteps + 1))%(self.n_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt_obs = extract(self.log_cumprod_bt_obs, t, log_x_start.shape)         # bt~
        log_cumprod_bt_act = extract(self.log_cumprod_bt_act, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        #NOTE：what does this mean
        log_probs = torch.cat(
                [
                    log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt_act),
   
                    log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 

                ],
                dim=1
            )

        return log_probs
    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.n_timesteps + 1))%(self.n_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt_obs = extract(self.log_cumprod_bt_obs, t, log_x_start.shape)         # bt~
        log_cumprod_bt_act = extract(self.log_cumprod_bt_act, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        #NOTE：what does this mean
        log_probs = torch.cat(
                [
                    log_add_exp(log_x_start[:,:self.obs_classes,:]+log_cumprod_at, log_cumprod_bt_obs),
                    log_add_exp(log_x_start[:,self.obs_classes:self.obs_classes+self.act_classes,:]+log_cumprod_at, log_cumprod_bt_act),
                    log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 
                    #torch.log(torch.exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct) + torch.exp(log_cumprod_ct))
                ],
                dim=1
            )
        #if self.pretrain: log_probs[:, :, self.obs_length:] = log_x_start[:, :, self.obs_length:]
        return log_probs
    def gumbel_sample(self, logits, flag=None):
        if flag =='obs':
            logits[:,self.obs_classes:-1,:] = -1e4
        else:
            logits[:,:self.obs_classes,:] = -1e4
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (logits + gumbel_noise).argmax(dim=1)
        return sample
    #REVIEW: Important to note ###
    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        #logits_obs = self.gumbel_sample(logits[:,:self.obs_classes,:-self.horizon*self.action_dim])
        #logits_act = self.gumbel_sample(logits[:,self.obs_classes:,-self.horizon*self.action_dim:]) + self.obs_classes
        logits_obs = self.gumbel_sample(logits[:,:,:-self.horizon*self.action_dim], flag='obs')
        logits_act = self.gumbel_sample(logits[:,:,-self.horizon*self.action_dim:], flag='act')
        sample = torch.cat([logits_obs, logits_act], dim=-1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def log_sample_categorical_sample(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        #print(">>>",logits.argmax(dim=1).max(),logits.argmax(dim=1).min())
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (logits + gumbel_noise).argmax(dim=1)
        #sample = self.gumbel2index(gumbel_noise + logits)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample
    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0[:,])
        #if self.pretrain: log_sample[:, :, self.obs_length:] = log_x_start[:, :, self.obs_length:]
        return log_sample
    def predict_start(self, log_x_t, t, task, value, x_condition, clf=False, force=True, imgs=None):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        max_neg_value = -1e4
        #print(x_t[:,self.obs_length:])
        #print("###",x_t.max(),x_t.min())
        if self.amp == True:
            with autocast():
                traj, action = self.model(x_t, t, task, value, x_condition=x_condition, clf=clf, force=force, pretrain=self.pretrain, imgs=imgs)
                #out = combine(traj, action, onehot=True)
        else:
            traj, action = self.model(x_t, t, task, value, x_condition=x_condition, clf=clf, force=force, pretrain=self.pretrain, imgs=imgs)
            #out = combine(traj, action, onehot=True)
        #action = torch.zeros((traj.shape[0], self.horizon*self.action_dim, self.act_classes)).to(traj.device)
        logits_traj = F.pad(traj,[0,self.act_classes],value=max_neg_value)
        logits_act = F.pad(action,[self.obs_classes,0],value=max_neg_value)

        out = combine(logits_traj, logits_act, onehot=True)
        #logits = logits_traj.permute(0,2,1)
        #out = einops.rearrange(logits_traj, 'B T H W c-> B c (T H W)')
        self.zero_vector = torch.zeros(log_x_t.size()[0], 1, self.shape[1]).type_as(log_x_t)- 70
        
        #print(out.shape)
        #NOTE : out is a one-hot prediction
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        #assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        # if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
        #     self.zero_vector = torch.zeros(batch_size, 1, self.content_seq_len).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred
    
    def p_losses(self, x_start, imgs, cond, task, value, t, pt):
        #x_start = torch.cat(x_start)
        #value = None
        #for i in range(10):
        #    cond[i] = einops.rearrange(cond[i], 'i j k -> (i j) k')
        self.shape = x_start.shape
        cond = einops.rearrange(cond, 'i j h b k c -> (i j) (h b) k c')
        task = einops.rearrange(task, 'i j d-> (i j) d') #task.reshape(-1, self.num_tasks)
        log_x_start = index_to_log_onehot(x_start.long(), self.num_classes)
        #pdb.set_trace()
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        #pdb.set_trace()
        #xt = log_onehot_to_index(log_xt)
        ############### go to p_theta function ###############
        if self.pretrain:
            # x_start = x_start[:,:self.obs_length]
            # log_xt = log_xt[:,:,:self.obs_length]
            # cond_prob = log_x_start[:,:,self.obs_length:]
            # log_x_start = log_x_start[:,:,:self.obs_length]
            # log_x0_recon_cond = self.predict_start(torch.cat([log_xt,cond_prob],dim=-1), t, task, value, x_condition=cond)            # P_theta(x0|xt)
            # log_x0_recon = log_x0_recon_cond
            # # log_x0_recon = torch.cat((log_x0_recon_cond,log_x_start[:,:,self._denoise_fn.transformer.image_length:]),-1)
            # log_model_prob = self.q_posterior(log_x_start=log_x0_recon_cond, log_x_t=log_xt, t=t)        
        ################## compute acc list ################
            #log_xt[:, :, self.obs_length:] = log_x_start[:, :, self.obs_length:]
            log_x0_recon = self.predict_start(log_xt, t, task, value, x_condition=cond, force=self.force)            # P_theta(x0|xt)
            #log_x0_recon[:, :, self.obs_length:] = log_x_start[:, :, self.obs_length:]
            log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)       # go through q(xt_1|xt,x0)
            # log_x0_recon_cond_act = torch.cat((log_x0_recon[:,:,:self.obs_length],log_x_start[:,:,self.obs_length:]),-1)
            # log_x0_recon_cond_obs = torch.cat((log_x_start[:,:,:self.obs_length],log_x0_recon[:,:,self.obs_length:]),-1)

            # log_model_prob_cond_act = self.q_posterior(log_x_start=log_x0_recon_cond_act, log_x_t=log_xt, t=t)      # go through q(xt_1_img|xt,x0,xt_1_txt)
            # log_model_prob_cond_obs = self.q_posterior(log_x_start=log_x0_recon_cond_obs, log_x_t=log_xt, t=t)      # go through q(xt_1_txt|xt,x0,xt_1_img)

            # log_model_prob = torch.cat([log_model_prob_cond_act[:,:,:self.obs_length],log_model_prob_cond_obs[:,:,self.obs_length:]],-1)
            
        else:
            log_x0_recon = self.predict_start(log_xt, t, task, value, x_condition=cond, imgs=imgs, force=self.force)            # P_theta(x0|xt)
            #log_x0_recon[:, :, self.obs_length:] = log_x_start[:, :, self.obs_length:]
            log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)       # go through q(xt_1|xt,x0)
            #print(log_xt.shape)
            # log_x0_recon = self.predict_start(log_xt, t, task, value, x_condition=cond, imgs=imgs)            # P_theta(x0|xt)
            # #log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)
            # log_x0_recon_ = log_x0_recon
            # #log_x0_recon_cond_act = torch.cat((log_x0_recon[:,:,:self.obs_length],log_x_start[:,:,self.obs_length:]),-1)
            # log_x0_recon_cond_act = log_x0_recon
            # log_x0_recon_cond_obs = torch.cat((log_x_start[:,:,:self.obs_length],log_x0_recon[:,:,self.obs_length:]),-1)

            # log_model_prob_cond_act = self.q_posterior(log_x_start=log_x0_recon_cond_act, log_x_t=log_xt, t=t)      # go through q(xt_1_img|xt,x0,xt_1_txt)
            # log_model_prob_cond_obs = self.q_posterior(log_x_start=log_x0_recon_cond_obs, log_x_t=log_xt, t=t)      # go through q(xt_1_txt|xt,x0,xt_1_img)

            # log_model_prob = torch.cat([log_model_prob_cond_act[:,:,:self.obs_length],log_model_prob_cond_obs[:,:,self.obs_length:]],-1)
        #log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)
         ################## compute acc list ################
        
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu()/x0_real.size()[1]  #compute acc rate
            self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9 #smooth acc rate and store
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu()/xt_recon.size()[1] #REVIEW
            self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9
        
        #logits[:,:self.obs_classes,:-self.horizon*self.action_dim])
        #logits_act = gumble_sample(logits[:,self.obs_classes:,-self.horizon*self.action_dim:]) +
        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)

        # log_x0_recon = log_x0_recon[:, :, self.obs_length:]
        # log_model_prob = log_model_prob[:, :, self.obs_length:]
        # log_true_prob = log_true_prob[:, :, self.obs_length:]
        # log_xt = log_xt[:, :, self.obs_length:]
        # x_start = x_start[:,self.obs_length:]
        # log_x_start = log_x_start[:, :, self.obs_length:]
        #if self.pretrain: log_true_prob[:, :, self.obs_length:] = 0
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        if self.focal:
            #import pdb;pdb.set_trace()
            weight_pt = torch.gather(F.softmax(log_x0_recon, dim=1), dim=1, index=x_start.long().unsqueeze(1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
            weight_pt = weight_pt.squeeze(1)
            #weight_pt = F.softmax(weight_pt, dim=1)
        else:
            weight_pt = 0
        #print(f"Min:{weight_pt.min()},Max:{weight_pt.max()},{weight_pt.shape}")
        #kl = self.multinomial_kl(log_true_prob[:,:,:-self.horizon*self.action_dim], log_model_prob[:,:, :-self.horizon*self.action_dim]) #TODO
        # mask_region = (xt == self.num_classes-1).float()
        # mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        # kl = kl * mask_weight
        log_loss_s = sum_except_batch(kl[:,:self.obs_length])
        log_loss_a_1 = sum_except_batch(kl[:,self.obs_length:self.obs_length+4])
        log_loss_a_2 = sum_except_batch(kl[:,self.obs_length+4:])
        if self.focal:
            kl = (1-weight_pt)**self.gamma*kl
        kl = sum_except_batch(kl)
        #import pdb;pdb.set_trace()
        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        if self.focal:
            #print("True")
            devoder_nll = (1-weight_pt)**self.gamma*decoder_nll
        mask = (t == torch.zeros_like(t)).float()
        log_loss_s = (mask * sum_except_batch(decoder_nll[:,:self.obs_length]) + (1. - mask) * log_loss_s) / pt
        log_loss_a_1 = (mask * sum_except_batch(decoder_nll[:,self.obs_length:self.obs_length+4]) + (1. -mask) * log_loss_a_1) / pt
        log_loss_a_2 = (mask * sum_except_batch(decoder_nll[:,self.obs_length+4:]) + (1. -mask) * log_loss_a_2) / pt
        #decoder_nll = -log_categorical(log_x_start[:,:,:-self.horizon*self.action_dim], log_model_prob[:,:,:-self.horizon*self.action_dim]) #TODO
        decoder_nll_s = sum_except_batch(decoder_nll[:,:self.obs_length])
        decoder_nll_a_1 = sum_except_batch(decoder_nll[:,self.obs_length:self.obs_length+4])
        decoder_nll_a_2 = sum_except_batch(decoder_nll[:,self.obs_length+4:])
        decoder_nll = sum_except_batch(decoder_nll)

        kl_loss = mask * decoder_nll + (1. - mask) * kl
        

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt 
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            #kl_aux = self.multinomial_kl(log_x_start[:,:self.obs_classes,:-self.horizon*self.action_dim], log_x0_recon[:,:self.obs_classes,:-self.horizon*self.action_dim]) #TODO
            #kl_aux = kl_aux * mask_weight
            #a_s = sum_except_batch(kl_aux[:,:self.obs_length])
            #a_a = sum_except_batch(kl_aux[:,self.obs_length:])
            aux_s = sum_except_batch(kl_aux[:,:self.obs_length])
            aux_a_1 = sum_except_batch(kl_aux[:,self.obs_length:self.obs_length+4])
            aux_a_2 = sum_except_batch(kl_aux[:,self.obs_length+4:])
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            kl_aux_loss_s = mask * decoder_nll_s + (1. - mask) * aux_s
            kl_aux_loss_a_1 = mask * decoder_nll_a_1 + (1. - mask) * aux_a_1
            kl_aux_loss_a_2 = mask * decoder_nll_a_2 + (1. - mask) * aux_a_2
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1-t/self.n_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            loss2_s = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_s / pt
            loss2_a_1 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_a_1 / pt
            loss2_a_2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_a_2 / pt
            vb_loss += loss2
        final_loss_s = log_loss_s + loss2_s
        final_loss_s = final_loss_s.sum()/(x_start.size()[0] * x_start.size()[1])
        final_loss_a_1 = log_loss_a_1 + loss2_a_1
        final_loss_a_2 = log_loss_a_2 + loss2_a_2
        final_loss_a = (5*final_loss_a_1.sum()+final_loss_a_2.sum())/(x_start.size()[0] * x_start.size()[1])
        # final_loss_a = final_loss_a.sum()/(x_start.size()[0] * x_start.size()[1])
        loss = vb_loss.sum()/(x_start.size()[0] * x_start.size()[1])
        #import pdb;pdb.set_trace()
        # if self.step%50==0:
        #     print(x0_recon[:,-16:])
        # self.step+=1
        info = {'a0_loss': loss,'loss_state':final_loss_s,'loss_action':final_loss_a}
        # noise = torch.randn_like(x_start)
        # context_mask = None
        # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)index_to_log_onehot
        # x_recon, act_recon = self.model(x_noisy, t, task, value, context_mask, x_condition=cond)
        # assert noise.shape == x_recon.shape

        # if self.predict_epsilon:
        #     loss, info = self.loss_fn(x_recon, noise)
        # else:
        #     loss, info = self.loss_fn(x_recon, x_start)

        return loss, info
        #return final_loss_a, info
        # return log_loss_a.sum()/(x_start.size()[0] * x_start.size()[1]), info
    def loss(self, x, *args):
        if self.pretrain:
            #x_obs = self.vqvae.codebook.dictionary_lookup(x['obs'])
            x_obs = einops.rearrange(x['obs'], 'i j h k b c-> (i j) (h k) b c')
            #print("MAX_MIN:",x_obs.max(),x_obs.min())
            x_act = torch.zeros((x_obs.shape[0], self.horizon, self.action_dim)).to(x['obs'].device) + self.obs_classes
            x = combine(x_obs, x_act)
            imgs = None
            #print("MAX_MIN:",x_obs.max(),x_obs.min())
        else:
            x_act = x['act'] + self.obs_classes
            imgs = x['imgs']
            #print(x_act)
            x_obs = einops.rearrange(x['obs'], 'i j h k b c-> (i j) (h k) b c')
            x_act = einops.rearrange(x_act, 'i j L D-> (i j) L D')
            x = combine(x_obs, x_act)
        batch_size, device = len(x), x.device
        #print(batch_size)
        t, pt = self.sample_time(batch_size, device, 'importance')
        diffusion_loss, info = self.p_losses(x, imgs, *args, t, pt) #TODO
        loss = diffusion_loss
        return loss, info


    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt)
        self.vqvae.eval()
    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.n_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.n_timesteps
            return t, pt
        else:
            raise ValueError
    # def p_pred(self, log_x, cond_emb, t):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
    #     if self.parametrization == 'x0':
    #         log_x_recon = self.predict_start(log_x, cond_emb, t)
    #         log_model_pred = self.q_posterior(
    #             log_x_start=log_x_recon, log_x_t=log_x, t=t)
    #     else:
    #         raise ValueError
    #     return log_model_pred, log_x_recon
    def sample_chain_mask_cond(self, num_samples, task=None, x_condition=None):
        b = num_samples
        device = self.log_at.device

        self.shape = (b,self.horizon*24*24+self.horizon*self.action_dim)
        zero_logits = torch.zeros((b, self.num_classes-1, self.obs_length),device=device)
        one_logits = torch.ones((b, 1, self.obs_length),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_obs = torch.log(mask_logits)
        cond = torch.zeros((b, self.horizon*self.action_dim), device=device).long()
        cond_prob = index_to_log_onehot(cond+self.obs_classes, self.num_classes)

        cond_prob = self.log_sample_categorical_sample(cond_prob) 
        log_obs = self.log_sample_categorical_sample(log_obs)

        zs = torch.zeros((self.n_timesteps, b) + (self.obs_length,), device=device).long()
        x_condition = einops.rearrange(x_condition, 'i j h b k c -> (i j) (h b) k c')
        task = einops.rearrange(task, 'i j d-> (i j) d') #task.reshape(-1, self.num_tasks)
        log_z = log_obs
        sample_type="top0.86r"
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])

        for i in reversed(range(0, self.n_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, task=task, x_condition=x_condition, cond='txt', cond_prob=cond_prob)

            zs[i] = log_onehot_to_index(log_z)
            
        print()
        #import pdb;pdb.set_trace()
        return einops.rearrange(zs[0], 'B (T H W) -> B T H W', T=4, H=24, W=24)
    @torch.no_grad()
    def sample_mask(self, num_samples, task=None, x_condition=None, f=None):
        b = num_samples
        device = self.log_at.device
        #device = 'cpu'
        #self.model = self.model.to(device)
        #self.horizon=1
        self.shape = (b,1*24*24+self.horizon*self.action_dim)
        zero_logits = torch.zeros((b, self.num_classes-1, self.shape[1]),device=device)
        #zero_logits = zero_logits[:,]
        one_logits = torch.ones((b, 1, self.shape[1]),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        #self.load_vqvae()
        # zs = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)

        #log_z = self.log_sample_categorical(log_z)
        x_condition = einops.rearrange(x_condition, 'i j h b k c -> (i j) (h b) k c')
        task = einops.rearrange(task, 'i j d-> (i j) d') #task.reshape(-1, self.num_tasks)
        sample_type="top0.88r"
        # sample_wrap = self.p_sample_with_truncation(self.p_sample,sample_type.split(',')[1])
        self.predict_start_ = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        from tqdm import tqdm
        #for i in tqdm(reversed(range(0, self.n_timesteps)),desc="Chain timestep ",total=self.n_timesteps):
        for i in reversed(range(0, self.n_timesteps)):
            # print(f'\nChain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, task=task, x_condition=x_condition)

            # zs[i] = log_onehot_to_index(log_z)

        zs=log_onehot_to_index(log_z)
        obs, act = split(zs, self.action_dim)
        return obs, act
    def predict_start_with_truncation(self, func, sample_type):

        truncation_r = float(sample_type[:-1].replace('top', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            # notice for different batches, out are same, we do it on out[0]
            temp, indices = torch.sort(out, 1, descending=True) 
            temp1 = torch.exp(temp)
            temp2 = temp1.cumsum(dim=1)
            temp3 = temp2 < truncation_r
            new_temp = torch.full_like(temp3[:,0:1,:], True)
            temp6 = torch.cat((new_temp, temp3), dim=1)
            temp3 = temp6[:,:-1,:]
            temp4 = temp3.gather(1, indices.argsort(1))
            temp5 = temp4.float()*out+(1-temp4.float())*(-70)
            probs = temp5
            return probs
        return wrapper
    @torch.no_grad()
    def p_sample(self, log_x, t, delta=None, cond=None, cond_prob=None, task=None, x_condition=None):
        model_log_prob = self.p_pred(log_x=log_x, t=t, delta=delta, cond=cond, cond_prob=cond_prob, task=task, x_condition=x_condition)
        out = self.log_sample_categorical(model_log_prob)
        return out
    def p_pred(self, log_x, t, delta=None, cond=None, cond_prob=None, task=None, x_condition=None):
        if self.parametrization == 'x0': 
            log_x_recon = self.predict_start_(log_x, t=t, task=task, x_condition=x_condition, value=None)
            if delta is None:
                delta = 0
            log_x0_recon = log_x_recon
            #import pdb; pdb.set_trace()
            if t[0].item() >= delta:
                log_model_pred = self.q_posterior(
                    log_x_start=log_x0_recon, log_x_t=log_x, t=t-delta, cond=cond)
            else:
                log_model_pred = self.q_posterior(
                    log_x_start=log_x0_recon, log_x_t=log_x, t=t, cond=cond)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t)
        else:
            raise ValueError
        return log_model_pred
    def cf_predict_start(self, log_x, t, task=None, x_condition=None, value=None):
        guidance_scale = 1.2
        log_x_recon = self.predict_start(log_x, t=t, task=task, x_condition=x_condition, value=None, force=True)[:, :-1]
        cf_log_x_recon = self.predict_start(log_x, t=t, task=task, x_condition=x_condition, value=None, clf=True, force=True)[:, :-1]
        log_new_x_recon = cf_log_x_recon + guidance_scale * (log_x_recon - cf_log_x_recon)
        log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
        log_new_x_recon = log_new_x_recon.clamp(-70, 0)
        log_pred = torch.cat((log_new_x_recon, self.zero_vector), dim=1)
        return log_pred

class MultiviewVideoDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, beta_schedule='vp',
        action_weight=1.0, loss_discount=1.0, loss_weights=None, drop_prob=0.25, obs_classes=2048, act_classes=256,
        auxiliary_loss_weight=5e-4, adaptive_auxiliary_loss=True, mask_weight=[1,1], pretrain=True,
        patch_size=2, hidden_dim=256, parametrization= 'x0', focal=False, force = False,
        alpha_init_type='alpha1', learnable_cf=False, vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
    ):
        super().__init__()
        self.drop_prob = torch.tensor(drop_prob)
        print(self.drop_prob)
        self.horizon = 4*horizon
        self.action_dim = action_dim
        self.parametrization = parametrization
        self.hidden_size = hidden_dim
        self.model = model
        self.pretrain = pretrain
        self.n_timesteps = int(n_timesteps)
        self.obs_classes = obs_classes
        self.act_classes = act_classes
        self.num_classes = self.obs_classes + self.act_classes + 1
        self.obs_length = 1*24*24
        self.gamma = 5
        #self.focal = True
        self.focal = focal
        self.force = force
        if not pretrain:
            self.focal = False
        ###SECTION : discrete diffusion model ###
        self.amp = False
        self.loss_type = 'vb_stochastic'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight
        if alpha_init_type == "alpha1":
            at, bt_obs, bt_act, ct, att, btt_obs, btt_act, ctt = alpha_schedule(self.n_timesteps, N_obs=self.obs_classes, N_act=self.act_classes)
        else:
            print("alpha_init_type is Wrong !! ")
        at = torch.tensor(at.astype('float64'))
        bt_obs= torch.tensor(bt_obs.astype('float64'))
        bt_act = torch.tensor(bt_act.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt_obs = torch.log(bt_obs)
        log_bt_act = torch.log(bt_act)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt_obs = torch.tensor(btt_obs.astype('float64'))
        btt_act = torch.tensor(btt_act.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt_obs = torch.log(btt_obs)
        log_cumprod_bt_act = torch.log(btt_act)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)
        
        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5 
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.n_timesteps
        self.diffusion_keep_list = [0] * self.n_timesteps
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt_obs', log_bt_obs.float())
        self.register_buffer('log_bt_act', log_bt_act.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt_obs', log_cumprod_bt_obs.float())
        self.register_buffer('log_cumprod_bt_act', log_cumprod_bt_act.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())
        
        self.register_buffer('Lt_history', torch.zeros(self.n_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.n_timesteps))
    
    @property
    def device(self):
        return self.model.to_logits_act[-1].weight.device
    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl
    def q_pred_one_timestep(self, log_x_t, t):
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt_obs = extract(self.log_bt_obs, t, log_x_t.shape)             # bt
        log_bt_act = extract(self.log_bt_act, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:self.obs_classes,:]+log_at, log_bt_obs),
                log_add_exp(log_x_t[:,self.obs_classes:self.obs_classes+self.act_classes,:]+log_at, log_bt_act),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )
        #if self.pretrain: log_probs[:, :, self.obs_length:] = log_x_t[:, :, self.obs_length:]
        return log_probs
    def q_posterior(self, log_x_start, log_x_t, t, cond=None):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.n_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) 
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        if self.pretrain:
            log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
            #log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.obs_length)
            log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.shape[1])
        else:
            log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.shape[1])
            log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector
        

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)
    def q_pred_cond(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.n_timesteps + 1))%(self.n_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt_obs = extract(self.log_cumprod_bt_obs, t, log_x_start.shape)         # bt~
        log_cumprod_bt_act = extract(self.log_cumprod_bt_act, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        #NOTE：what does this mean
        log_probs = torch.cat(
                [
                    log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt_act),
   
                    log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 

                ],
                dim=1
            )

        return log_probs
    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.n_timesteps + 1))%(self.n_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt_obs = extract(self.log_cumprod_bt_obs, t, log_x_start.shape)         # bt~
        log_cumprod_bt_act = extract(self.log_cumprod_bt_act, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        #NOTE：what does this mean
        log_probs = torch.cat(
                [
                    log_add_exp(log_x_start[:,:self.obs_classes,:]+log_cumprod_at, log_cumprod_bt_obs),
                    log_add_exp(log_x_start[:,self.obs_classes:self.obs_classes+self.act_classes,:]+log_cumprod_at, log_cumprod_bt_act),
                    log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 
                    #torch.log(torch.exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct) + torch.exp(log_cumprod_ct))
                ],
                dim=1
            )
        #if self.pretrain: log_probs[:, :, self.obs_length:] = log_x_start[:, :, self.obs_length:]
        return log_probs
    def gumbel_sample(self, logits, flag=None):
        if flag =='obs':
            logits[:,self.obs_classes:-1,:] = -1e4
        else:
            logits[:,:self.obs_classes,:] = -1e4
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (logits + gumbel_noise).argmax(dim=1)
        return sample
    #REVIEW: Important to note ###
    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        #logits_obs = self.gumbel_sample(logits[:,:self.obs_classes,:-self.horizon*self.action_dim])
        #logits_act = self.gumbel_sample(logits[:,self.obs_classes:,-self.horizon*self.action_dim:]) + self.obs_classes
        logits_obs = self.gumbel_sample(logits[:,:,:-self.horizon*self.action_dim], flag='obs')
        logits_act = self.gumbel_sample(logits[:,:,-self.horizon*self.action_dim:], flag='act')
        sample = torch.cat([logits_obs, logits_act], dim=-1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def log_sample_categorical_sample(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        #print(">>>",logits.argmax(dim=1).max(),logits.argmax(dim=1).min())
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (logits + gumbel_noise).argmax(dim=1)
        #sample = self.gumbel2index(gumbel_noise + logits)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample
    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0[:,])
        #if self.pretrain: log_sample[:, :, self.obs_length:] = log_x_start[:, :, self.obs_length:]
        return log_sample
    def predict_start(self, log_x_t, t, task, value, x_condition, view_condition=None, clf=False, force=True, imgs=None):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        max_neg_value = -1e4
        #print(x_t[:,self.obs_length:])
        #print("###",x_t.max(),x_t.min())
        if self.amp == True:
            with autocast():
                traj, action = self.model(x_t, t, task, value, x_condition=x_condition, view_condition=view_condition, clf=clf, force=force, pretrain=self.pretrain, imgs=imgs)
                #out = combine(traj, action, onehot=True)
        else:
            traj, action = self.model(x_t, t, task, value, x_condition=x_condition, view_condition=view_condition, clf=clf, force=force, pretrain=self.pretrain, imgs=imgs)
            #out = combine(traj, action, onehot=True)
        #action = torch.zeros((traj.shape[0], self.horizon*self.action_dim, self.act_classes)).to(traj.device)
        logits_traj = F.pad(traj,[0,self.act_classes],value=max_neg_value)
        logits_act = F.pad(action,[self.obs_classes,0],value=max_neg_value)

        out = combine(logits_traj, logits_act, onehot=True)
        #logits = logits_traj.permute(0,2,1)
        #out = einops.rearrange(logits_traj, 'B T H W c-> B c (T H W)')
        self.zero_vector = torch.zeros(log_x_t.size()[0], 1, self.shape[1]).type_as(log_x_t)- 70
        
        #print(out.shape)
        #NOTE : out is a one-hot prediction
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        #assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        # if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
        #     self.zero_vector = torch.zeros(batch_size, 1, self.content_seq_len).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred
    
    def p_losses(self, x_start, cond, task, value, t, pt,view_condition=None,imgs=None):
        #x_start = torch.cat(x_start)
        #value = None
        #for i in range(10):
        #    cond[i] = einops.rearrange(cond[i], 'i j k -> (i j) k')
        self.shape = x_start.shape
        # if not self.pretrain:
        #     cond = einops.rearrange(cond, 'i j h b k c -> i j (h b) k c')
        # else:
        cond = einops.rearrange(cond, 'i j h b k c -> (i j) (h b) k c')
        task = einops.rearrange(task, 'i j d-> (i j) d') #task.reshape(-1, self.num_tasks)
        log_x_start = index_to_log_onehot(x_start.long(), self.num_classes)
        #pdb.set_trace()
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        # left = self.q_sample(log_x_start = index_to_log_onehot(view_condition['left'].long(), self.num_classes), t=t)
        # right = self.q_sample(log_x_start = index_to_log_onehot(view_condition['right'].long(), self.num_classes), t=t)
        # wrist = self.q_sample(log_x_start = index_to_log_onehot(view_condition['wrist'].long(), self.num_classes), t=t)
        view_condition = {
            # 'left':log_onehot_to_index(left),
            # 'right':log_onehot_to_index(right),
            # 'wrist':log_onehot_to_index(wrist),
            'left':view_condition['left_'],
            'right':view_condition['right_'],
            'wrist':view_condition['wrist_']
        }
        #pdb.set_trace()
        #xt = log_onehot_to_index(log_xt)
        ############### go to p_theta function ###############
        if self.pretrain:
            # x_start = x_start[:,:self.obs_length]
            # log_xt = log_xt[:,:,:self.obs_length]
            # cond_prob = log_x_start[:,:,self.obs_length:]
            # log_x_start = log_x_start[:,:,:self.obs_length]
            # log_x0_recon_cond = self.predict_start(torch.cat([log_xt,cond_prob],dim=-1), t, task, value, x_condition=cond)            # P_theta(x0|xt)
            # log_x0_recon = log_x0_recon_cond
            # # log_x0_recon = torch.cat((log_x0_recon_cond,log_x_start[:,:,self._denoise_fn.transformer.image_length:]),-1)
            # log_model_prob = self.q_posterior(log_x_start=log_x0_recon_cond, log_x_t=log_xt, t=t)        
        ################## compute acc list ################
            #log_xt[:, :, self.obs_length:] = log_x_start[:, :, self.obs_length:]
            log_x0_recon = self.predict_start(log_xt, t, task, value, x_condition=cond, force=self.force)            # P_theta(x0|xt)
            #log_x0_recon[:, :, self.obs_length:] = log_x_start[:, :, self.obs_length:]
            log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)       # go through q(xt_1|xt,x0)
            # log_x0_recon_cond_act = torch.cat((log_x0_recon[:,:,:self.obs_length],log_x_start[:,:,self.obs_length:]),-1)
            # log_x0_recon_cond_obs = torch.cat((log_x_start[:,:,:self.obs_length],log_x0_recon[:,:,self.obs_length:]),-1)

            # log_model_prob_cond_act = self.q_posterior(log_x_start=log_x0_recon_cond_act, log_x_t=log_xt, t=t)      # go through q(xt_1_img|xt,x0,xt_1_txt)
            # log_model_prob_cond_obs = self.q_posterior(log_x_start=log_x0_recon_cond_obs, log_x_t=log_xt, t=t)      # go through q(xt_1_txt|xt,x0,xt_1_img)

            # log_model_prob = torch.cat([log_model_prob_cond_act[:,:,:self.obs_length],log_model_prob_cond_obs[:,:,self.obs_length:]],-1)
            
        else:
            log_x0_recon = self.predict_start(log_xt, t, task, value, x_condition=cond, view_condition=view_condition, force=self.force,imgs=imgs)            # P_theta(x0|xt)
            #log_x0_recon[:, :, self.obs_length:] = log_x_start[:, :, self.obs_length:]
            log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)       # go through q(xt_1|xt,x0)
            #print(log_xt.shape)
            # log_x0_recon = self.predict_start(log_xt, t, task, value, x_condition=cond, imgs=imgs)            # P_theta(x0|xt)
            # #log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)
            # log_x0_recon_ = log_x0_recon
            # #log_x0_recon_cond_act = torch.cat((log_x0_recon[:,:,:self.obs_length],log_x_start[:,:,self.obs_length:]),-1)
            # log_x0_recon_cond_act = log_x0_recon
            # log_x0_recon_cond_obs = torch.cat((log_x_start[:,:,:self.obs_length],log_x0_recon[:,:,self.obs_length:]),-1)

            # log_model_prob_cond_act = self.q_posterior(log_x_start=log_x0_recon_cond_act, log_x_t=log_xt, t=t)      # go through q(xt_1_img|xt,x0,xt_1_txt)
            # log_model_prob_cond_obs = self.q_posterior(log_x_start=log_x0_recon_cond_obs, log_x_t=log_xt, t=t)      # go through q(xt_1_txt|xt,x0,xt_1_img)

            # log_model_prob = torch.cat([log_model_prob_cond_act[:,:,:self.obs_length],log_model_prob_cond_obs[:,:,self.obs_length:]],-1)
        #log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)
         ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu()/x0_real.size()[1]  #compute acc rate
            self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9 #smooth acc rate and store
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu()/xt_recon.size()[1] #REVIEW
            self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9
        
        #logits[:,:self.obs_classes,:-self.horizon*self.action_dim])
        #logits_act = gumble_sample(logits[:,self.obs_classes:,-self.horizon*self.action_dim:]) +
        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)

        # log_x0_recon = log_x0_recon[:, :, self.obs_length:]
        # log_model_prob = log_model_prob[:, :, self.obs_length:]
        # log_true_prob = log_true_prob[:, :, self.obs_length:]
        # log_xt = log_xt[:, :, self.obs_length:]
        # x_start = x_start[:,self.obs_length:]
        # log_x_start = log_x_start[:, :, self.obs_length:]
        #if self.pretrain: log_true_prob[:, :, self.obs_length:] = 0
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        if self.focal:
            #import pdb;pdb.set_trace()
            weight_pt = torch.gather(F.softmax(log_x0_recon, dim=1), dim=1, index=x_start.long().unsqueeze(1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
            weight_pt = weight_pt.squeeze(1)
            #weight_pt = F.softmax(weight_pt, dim=1)
        else:
            weight_pt = 0
        #print(f"Min:{weight_pt.min()},Max:{weight_pt.max()},{weight_pt.shape}")
        #kl = self.multinomial_kl(log_true_prob[:,:,:-self.horizon*self.action_dim], log_model_prob[:,:, :-self.horizon*self.action_dim]) #TODO
        # mask_region = (xt == self.num_classes-1).float()
        # mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        # kl = kl * mask_weight
        log_loss_s = sum_except_batch(kl[:,:self.obs_length])
        log_loss_a_1 = sum_except_batch(kl[:,self.obs_length:self.obs_length+4])
        log_loss_a_2 = sum_except_batch(kl[:,self.obs_length+4:])
        if self.focal:
            kl = (1-weight_pt)**self.gamma*kl
        kl = sum_except_batch(kl)
        #import pdb;pdb.set_trace()
        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        if self.focal:
            #print("True")
            devoder_nll = (1-weight_pt)**self.gamma*decoder_nll
        mask = (t == torch.zeros_like(t)).float()
        log_loss_s = (mask * sum_except_batch(decoder_nll[:,:self.obs_length]) + (1. - mask) * log_loss_s) / pt
        log_loss_a_1 = (mask * sum_except_batch(decoder_nll[:,self.obs_length:self.obs_length+4]) + (1. -mask) * log_loss_a_1) / pt
        log_loss_a_2 = (mask * sum_except_batch(decoder_nll[:,self.obs_length+4:]) + (1. -mask) * log_loss_a_2) / pt
        #decoder_nll = -log_categorical(log_x_start[:,:,:-self.horizon*self.action_dim], log_model_prob[:,:,:-self.horizon*self.action_dim]) #TODO
        decoder_nll_s = sum_except_batch(decoder_nll[:,:self.obs_length])
        decoder_nll_a_1 = sum_except_batch(decoder_nll[:,self.obs_length:self.obs_length+4])
        decoder_nll_a_2 = sum_except_batch(decoder_nll[:,self.obs_length+4:])
        decoder_nll = sum_except_batch(decoder_nll)

        kl_loss = mask * decoder_nll + (1. - mask) * kl
        

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt 
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            #kl_aux = self.multinomial_kl(log_x_start[:,:self.obs_classes,:-self.horizon*self.action_dim], log_x0_recon[:,:self.obs_classes,:-self.horizon*self.action_dim]) #TODO
            #kl_aux = kl_aux * mask_weight
            #a_s = sum_except_batch(kl_aux[:,:self.obs_length])
            #a_a = sum_except_batch(kl_aux[:,self.obs_length:])
            aux_s = sum_except_batch(kl_aux[:,:self.obs_length])
            aux_a_1 = sum_except_batch(kl_aux[:,self.obs_length:self.obs_length+4])
            aux_a_2 = sum_except_batch(kl_aux[:,self.obs_length+4:])
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            kl_aux_loss_s = mask * decoder_nll_s + (1. - mask) * aux_s
            kl_aux_loss_a_1 = mask * decoder_nll_a_1 + (1. - mask) * aux_a_1
            kl_aux_loss_a_2 = mask * decoder_nll_a_2 + (1. - mask) * aux_a_2
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1-t/self.n_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            loss2_s = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_s / pt
            loss2_a_1 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_a_1 / pt
            loss2_a_2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_a_2 / pt
            vb_loss += loss2
        final_loss_s = log_loss_s + loss2_s
        final_loss_s = final_loss_s.sum()/(x_start.size()[0] * x_start.size()[1])
        final_loss_a_1 = log_loss_a_1 + loss2_a_1
        final_loss_a_2 = log_loss_a_2 + loss2_a_2
        final_loss_a = (10*final_loss_a_1.sum()+final_loss_a_2.sum())/(x_start.size()[0] * x_start.size()[1])
        # final_loss_a = final_loss_a.sum()/(x_start.size()[0] * x_start.size()[1])
        loss = vb_loss.sum()/(x_start.size()[0] * x_start.size()[1])
        #import pdb;pdb.set_trace()
        # if self.step%50==0:
        #     print(x0_recon[:,-16:])
        # self.step+=1
        info = {'a0_loss': loss,'loss_state':final_loss_s,'loss_action':final_loss_a}
        # noise = torch.randn_like(x_start)
        # context_mask = None
        # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)index_to_log_onehot
        # x_recon, act_recon = self.model(x_noisy, t, task, value, context_mask, x_condition=cond)
        # assert noise.shape == x_recon.shape

        # if self.predict_epsilon:
        #     loss, info = self.loss_fn(x_recon, noise)
        # else:
        #     loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

        # x0_recon = log_onehot_to_index(log_x0_recon)
        # x0_real = x_start
        # xt_1_recon = log_onehot_to_index(log_model_prob)
        # xt_recon = log_onehot_to_index(log_xt)
        # for index in range(t.size()[0]):
        #     this_t = t[index].item()
        #     same_rate = (x0_recon[index] == x0_real[index]).sum().cpu()/x0_real.size()[1]  #compute acc rate
        #     self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9 #smooth acc rate and store
        #     same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu()/xt_recon.size()[1] #REVIEW
        #     self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9
        
        # #logits[:,:self.obs_classes,:-self.horizon*self.action_dim])
        # #logits_act = gumble_sample(logits[:,self.obs_classes:,-self.horizon*self.action_dim:]) +
        # # compute log_true_prob now 
        # log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        # #if self.pretrain: log_true_prob[:, :, self.obs_length:] = 0
        # kl = self.multinomial_kl(log_true_prob, log_model_prob)
        # if self.focal:
        #     #import pdb;pdb.set_trace()
        #     weight_pt = torch.gather(F.softmax(log_x0_recon, dim=1), dim=1, index=x_start.long().unsqueeze(1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        #     weight_pt = weight_pt.squeeze(1)
        #     #weight_pt = F.softmax(weight_pt, dim=1)
        # else:
        #     weight_pt = 0
        # #print(f"Min:{weight_pt.min()},Max:{weight_pt.max()},{weight_pt.shape}")
        # #kl = self.multinomial_kl(log_true_prob[:,:,:-self.horizon*self.action_dim], log_model_prob[:,:, :-self.horizon*self.action_dim]) #TODO
        # # mask_region = (xt == self.num_classes-1).float()
        # # mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        # # kl = kl * mask_weight
        # log_loss_s = sum_except_batch(kl[:,:self.obs_length])
        # log_loss_a = sum_except_batch(kl[:,self.obs_length:])
        # kl = (1-weight_pt)**self.gamma*kl
        # kl = sum_except_batch(kl)
        # #import pdb;pdb.set_trace()
        # decoder_nll = -log_categorical(log_x_start, log_model_prob)
        # devoder_nll = (1-weight_pt)**self.gamma*decoder_nll
        # mask = (t == torch.zeros_like(t)).float()
        # log_loss_s = (mask * sum_except_batch(decoder_nll[:,:self.obs_length]) + (1. - mask) * log_loss_s) / pt
        # log_loss_a = (mask * sum_except_batch(decoder_nll[:,self.obs_length:]) + (1. -mask) * log_loss_a) / pt
        # #decoder_nll = -log_categorical(log_x_start[:,:,:-self.horizon*self.action_dim], log_model_prob[:,:,:-self.horizon*self.action_dim]) #TODO
        # decoder_nll_s = sum_except_batch(decoder_nll[:,:self.obs_length])
        # decoder_nll = sum_except_batch(decoder_nll)

        # kl_loss = mask * decoder_nll + (1. - mask) * kl
        

        # Lt2 = kl_loss.pow(2)
        # Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        # new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        # self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        # self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # # Upweigh loss term of the kl
        # # vb_loss = kl_loss / pt + kl_prior
        # loss1 = kl_loss / pt 
        # vb_loss = loss1
        # if self.auxiliary_loss_weight != 0:
        #     kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
        #     #kl_aux = self.multinomial_kl(log_x_start[:,:self.obs_classes,:-self.horizon*self.action_dim], log_x0_recon[:,:self.obs_classes,:-self.horizon*self.action_dim]) #TODO
        #     #kl_aux = kl_aux * mask_weight
        #     #a_s = sum_except_batch(kl_aux[:,:self.obs_length])
        #     #a_a = sum_except_batch(kl_aux[:,self.obs_length:])
        #     aux_s = sum_except_batch(kl_aux[:,:self.obs_length])
        #     kl_aux = sum_except_batch(kl_aux)
        #     kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
        #     kl_aux_loss_s = mask * decoder_nll_s + (1. - mask) * aux_s
        #     if self.adaptive_auxiliary_loss == True:
        #         addition_loss_weight = (1-t/self.n_timesteps) + 1.0
        #     else:
        #         addition_loss_weight = 1.0

        #     loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
        #     loss2_s = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_s / pt
        #     vb_loss += loss2
        # final_loss_s = log_loss_s + loss2_s
        # final_loss_s = final_loss_s.sum()/(x_start.size()[0] * x_start.size()[1])
        # loss = vb_loss.sum()/(x_start.size()[0] * x_start.size()[1])
        # info = {'a0_loss': loss,'loss_state':log_loss_s.sum()/(x_start.size()[0] * x_start.size()[1]),'loss_action':log_loss_a.sum()/(x_start.size()[0] * x_start.size()[1])}
        # # noise = torch.randn_like(x_start)
        # # context_mask = None
        # # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)index_to_log_onehot
        # # x_recon, act_recon = self.model(x_noisy, t, task, value, context_mask, x_condition=cond)
        # # assert noise.shape == x_recon.shape

        # # if self.predict_epsilon:
        # #     loss, info = self.loss_fn(x_recon, noise)
        # # else:
        # #     loss, info = self.loss_fn(x_recon, x_start)

        # return loss, info
    def loss(self, x, *args):
        if self.pretrain:
            #x_obs = self.vqvae.codebook.dictionary_lookup(x['obs'])
            #x_obs = einops.rearrange(x['obs'], 'i (a d) h k b c-> (i a) (d h k) b c', a=2,d=4)
            x_obs = einops.rearrange(x['obs'], 'i j h k b c-> (i j) (h k) b c')
            #print("MAX_MIN:",x_obs.max(),x_obs.min())
            x_act = torch.zeros((x_obs.shape[0], self.horizon, self.action_dim)).to(x['obs'].device) + self.obs_classes
            x = combine(x_obs, x_act)
            views = None
            #print("MAX_MIN:",x_obs.max(),x_obs.min())
        else:
            x_act = x['act'] + self.obs_classes
            #x_act = x_act.repeat(4, 1, 1)
            #views = x['views']
            #print(x_act)
            #x_obs = x['obs'][:,0:1]
            view_condition={
                # 'left':combine(einops.rearrange(x['left'], 'i j h k b c-> (i j) (h k) b c'), x_act),
                # 'right':combine(einops.rearrange(x['right'], 'i j h k b c-> (i j) (h k) b c'),x_act),
                # 'wrist':combine(einops.rearrange(x['wrist'], 'i j h k b c-> (i j) (h k) b c'),x_act),
                'left_':einops.rearrange(x['left'], 'i j h k b c-> (i j) (h k) b c'),
                'right_':einops.rearrange(x['right'], 'i j h k b c-> (i j) (h k) b c'),
                'wrist_':einops.rearrange(x['wrist'], 'i j h k b c-> (i j) (h k) b c'),
            }
            imgs=x['imgs']
            x_obs = einops.rearrange(x['obs'], 'i j h k b c-> (i j) (h k) b c')
            x = combine(x_obs, x_act)
            
        batch_size, device = len(x), x.device
        #print(batch_size)
        t, pt = self.sample_time(batch_size, device, 'importance')
        #t, pt = t.repeat(4), pt.repeat(4)
        diffusion_loss, info = self.p_losses(x, *args, t, pt, view_condition=view_condition,imgs=imgs) #TODO
        loss = diffusion_loss
        return loss, info


    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt)
        self.vqvae.eval()
    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.n_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.n_timesteps
            return t, pt
        else:
            raise ValueError
    # def p_pred(self, log_x, cond_emb, t):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
    #     if self.parametrization == 'x0':
    #         log_x_recon = self.predict_start(log_x, cond_emb, t)
    #         log_model_pred = self.q_posterior(
    #             log_x_start=log_x_recon, log_x_t=log_x, t=t)
    #     else:
    #         raise ValueError
    #     return log_model_pred, log_x_recon
    def sample_chain_mask_cond(self, num_samples, task=None, x_condition=None):
        b = num_samples
        device = self.log_at.device

        self.shape = (b,self.horizon*24*24+self.horizon*self.action_dim)
        zero_logits = torch.zeros((b, self.num_classes-1, self.obs_length),device=device)
        one_logits = torch.ones((b, 1, self.obs_length),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_obs = torch.log(mask_logits)
        cond = torch.zeros((b, self.horizon*self.action_dim), device=device).long()
        cond_prob = index_to_log_onehot(cond+self.obs_classes, self.num_classes)

        cond_prob = self.log_sample_categorical_sample(cond_prob) 
        log_obs = self.log_sample_categorical_sample(log_obs)

        zs = torch.zeros((self.n_timesteps, b) + (self.obs_length,), device=device).long()
        x_condition = einops.rearrange(x_condition, 'i j h b k c -> (i j) (h b) k c')
        task = einops.rearrange(task, 'i j d-> (i j) d') #task.reshape(-1, self.num_tasks)
        log_z = log_obs
        sample_type="top0.86r"
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])

        for i in reversed(range(0, self.n_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, task=task, x_condition=x_condition, cond='txt', cond_prob=cond_prob)

            zs[i] = log_onehot_to_index(log_z)
            
        print()
        #import pdb;pdb.set_trace()
        return einops.rearrange(zs[0], 'B (T H W) -> B T H W', T=4, H=24, W=24)
    @torch.no_grad()
    def sample_mask(self, num_samples, task=None, x_condition=None, f=None):
        b = num_samples
        device = self.log_at.device
        #device = 'cpu'
        #self.model = self.model.to(device)
        self.shape = (b,self.horizon*24*24+self.horizon*self.action_dim)
        zero_logits = torch.zeros((b, self.num_classes-1, self.shape[1]),device=device)
        #zero_logits = zero_logits[:,]
        one_logits = torch.ones((b, 1, self.shape[1]),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        #self.load_vqvae()
        # zs = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)

        #log_z = self.log_sample_categorical(log_z)
        x_condition = einops.rearrange(x_condition, 'i j h b k c -> (i j) (h b) k c')
        task = einops.rearrange(task, 'i j d-> (i j) d') #task.reshape(-1, self.num_tasks)
        sample_type="top0.88r"
        # sample_wrap = self.p_sample_with_truncation(self.p_sample,sample_type.split(',')[1])
        self.predict_start_ = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        from tqdm import tqdm
        for i in tqdm(reversed(range(0, self.n_timesteps)),desc="Chain timestep ",total=self.n_timesteps):
            # print(f'\nChain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, task=task, x_condition=x_condition)

            # zs[i] = log_onehot_to_index(log_z)

        zs=log_onehot_to_index(log_z)
        obs, act = split(zs, self.action_dim)
        return obs, act
    def predict_start_with_truncation(self, func, sample_type):

        truncation_r = float(sample_type[:-1].replace('top', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            # notice for different batches, out are same, we do it on out[0]
            temp, indices = torch.sort(out, 1, descending=True) 
            temp1 = torch.exp(temp)
            temp2 = temp1.cumsum(dim=1)
            temp3 = temp2 < truncation_r
            new_temp = torch.full_like(temp3[:,0:1,:], True)
            temp6 = torch.cat((new_temp, temp3), dim=1)
            temp3 = temp6[:,:-1,:]
            temp4 = temp3.gather(1, indices.argsort(1))
            temp5 = temp4.float()*out+(1-temp4.float())*(-70)
            probs = temp5
            return probs
        return wrapper
    @torch.no_grad()
    def p_sample(self, log_x, t, delta=None, cond=None, cond_prob=None, task=None, x_condition=None):
        model_log_prob = self.p_pred(log_x=log_x, t=t, delta=delta, cond=cond, cond_prob=cond_prob, task=task, x_condition=x_condition)
        out = self.log_sample_categorical(model_log_prob)
        return out
    def p_pred(self, log_x, t, delta=None, cond=None, cond_prob=None, task=None, x_condition=None):
        if self.parametrization == 'x0': 
            log_x_recon = self.predict_start_(log_x, t=t, task=task, x_condition=x_condition, value=None)
            if delta is None:
                delta = 0
            log_x0_recon = log_x_recon
            #import pdb; pdb.set_trace()
            if t[0].item() >= delta:
                log_model_pred = self.q_posterior(
                    log_x_start=log_x0_recon, log_x_t=log_x, t=t-delta, cond=cond)
            else:
                log_model_pred = self.q_posterior(
                    log_x_start=log_x0_recon, log_x_t=log_x, t=t, cond=cond)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t)
        else:
            raise ValueError
        return log_model_pred
    def cf_predict_start(self, log_x, t, task=None, x_condition=None, value=None):
        guidance_scale = 1.2
        log_x_recon = self.predict_start(log_x, t=t, task=task, x_condition=x_condition, value=None, force=True)[:, :-1]
        cf_log_x_recon = self.predict_start(log_x, t=t, task=task, x_condition=x_condition, value=None, clf=True, force=True)[:, :-1]
        log_new_x_recon = cf_log_x_recon + guidance_scale * (log_x_recon - cf_log_x_recon)
        log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
        log_new_x_recon = log_new_x_recon.clamp(-70, 0)
        log_pred = torch.cat((log_new_x_recon, self.zero_vector), dim=1)
        return log_pred

"""
nohup python -u scripts/train_diffusion.py --model models.TasksDT --diffusion models.GaussianDTInvDiffusion --loss_type statehuber --loader datasets.RTGDataset --device cuda:2 >test_DT.log 2>&1 &
"""
def alpha_schedule_1(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct
    bt = (1-at-ct)/N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    return at, bt, ct, att, btt, ctt
class VQVideoDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, beta_schedule='vp',
        action_weight=1.0, loss_discount=1.0, loss_weights=None, drop_prob=0.25, obs_classes=2048, act_classes=256,
        auxiliary_loss_weight=5e-4, adaptive_auxiliary_loss=True, mask_weight=[1,1], pretrain=True,
        patch_size=2, hidden_dim=256, parametrization= 'x0', focal=False,force=False,
        alpha_init_type='alpha1', learnable_cf=False, vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
    ):
        super().__init__()
        self.drop_prob = torch.tensor(drop_prob)
        print(self.drop_prob)
        self.horizon = 4*horizon
        self.action_dim = 4
        self.parametrization = parametrization
        #self.parametrization = 'direct'
        self.hidden_size = hidden_dim
        self.model = model
        self.pretrain = pretrain
        self.n_timesteps = int(n_timesteps)
        self.obs_classes = obs_classes
        self.act_classes = act_classes
        if act_classes==48:
            self.num_classes = act_classes +1
        else:
            self.num_classes = self.obs_classes + 1
        #self.num_classes = self.act_classes + 1
        self.obs_length = 1*24*24
        self.focal = focal
        self.gamma = 5
        ###SECTION : discrete diffusion model ###
        self.amp = False
        self.loss_type = 'vb_stochastic'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight
        if alpha_init_type == "alpha1":
            at, bt, ct, att, btt, ctt = alpha_schedule_1(self.n_timesteps, N=self.num_classes-1)
        else:
            print("alpha_init_type is Wrong !! ")
        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)
        
        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5 
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.n_timesteps
        self.diffusion_keep_list = [0] * self.n_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.n_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.n_timesteps))
        self.zero_vector = None
    
    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl
    def q_pred_one_timestep(self, log_x_t, t):
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs
    def q_posterior(self, log_x_start, log_x_t, t, cond=None):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.n_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) 
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.shape[1])

        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector
        

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)
    def q_pred_cond(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.n_timesteps + 1))%(self.n_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt_obs = extract(self.log_cumprod_bt_obs, t, log_x_start.shape)         # bt~
        log_cumprod_bt_act = extract(self.log_cumprod_bt_act, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        #NOTE：what does this mean
        log_probs = torch.cat(
                [
                    log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt_act),
   
                    log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 

                ],
                dim=1
            )

        return log_probs
    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.n_timesteps + 1))%(self.n_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs
    def gumbel2index(self, logits):
        logit_obs = logits[:,:self.obs_classes,:-self.horizon*self.action_dim].argmax(dim=1)
        logit_act = logits[:,self.obs_classes:,-self.horizon*self.action_dim:].argmax(dim=1) + self.obs_classes
        return torch.concat([logit_obs,logit_act],dim=-1)
    #REVIEW: Important to note ###
    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        #print(">>>",logits.argmax(dim=1).max(),logits.argmax(dim=1).min())
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        #sample = logits.argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample
    def log_sample_categorical_sample(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        #print(">>>",logits.argmax(dim=1).max(),logits.argmax(dim=1).min())
        #gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (logits).argmax(dim=1)
        #sample = self.gumbel2index(gumbel_noise + logits)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample
    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample
    def predict_start(self, log_x_t, t, task, value, x_condition,imgs=None):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        max_neg_value = -1e4
        #print("###",x_t.max(),x_t.min())
        if self.amp == True:
            with autocast():
                traj = self.model(x_t, t, task, value, x_condition=x_condition,imgs=imgs)
                #out = combine(traj, action, onehot=True)
        else:
            traj = self.model(x_t, t, task, value, x_condition=x_condition,imgs=imgs)
            #out = combine(traj, action, onehot=True)
        #NOTE : out is a one-hot prediction
        #out = einops.rearrange(traj, 'B T H W c-> B c (T H W)')
        out = einops.rearrange(traj, 'B X c-> B c X')
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        #print(out.size())
        #assert out.size()[2:] == x_t.size()[1:]n
        
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = torch.zeros(batch_size, 1, self.shape[1]).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred
    
    def p_losses(self, x_start, imgs, cond, task, value, t, pt):
        #x_start = torch.cat(x_start)
        #value = None
        #for i in range(10):
        #    cond[i] = einops.rearrange(cond[i], 'i j k -> (i j) k')
        self.shape = x_start.shape
        cond = einops.rearrange(cond, 'i j h b k c -> (i j) (h b) k c')
        #cond = einops.rearrange(cond, 'i j h w c -> (i j) h w c')
        task = einops.rearrange(task, 'i j d-> (i j) d') #task.reshape(-1, self.num_tasks)
        log_x_start = index_to_log_onehot(x_start.long(), self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)
        ############### go to p_theta function ###############
        
        log_x0_recon = self.predict_start(log_xt, t, task, value, x_condition=cond, imgs=imgs)            # P_theta(x0|xt)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)
        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu()/x0_real.size()[1]  #compute acc rate 
            self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9 #smooth acc rate and store
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu()/xt_recon.size()[1] #REVIEW
            self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9

        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        #mask_region = (xt == self.num_classes-1).float()
        #mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        # kl = kl * mask_weight
        if self.focal:
            weight_pt = torch.gather(F.softmax(log_x0_recon, dim=1), dim=1, index=x_start.long().unsqueeze(1))
            #weight_pt = torch.gather(log_x0_recon, dim=1, index=x_start.long().unsqueeze(1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
            weight_pt = weight_pt.squeeze(1)
            weight_pt = F.softmax(weight_pt, dim=1)
        else:
            weight_pt = 0
        kl = (1-weight_pt)**self.gamma*kl
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        devoder_nll = (1-weight_pt)**self.gamma*decoder_nll
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl
        

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt 
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            #kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1-t/self.n_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2
        loss = vb_loss.sum()/(x_start.size()[0] * x_start.size()[1])
        info = {'a0_loss': loss}
        # noise = torch.randn_like(x_start)
        # context_mask = None
        # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)index_to_log_onehot
        # x_recon, act_recon = self.model(x_noisy, t, task, value, context_mask, x_condition=cond)
        # assert noise.shape == x_recon.shape

        # if self.predict_epsilon:
        #     loss, info = self.loss_fn(x_recon, noise)
        # else:
        #     loss, info = self.loss_fn(x_recon, x_start)

        return loss, info
    def loss(self, x, *args):
        if self.pretrain:
            #x_obs = self.vqvae.codebook.dictionary_lookup(x['obs'])
            #x_obs = x['act'] + self.obs_classes#einops.rearrange(x['obs'], 'i j h k b c-> (i j) (h k) b c')
            x_obs = einops.rearrange(x['obs'], 'i j h k b c-> (i j) (h k) b c')
            #print("MAX_MIN:",x_obs.max(),x_obs.min())
            imgs = None
            #x = einops.rearrange(x_obs, 'B L D-> B (L D)')#'B T H W-> B (T H W)')
            x = einops.rearrange(x_obs,'B T H W-> B (T H W)')
            #print("MAX_MIN:",x_obs.max(),x_obs.min())
        else:
            x_obs = x#['act'] #+ self.obs_classes#einops.rearrange(x['obs'], 'i j h k b c-> (i j) (h k) b c')
            #print("MAX_MIN:",x_obs.max(),x_obs.min())
            imgs = None
            x = einops.rearrange(x_obs, 'i j L D-> (i j) (L D)')#'B T H W-> B (T H W)')
            #x = combine(x['obs'], x['act'])
        batch_size, device = len(x), x.device
        t, pt = self.sample_time(batch_size, device, 'importance')
        diffusion_loss, info = self.p_losses(x, imgs,  *args, t, pt) #TODO
        loss = diffusion_loss
        return loss, info

    
    
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt)
        self.vqvae.eval()
    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.n_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.n_timesteps
            return t, pt
        else:
            raise ValueError

    @torch.no_grad()
    def sample_mask(self, num_samples, task=None, x_condition=None):
        b = num_samples
        device = self.log_at.device
        if self.num_classes==2048+1:
            self.shape = (b,24*24)
        else:
            self.shape = (b,self.horizon*self.action_dim)
        #pdb.set_trace()
        zero_logits = torch.zeros((b, self.num_classes-1, self.shape[1]),device=device)
        one_logits = torch.ones((b, 1, self.shape[1]),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        #self.load_vqvae()
        # zs = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)
        x_condition = einops.rearrange(x_condition, 'i j h w c -> (i j) c h w') #TODO
        task = einops.rearrange(task, 'i j d-> (i j) d') #task.reshape(-1, self.num_tasks)
        sample_type="top0.88r"
        # sample_wrap = self.p_sample_with_truncation(self.p_sample,sample_type.split(',')[1])
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        from tqdm import tqdm
        for i in reversed(range(0, self.n_timesteps)):
            # print(f'\nChain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, task=task, x_condition=x_condition)
            #pdb.set_trace()
            # zs[i] = log_onehot_to_index(log_z)

        zs=log_onehot_to_index(log_z)
        return einops.rearrange(zs, 'B (t d) -> B t d', t=4, d=self.action_dim) if self.num_classes!=2048+1 else einops.rearrange(zs, 'B (T H W) -> B T H W', T=1, H=24, W=24)
    def predict_start_with_truncation(self, func, sample_type):

        truncation_r = float(sample_type[:-1].replace('top', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            # notice for different batches, out are same, we do it on out[0]
            temp, indices = torch.sort(out, 1, descending=True) 
            temp1 = torch.exp(temp)
            temp2 = temp1.cumsum(dim=1)
            temp3 = temp2 < truncation_r
            new_temp = torch.full_like(temp3[:,0:1,:], True)
            temp6 = torch.cat((new_temp, temp3), dim=1)
            temp3 = temp6[:,:-1,:]
            temp4 = temp3.gather(1, indices.argsort(1))
            temp5 = temp4.float()*out+(1-temp4.float())*(-70)
            probs = temp5
            return probs
        return wrapper
    @torch.no_grad()
    def p_sample(self, log_x, t, delta=None, cond=None, cond_prob=None, task=None, x_condition=None):
        model_log_prob, log_x_recon = self.p_pred(log_x=log_x, t=t, delta=delta, cond=cond, cond_prob=cond_prob, task=task, x_condition=x_condition)
        out = self.log_sample_categorical(model_log_prob)
        return out
    def p_pred(self, log_x, t, delta=None, cond=None, cond_prob=None, task=None, x_condition=None):
        if self.parametrization == 'x0': 
            log_x_recon = self.predict_start(log_x, t=t, task=task, x_condition=x_condition, value=None)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t, cond=cond)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t, task=task, x_condition=x_condition, value=None)
            log_x_recon = log_model_pred
        else:
            raise ValueError
        return log_model_pred, log_x_recon
class sodaDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, beta_schedule='vp',
        action_weight=1.0, loss_discount=1.0, loss_weights=None, drop_prob=0.25, obs_classes=2048, act_classes=256,
        auxiliary_loss_weight=5e-4, adaptive_auxiliary_loss=True, mask_weight=[1,1], pretrain=True,
        patch_size=2, hidden_dim=256, parametrization= 'x0', focal=False,force=False,
        alpha_init_type='alpha1', learnable_cf=False, vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
    ):
        super().__init__()
        self.drop_prob = torch.tensor(drop_prob)
        print(self.drop_prob)
        self.horizon = 4*horizon
        self.action_dim = 4
        self.parametrization = parametrization
        #self.parametrization = 'direct'
        self.hidden_size = hidden_dim
        self.model = model
        self.pretrain = pretrain
        self.n_timesteps = int(n_timesteps)
        self.obs_classes = obs_classes
        self.act_classes = act_classes
        if act_classes==48:
            self.num_classes = act_classes +1
        else:
            self.num_classes = self.obs_classes + 1
        #self.num_classes = self.act_classes + 1
        self.obs_length = 1*24*24
        self.focal = focal
        self.gamma = 5
        ###SECTION : discrete diffusion model ###
        self.amp = False
        self.loss_type = 'vb_stochastic'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight
        if alpha_init_type == "alpha1":
            at, bt, ct, att, btt, ctt = alpha_schedule_1(self.n_timesteps, N=self.num_classes-1)
        else:
            print("alpha_init_type is Wrong !! ")
        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)
        
        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5 
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.n_timesteps
        self.diffusion_keep_list = [0] * self.n_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.n_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.n_timesteps))
        self.zero_vector = None
    
    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl
    def q_pred_one_timestep(self, log_x_t, t):
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs
    def q_posterior(self, log_x_start, log_x_t, t, cond=None):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.n_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) 
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.shape[1])

        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector
        

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)
    def q_pred_cond(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.n_timesteps + 1))%(self.n_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt_obs = extract(self.log_cumprod_bt_obs, t, log_x_start.shape)         # bt~
        log_cumprod_bt_act = extract(self.log_cumprod_bt_act, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        #NOTE：what does this mean
        log_probs = torch.cat(
                [
                    log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt_act),
   
                    log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 

                ],
                dim=1
            )

        return log_probs
    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.n_timesteps + 1))%(self.n_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs
    def gumbel2index(self, logits):
        logit_obs = logits[:,:self.obs_classes,:-self.horizon*self.action_dim].argmax(dim=1)
        logit_act = logits[:,self.obs_classes:,-self.horizon*self.action_dim:].argmax(dim=1) + self.obs_classes
        return torch.concat([logit_obs,logit_act],dim=-1)
    #REVIEW: Important to note ###
    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        #print(">>>",logits.argmax(dim=1).max(),logits.argmax(dim=1).min())
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        #sample = logits.argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample
    def log_sample_categorical_sample(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        #print(">>>",logits.argmax(dim=1).max(),logits.argmax(dim=1).min())
        #gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (logits).argmax(dim=1)
        #sample = self.gumbel2index(gumbel_noise + logits)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample
    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample
    def predict_start(self, log_x_t, t, task, value, x_condition,imgs=None):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        max_neg_value = -1e4
        #print("###",x_t.max(),x_t.min())
        if self.amp == True:
            with autocast():
                traj = self.model(x_t, t, task, value, x_condition=x_condition,imgs=imgs)
                #out = combine(traj, action, onehot=True)
        else:
            traj = self.model(x_t, t, task, value, x_condition=x_condition,imgs=imgs)
            #out = combine(traj, action, onehot=True)
        #NOTE : out is a one-hot prediction
        #out = einops.rearrange(traj, 'B T H W c-> B c (T H W)')
        out = einops.rearrange(traj, 'B X c-> B c X')
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        #print(out.size())
        #assert out.size()[2:] == x_t.size()[1:]n
        
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = torch.zeros(batch_size, 1, self.shape[1]).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred
    
    def p_losses(self, x_start, imgs, cond, task, value, t, pt):
        #x_start = torch.cat(x_start)
        #value = None
        #for i in range(10):
        #    cond[i] = einops.rearrange(cond[i], 'i j k -> (i j) k')
        self.shape = x_start.shape
        cond = einops.rearrange(cond, 'i j h b k c -> (i j) (h b) k c')
        #cond = einops.rearrange(cond, 'i j h w c -> (i j) h w c')
        task = einops.rearrange(task, 'i j d-> (i j) d') #task.reshape(-1, self.num_tasks)
        log_x_start = index_to_log_onehot(x_start.long(), self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)
        ############### go to p_theta function ###############
        
        log_x0_recon = self.predict_start(log_xt, t, task, value, x_condition=cond, imgs=imgs)            # P_theta(x0|xt)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0)
        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu()/x0_real.size()[1]  #compute acc rate 
            self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9 #smooth acc rate and store
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu()/xt_recon.size()[1] #REVIEW
            self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9

        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        #mask_region = (xt == self.num_classes-1).float()
        #mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]
        # kl = kl * mask_weight
        if self.focal:
            weight_pt = torch.gather(F.softmax(log_x0_recon, dim=1), dim=1, index=x_start.long().unsqueeze(1))
            #weight_pt = torch.gather(log_x0_recon, dim=1, index=x_start.long().unsqueeze(1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
            weight_pt = weight_pt.squeeze(1)
            weight_pt = F.softmax(weight_pt, dim=1)
        else:
            weight_pt = 0
        kl = (1-weight_pt)**self.gamma*kl
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        devoder_nll = (1-weight_pt)**self.gamma*decoder_nll
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl
        

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt 
        vb_loss = loss1
        if self.auxiliary_loss_weight != 0:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            #kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1-t/self.n_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss += loss2
        loss = vb_loss.sum()/(x_start.size()[0] * x_start.size()[1])
        info = {'a0_loss': loss}
        # noise = torch.randn_like(x_start)
        # context_mask = None
        # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)index_to_log_onehot
        # x_recon, act_recon = self.model(x_noisy, t, task, value, context_mask, x_condition=cond)
        # assert noise.shape == x_recon.shape

        # if self.predict_epsilon:
        #     loss, info = self.loss_fn(x_recon, noise)
        # else:
        #     loss, info = self.loss_fn(x_recon, x_start)

        return loss, info
    def loss(self, x, *args):
        if self.pretrain:
            #x_obs = self.vqvae.codebook.dictionary_lookup(x['obs'])
            #x_obs = x['act'] + self.obs_classes#einops.rearrange(x['obs'], 'i j h k b c-> (i j) (h k) b c')
            x_obs = einops.rearrange(x['obs'], 'i j h k b c-> (i j) (h k) b c')
            #print("MAX_MIN:",x_obs.max(),x_obs.min())
            imgs = None
            #x = einops.rearrange(x_obs, 'B L D-> B (L D)')#'B T H W-> B (T H W)')
            x = einops.rearrange(x_obs,'B T H W-> B (T H W)')
            #print("MAX_MIN:",x_obs.max(),x_obs.min())
        else:
            x_obs = x['act'] #+ self.obs_classes#einops.rearrange(x['obs'], 'i j h k b c-> (i j) (h k) b c')
            #print("MAX_MIN:",x_obs.max(),x_obs.min())
            imgs = None
            x = einops.rearrange(x_obs, 'i j L D-> (i j) (L D)')#'B T H W-> B (T H W)')
            #x = combine(x['obs'], x['act'])
        batch_size, device = len(x), x.device
        t, pt = self.sample_time(batch_size, device, 'importance')
        diffusion_loss, info = self.p_losses(x, imgs,  *args, t, pt) #TODO
        loss = diffusion_loss
        return loss, info

    
    
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt)
        self.vqvae.eval()
    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.n_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.n_timesteps
            return t, pt
        else:
            raise ValueError

    @torch.no_grad()
    def sample_mask(self, num_samples, task=None, x_condition=None):
        b = num_samples
        device = self.log_at.device
        if self.num_classes==2048+1:
            self.shape = (b,24*24)
        else:
            self.shape = (b,self.horizon*self.action_dim)
        #pdb.set_trace()
        zero_logits = torch.zeros((b, self.num_classes-1, self.shape[1]),device=device)
        one_logits = torch.ones((b, 1, self.shape[1]),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        #self.load_vqvae()
        # zs = torch.zeros((self.num_timesteps, b) + (self._denoise_fn.transformer.image_length+self._denoise_fn.transformer.text_length,)).long().to(device)
        x_condition = einops.rearrange(x_condition, 'i j h w c -> (i j) h w c')
        task = einops.rearrange(task, 'i j d-> (i j) d') #task.reshape(-1, self.num_tasks)
        sample_type="top0.88r"
        # sample_wrap = self.p_sample_with_truncation(self.p_sample,sample_type.split(',')[1])
        self.predict_start = self.predict_start_with_truncation(self.predict_start, sample_type.split(',')[0])
        from tqdm import tqdm
        for i in reversed(range(0, self.n_timesteps)):
            # print(f'\nChain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t, task=task, x_condition=x_condition)
            #pdb.set_trace()
            # zs[i] = log_onehot_to_index(log_z)

        zs=log_onehot_to_index(log_z)
        return einops.rearrange(zs, 'B (t d) -> B t d', t=4, d=self.action_dim) if self.num_classes!=2048+1 else einops.rearrange(zs, 'B (T H W) -> B T H W', T=1, H=24, W=24)
    def predict_start_with_truncation(self, func, sample_type):

        truncation_r = float(sample_type[:-1].replace('top', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            # notice for different batches, out are same, we do it on out[0]
            temp, indices = torch.sort(out, 1, descending=True) 
            temp1 = torch.exp(temp)
            temp2 = temp1.cumsum(dim=1)
            temp3 = temp2 < truncation_r
            new_temp = torch.full_like(temp3[:,0:1,:], True)
            temp6 = torch.cat((new_temp, temp3), dim=1)
            temp3 = temp6[:,:-1,:]
            temp4 = temp3.gather(1, indices.argsort(1))
            temp5 = temp4.float()*out+(1-temp4.float())*(-70)
            probs = temp5
            return probs
        return wrapper
    @torch.no_grad()
    def p_sample(self, log_x, t, delta=None, cond=None, cond_prob=None, task=None, x_condition=None):
        model_log_prob, log_x_recon = self.p_pred(log_x=log_x, t=t, delta=delta, cond=cond, cond_prob=cond_prob, task=task, x_condition=x_condition)
        out = self.log_sample_categorical(model_log_prob)
        return out
    def p_pred(self, log_x, t, delta=None, cond=None, cond_prob=None, task=None, x_condition=None):
        if self.parametrization == 'x0': 
            log_x_recon = self.predict_start(log_x, t=t, task=task, x_condition=x_condition, value=None)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t, cond=cond)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t, task=task, x_condition=x_condition, value=None)
            log_x_recon = log_model_pred
        else:
            raise ValueError
        return log_model_pred, log_x_recon

Sample = namedtuple('Sample', 'trajectories values chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, task, value,  context_mask, t, **sample_kwargs):
    b, *_, device = *x.shape, x.device
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, task=task, value=value, context_mask=context_mask, t=t)
    #model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = 0.5*torch.randn_like(x)
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))) #TODO:add nonzero_mask
    #noise = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    #noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t
class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=True, predict_epsilon=True, beta_schedule='vp',
        action_weight=1.0, loss_discount=1.0, loss_weights=None, drop_prob=0.25,
    ):
        super().__init__()
        self.drop_prob = torch.tensor(drop_prob)
        print(self.drop_prob)
        self.transition_dim = action_dim
        

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        # self.transition_dim = observation_dim + 1# + action_dim# + 1#TODO
        self.model = model
        self.guide_s = 1.2
        self.act_rew_dim = self.action_dim# + 1#TODO
        """add beta schedule"""
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        #betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.action_weight = action_weight
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses[loss_type](loss_weights)
        ## get loss coefficients and initialize objective
        #print(loss_weights)
        ##loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        ##self.loss_fn = Losses[loss_type](loss_weights, self.act_rew_dim)
        #loss_weights = self.get_loss_weights(loss_discount)
        #self.loss_fn = Losses[loss_type](loss_weights)
        #self.action_weight = action_weight
        #self.inv_model = ARInvModel(hidden_dim=256, observation_dim=self.observation_dim, action_dim=self.action_dim)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        #self.action_weight = 1
        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        loss_weights[0, :] = self.action_weight
        #loss_weights[10, :] = 3

        return loss_weights
    
    
    def conditional_sample(self, cond, task, value=None, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = cond.shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, task, value, **sample_kwargs)
    
    
    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:  #via the equation between x_0 and x_t
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, task, value, context_mask, t):
        #double batch
        #x = x.repeat(2,1,1)
        #t = t.repeat(2,1,1)
        batch_size = x.shape[0]
        cond = einops.rearrange(cond, 'i j h w c -> (i j) c h w')
        task = einops.rearrange(task, 'i j d-> (i j) d').to(dtype=torch.float32)
        #noise_return_cond = self.model(x, task[batch_size:], value[:batch_size], context_mask[batch_size:], t,
        #                               force=True, return_cond=True)
        noise_task_cond = self.model(x, t, task, context_mask, x_condition=cond,
                                     force=True, return_cond=True)
        # noise_uncond = self.model(x, t, task[batch_size:], value[:batch_size], context_mask[batch_size:], x_condition=cond, force=True)
        #noise = (1+self.guide_s)*noise_cond-self.guide_s*noise_uncond
        #noise = noise_uncond + self.guide_s * (noise_task_cond-noise_uncond) #+ 1.2 * (noise_return_cond-noise_uncond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_task_cond)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, task, value, guidance, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        #print(">>>>",task.shape,value.shape)
        context_mask = torch.zeros_like(task).to(device)
        # double the batch
        batch_size = shape[0]
        #task = task.repeat(2)
        #value = value.repeat(2)
        #context_mask = context_mask.repeat(2)
        context_mask[batch_size:] = 1. # makes second half of batch context free
        x = torch.randn(shape, device=device)
        #x = apply_conditioning(x, cond, 0)

        chain = [x] if return_chain else None
        self.guide_s = guidance
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, task, value, context_mask, t, **sample_kwargs)
            #x = apply_conditioning(x, cond, 0) #TODO

            #progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        #progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

   

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, task, value, t):
        #x_start = torch.cat(x_start)
        #value = None
        #for i in range(10):
        #    cond[i] = einops.rearrange(cond[i], 'i j k -> (i j) k')
        
        cond = einops.rearrange(cond, 'i j h k c-> (i j) h k c').to(dtype=torch.float32)
        task = einops.rearrange(task, 'i j d-> (i j) d').to(dtype=torch.float32) #task.reshape(-1, self.num_tasks)
        #timestep = einops.rearrange(timestep, 'i j k -> (i j) k')
        value = None#einops.rearrange(value, 'i j 1 -> (i j) 1')
        noise = torch.randn_like(x_start)
        #task = [one_hot_dict[t] for t in task]
        #print(task.shape)
        #tmp = torch.cat((task,value),dim=-1)
        #print(tmp.shape)
        #print(value.shape)
        #flag = False
        ##if task[0]<0:
        ##    flag=True
        #print(t.shape,task.shape)
        ##context_mask = torch.bernoulli(torch.zeros_like(task)+self.drop_prob).to(x_start.device)
        context_mask = None
        #contex_mask_value = torch.bernoulli(torch.zeros_like(value)+self.drop_prob).to(x_start.device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        #x_noisy = apply_conditioning_v1(x_noisy, cond, self.action_dim) #TODO
        ##self.act_rew_dim) #only conditon on current observation and get a_t and r_t
        x_recon = self.model(x_noisy, t, task, value, context_mask, x_condition=cond)
        #x_recon = apply_conditioning(x_recon, cond, self.act_rew_dim)
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        x = einops.rearrange(x, 'i j h k -> (i j) h k')
        batch_size = len(x)
        #task = torch.zeros((batch_size),device=x.device)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffusion_loss, info = self.p_losses(x, *args, t) #TODO
        loss = diffusion_loss
        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
class DTDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=True, predict_epsilon=True, beta_schedule='vp',
        action_weight=1.0, loss_discount=1.0, loss_weights=None, drop_prob=0.25,
    ):
        super().__init__()
        self.drop_prob = torch.tensor(drop_prob)
        print(self.drop_prob)
        self.transition_dim = action_dim
        

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        # self.transition_dim = observation_dim + 1# + action_dim# + 1#TODO
        self.model = model
        self.guide_s = 1.2
        self.act_rew_dim = self.action_dim# + 1#TODO
        """add beta schedule"""
        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        #betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.action_weight = action_weight
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses[loss_type](loss_weights)
        ## get loss coefficients and initialize objective
        #print(loss_weights)
        ##loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        ##self.loss_fn = Losses[loss_type](loss_weights, self.act_rew_dim)
        #loss_weights = self.get_loss_weights(loss_discount)
        #self.loss_fn = Losses[loss_type](loss_weights)
        #self.action_weight = action_weight
        #self.inv_model = ARInvModel(hidden_dim=256, observation_dim=self.observation_dim, action_dim=self.action_dim)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        #self.action_weight = 1
        dim_weights = torch.ones(self.action_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        loss_weights[0, :] = self.action_weight
        #loss_weights[10, :] = 3

        return loss_weights
    
    
    def conditional_sample(self, cond, task, value=None, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = cond.shape[0]
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, task, value, **sample_kwargs)
    
    
    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:  #via the equation between x_0 and x_t
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, task, value, context_mask, t):
        #double batch
        #x = x.repeat(2,1,1)
        #t = t.repeat(2,1,1)
        batch_size = x.shape[0]
        cond = einops.rearrange(cond, 'i j h w c -> (i j) h w c')
        task = einops.rearrange(task, 'i j d-> (i j) d').to(dtype=torch.float32)
        #noise_return_cond = self.model(x, task[batch_size:], value[:batch_size], context_mask[batch_size:], t,
        #                               force=True, return_cond=True)
        noise_task_cond = self.model(x, t, task, context_mask, x_condition=cond,
                                     force=True, return_cond=True)
        # noise_uncond = self.model(x, t, task[batch_size:], value[:batch_size], context_mask[batch_size:], x_condition=cond, force=True)
        #noise = (1+self.guide_s)*noise_cond-self.guide_s*noise_uncond
        #noise = noise_uncond + self.guide_s * (noise_task_cond-noise_uncond) #+ 1.2 * (noise_return_cond-noise_uncond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_task_cond)
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, task, value, guidance, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        #print(">>>>",task.shape,value.shape)
        context_mask = torch.zeros_like(task).to(device)
        # double the batch
        batch_size = shape[0]
        #task = task.repeat(2)
        #value = value.repeat(2)
        #context_mask = context_mask.repeat(2)
        context_mask[batch_size:] = 1. # makes second half of batch context free
        x = torch.randn(shape, device=device)
        #x = apply_conditioning(x, cond, 0)

        chain = [x] if return_chain else None
        self.guide_s = guidance
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, task, value, context_mask, t, **sample_kwargs)
            #x = apply_conditioning(x, cond, 0) #TODO

            #progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        #progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

   

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, task, value, t):
        #x_start = torch.cat(x_start)
        #value = None
        #for i in range(10):
        #    cond[i] = einops.rearrange(cond[i], 'i j k -> (i j) k')
        
        cond = einops.rearrange(cond, 'i j h k c-> (i j) h k c').to(dtype=torch.float32)
        task = einops.rearrange(task, 'i j d-> (i j) d').to(dtype=torch.float32) #task.reshape(-1, self.num_tasks)
        #timestep = einops.rearrange(timestep, 'i j k -> (i j) k')
        value = None#einops.rearrange(value, 'i j 1 -> (i j) 1')
        noise = torch.randn_like(x_start)
        #task = [one_hot_dict[t] for t in task]
        #print(task.shape)
        #tmp = torch.cat((task,value),dim=-1)
        #print(tmp.shape)
        #print(value.shape)
        #flag = False
        ##if task[0]<0:
        ##    flag=True
        #print(t.shape,task.shape)
        ##context_mask = torch.bernoulli(torch.zeros_like(task)+self.drop_prob).to(x_start.device)
        context_mask = None
        #contex_mask_value = torch.bernoulli(torch.zeros_like(value)+self.drop_prob).to(x_start.device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        #x_noisy = apply_conditioning_v1(x_noisy, cond, self.action_dim) #TODO
        ##self.act_rew_dim) #only conditon on current observation and get a_t and r_t
        x_recon = self.model(x_noisy, t, task, value, context_mask, x_condition=cond)
        #x_recon = apply_conditioning(x_recon, cond, self.act_rew_dim)
        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        x = einops.rearrange(x, 'i j h k -> (i j) h k')
        batch_size = len(x)
        #task = torch.zeros((batch_size),device=x.device)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        diffusion_loss, info = self.p_losses(x, *args, t) #TODO
        loss = diffusion_loss
        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

def split(x, action_dim):
        #T, H, W = 4, 24, 24 #TODO
        T, H, W = 1, 24, 24
        z_dim = T * H * W
        act_dim = 4*action_dim
        z, a = x.split([z_dim, act_dim], dim=1) #TODO:action_dim
        z = einops.rearrange(z, 'B (T H W) -> B T H W', T=T, H=H, W=W)
        a = einops.rearrange(a, 'B (L D) -> B L D', L=4, D=action_dim)
        return z, a


def combine(traj, action, onehot=False):
    if onehot:
        #z = einops.rearrange(traj, 'B T H W c-> B c (T H W)')
        z = einops.rearrange(traj, 'B X c-> B c X')
        action = einops.rearrange(action, 'B L D -> B D L')
    else:
        z = einops.rearrange(traj, 'B T H W-> B (T H W)')
        action = einops.rearrange(action, 'B L D-> B (L D)')
    return torch.concat([z, action], dim=-1)