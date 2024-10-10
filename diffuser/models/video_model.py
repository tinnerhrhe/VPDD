import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pickle
import pdb
import transformers
from transformers import TransfoXLModel, TransfoXLConfig
from .GPT2 import GPT2Model
import numpy as np
import torch.nn.functional as F
from videogpt.utils import view_range, shift_dim, trunc_normal_, tensor_slice
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from .helpers import (
    SinusoidalPosEmb,
)
from torch.distributions import Bernoulli
from videogpt import VideoData, VideoGPT, load_videogpt, VQVAE
import os
import math
from .utils import *
LRELU_SLOPE = 0.02
############## Attention #######################
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class GeLU2(nn.Module):
    def forward(self, x):
        return (1.702 * x).sigmoid() * x

def split(x, act_dim):
        #T, H, W = 4, 24, 24 #TODO
        T, H, W = 1, 24, 24
        z_dim = T * H * W
        act_dim = 4*act_dim
        z, a = x.split([z_dim, act_dim], dim=1) #TODO:action_dim
        z = einops.rearrange(z, 'B (T H W) -> B T H W', T=T, H=H, W=W)
        #a = einops.rearrange(a, 'B (L D) -> B L D', L=act_dim, D=1)
        return z, a





class pretrainModel(nn.Module):
    def __init__(self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=256,
            hidden_dim=256,
            shape=(4,24,24),
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=12,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=8,
            max_ep_len=1000,
            patch_size=2,
            in_chans=4,
            act_cls=256,
            obs_cls=2048,
            vqvae=None,
            vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
            num_latents=2048,
        ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.prompt_embed = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed_history = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2*hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(2*hidden_dim, hidden_dim),
        )
        self.vqvae=vqvae.eval()
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_traj = nn.Linear(dim, hidden_dim)

        self.predict_traj = nn.Sequential(
            nn.Linear(hidden_size, self.obs_cls),
        )
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+2*12*12*8+2*12*12*8, hidden_size)) #T, language, history, pred_act, pred_traj
        """ Perceiver Transformer """
         # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

        # encoder cross attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(hidden_dim, PerAttention(hidden_dim,
                                          hidden_dim,
                                          heads=1,
                                          dim_head=32,
                                          dropout=0.1),
                    context_dim=hidden_dim),
            PreNorm(hidden_dim, FeedForward(hidden_dim))
        ])

        get_latent_attn = lambda: PreNorm(hidden_dim,
                                          PerAttention(hidden_dim, heads=1,
                                                    dim_head=32, dropout=0.1))
        get_latent_ff = lambda: PreNorm(hidden_dim, FeedForward(hidden_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # self attention layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': False}

        for i in range(depth//2):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # decoder cross attention
        self.decoder_cross_attn = PreNorm(hidden_dim, PerAttention(hidden_dim,
                                                                hidden_dim,
                                                                heads=1,
                                                                dim_head=32,
                                                                dropout=0.0),
                                          context_dim=hidden_dim)
        self.iterations = 1
    def forward(self, traj_obs, time, cond, value, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        with torch.no_grad():
            traj_obs = self.vqvae.codebook.dictionary_lookup(traj_obs.long())
            x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        traj_obs = shift_dim(traj_obs, -1, 1).flatten(2).transpose(1, 2)
        batch_size, seq_length = traj_obs.shape[0], traj_obs.shape[1]
        x_condition = shift_dim(x_condition, -1, 1).flatten(2).transpose(1, 2)
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        obs_embeddings = self.embed_history(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        traj_embeddings = self.embed_traj(traj_obs)
        stacked_inputs = torch.cat((t, prompt_embeddings, obs_embeddings, traj_embeddings), dim=1)
        stacked_inputs = prompt_embeddings * stacked_inputs
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        # batchify latents
        x = repeat(self.latents, 'n d -> b n d', b=batch_size)

        cross_attn, cross_ff = self.cross_attend_blocks

        for it in range(self.iterations):
            # encoder cross attention
            x = cross_attn(x, context=stacked_inputs, mask=mask) + x
            x = cross_ff(x) + x

            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # decoder cross attention
        latents = self.decoder_cross_attn(stacked_inputs, context=x)
        #x = self.pos_drop(x)

        predicted_token_traj = latents[:, -seq_length:, :]
        predicted_traj = self.predict_traj(predicted_token_traj)
        #import pdb;pdb.set_trace()
        return predicted_traj, predicted_token_traj
def act_layer(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU(LRELU_SLOPE)
    elif act == 'elu':
        return nn.ELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'prelu':
        return nn.PReLU()
    else:
        raise ValueError('%s not recognized.' % act)


def norm_layer2d(norm, channels):
    if norm == 'batch':
        return nn.BatchNorm2d(channels)
    elif norm == 'instance':
        return nn.InstanceNorm2d(channels, affine=True)
    elif norm == 'layer':
        return nn.GroupNorm(1, channels, affine=True)
    elif norm == 'group':
        return nn.GroupNorm(4, channels, affine=True)
    else:
        raise ValueError('%s not recognized.' % norm)

class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=3,
        strides=1,
        norm=None,
        activation=None,
        padding_mode="replicate",
        padding=None,
    ):
        super().__init__()
        padding = kernel_sizes // 2 if padding is None else padding
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_sizes,
            strides,
            padding=padding,
            padding_mode=padding_mode,
        )

       
       
        nn.init.kaiming_uniform_(
            self.conv2d.weight, a=LRELU_SLOPE, nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.conv2d.bias)
        

        self.activation = None
        if norm is not None:
            self.norm = norm_layer2d(norm, out_channels)
        else:
            self.norm = None
        if activation is not None:
            self.activation = act_layer(activation)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv2d(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x
class DenseBlock(nn.Module):

    def __init__(self, in_features, out_features, norm=None, activation=None):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

       
       
        nn.init.kaiming_uniform_(self.linear.weight, a=LRELU_SLOPE, nonlinearity='leaky_relu')
        nn.init.zeros_(self.linear.bias)
       

        self.activation = None
        self.norm = None
        if norm is not None:
            self.norm = norm_layer1d(norm, out_features)
        if activation is not None:
            self.activation = act_layer(activation)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.activation(x) if self.activation is not None else x
        return x


class finetuneModel(nn.Module):
    def __init__(self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=256,
            hidden_dim=256,
            shape=(4,24,24),
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=12,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=8,
            max_ep_len=1000,
            patch_size=2,
            in_chans=4,
            act_cls=256,
            obs_cls=2048,
            vqvae=None,
            vqvae_ckpt='./lightning_logs/version_45/checkpoints/last.ckpt',
        num_latents=2048,):
        super().__init__()
        self.action_dim =action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.prompt_embed = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.horizon = horizon
        self.vqvae=vqvae
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_act = torch.nn.Linear(hidden_size, self.act_cls)
        self.act_embeddings = nn.Embedding(self.act_cls+1, hidden_size)
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+2*12*12*8+horizon*self.action_dim+4*16*16, hidden_size)) #T, language, history, pred_act, pred_traj
        """ Perceiver Transformer """
         # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

        # encoder cross attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(hidden_dim, PerAttention(hidden_dim,
                                          hidden_dim,
                                          heads=1,
                                          dim_head=32,
                                          dropout=0.1),
                    context_dim=hidden_dim),
            PreNorm(hidden_dim, FeedForward(hidden_dim))
        ])

        get_latent_attn = lambda: PreNorm(hidden_dim,
                                          PerAttention(hidden_dim, heads=1,
                                                    dim_head=32, dropout=0.1))
        get_latent_ff = lambda: PreNorm(hidden_dim, FeedForward(hidden_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # self attention layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': False}

        for i in range(depth//2):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # decoder cross attention
        self.decoder_cross_attn = PreNorm(hidden_dim, PerAttention(hidden_dim,
                                                                hidden_dim,
                                                                heads=1,
                                                                dim_head=32,
                                                                dropout=0.0),
                                          context_dim=hidden_dim)
        self.iterations = 1
        
        '''images blocks'''
        activation = "lrelu"
        self.img_patch_size = 8
        self.im_channels = 64
        self.proprio_preprocess = DenseBlock(
               4, self.im_channels, norm=None, activation=activation,
            )
        self.input_preprocess = Conv2DBlock(
            3,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )
        inp_pre_out_dim = self.im_channels
        self.patchify = Conv2DBlock(
            inp_pre_out_dim,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )
        self.img_embed = nn.Sequential(
            nn.LayerNorm(self.im_channels*2),
            nn.Linear(self.im_channels*2, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attn_1 = CrossAttention(
                    4*16*16,
                    n_embd=dim,
                    condition_embd=dim,
                    n_head=1,
                    seq_len=4*16*16,
                    attn_pdrop=0.1,
                    resid_pdrop=0.1,
                    )
        self.cross_attn_2 = CrossAttention(
                    1,
                    n_embd=dim,
                    condition_embd=dim,
                    n_head=1,
                    seq_len=1,
                    attn_pdrop=0.1,
                    resid_pdrop=0.1,
                    )
    def forward(self, act, time, cond, proprio, predicted_traj, img, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        #img: (bs, num_img, img_feat_dim, h, w)
        x_act = self.act_embeddings(act-self.obs_cls)
        batch_size, seq_length = x_act.shape[0], x_act.shape[1]
        '''process img'''
        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        img = img.view(bs * num_img, img_feat_dim, h, w)
        d0 = self.input_preprocess(img)
        ins = self.patchify(d0)
        ins = (
            ins.view(
                bs,
                num_img,
                self.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2)
            .clone()
        )
        _, _, _d, _h, _w = ins.shape
        p = self.proprio_preprocess(proprio)              # [B,4] -> [B,64]
        p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
        ins = torch.cat([ins, p], dim=1)        
        ins = rearrange(ins, "b d ... -> b (...) d")  # [B, num_img * np * np, 64]
        image_emb = self.img_embed(ins)
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        #obs_embeddings = self.embed_history(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        stacked_inputs = torch.cat((t, prompt_embeddings, predicted_traj, image_emb, x_act), dim=1)
        #print("SHAPE:",t.shape, prompt_embeddings.shape, predicted_traj.shape, image_emb.shape, x_act.shape)
        stacked_inputs = prompt_embeddings * stacked_inputs# + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_inputs = self.cross_attn_1(stacked_inputs, image_emb, mask=mask)[0] + stacked_inputs
        stacked_inputs = self.cross_attn_2(stacked_inputs, prompt_embeddings, mask=mask)[0] + stacked_inputs
        # batchify latents
        x = repeat(self.latents, 'n d -> b n d', b=batch_size)

        cross_attn, cross_ff = self.cross_attend_blocks

        for it in range(self.iterations):
            # encoder cross attention
            x = cross_attn(x, context=stacked_inputs, mask=mask) + x
            x = cross_ff(x) + x

            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # decoder cross attention
        latents = self.decoder_cross_attn(stacked_inputs, context=x)
        #x = self.pos_drop(x)

        predicted_token_act = latents[:, -self.horizon*self.action_dim:, :]
        predicted_act = self.predict_act(predicted_token_act)
        return predicted_act
class MetaFinetuneModel(nn.Module):
    def __init__(self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=256,
            hidden_dim=256,
            shape=(4,24,24),
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=12,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=8,
            max_ep_len=1000,
            patch_size=2,
            in_chans=4,
            act_cls=256,
            obs_cls=2048,
            vqvae=None,
            vqvae_ckpt='./lightning_logs/version_45/checkpoints/last.ckpt',
        num_latents=2048,):
        super().__init__()
        num_latens=512
        self.action_dim =action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.prompt_embed = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.horizon = horizon
        self.vqvae=vqvae
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_act_x = nn.Sequential(
            nn.Linear(hidden_size, self.act_cls),
        )
        self.predict_act_y = nn.Sequential(
            nn.Linear(hidden_size, self.act_cls),
        )
        self.predict_act_z = nn.Sequential(
            nn.Linear(hidden_size, self.act_cls),
        )
        
        self.predict_gripper = torch.nn.Linear(hidden_size, self.act_cls)
        self.act_embeddings =  nn.Sequential(
            #nn.LayerNorm(self.act_cls),
            nn.Linear(1, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )#nn.Embedding(self.act_cls+1, hidden_size)
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+24*24+self.action_dim*self.action_dim, hidden_size)) #T, language, history, pred_act, pred_traj
        """ Perceiver Transformer """
         # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

        # encoder cross attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(hidden_dim, PerAttention(hidden_dim,
                                          hidden_dim,
                                          heads=1,
                                          dim_head=32,
                                          dropout=0.1),
                    context_dim=hidden_dim),
            PreNorm(hidden_dim, FeedForward(hidden_dim))
        ])

        get_latent_attn = lambda: PreNorm(hidden_dim,
                                          PerAttention(hidden_dim, heads=1,
                                                    dim_head=32, dropout=0.1))
        get_latent_ff = lambda: PreNorm(hidden_dim, FeedForward(hidden_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # self attention layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': False}

        for i in range(depth//2):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # decoder cross attention
        self.decoder_cross_attn = PreNorm(hidden_dim, PerAttention(hidden_dim,
                                                                hidden_dim,
                                                                heads=1,
                                                                dim_head=32,
                                                                dropout=0.0),
                                          context_dim=hidden_dim)
        self.iterations = 1
        
        '''images blocks'''
        activation = "lrelu"
        self.img_patch_size = 8
        self.im_channels = 64

        self.cross_attn_1 = CrossAttention(
                    1*24*24,
                    n_embd=dim,
                    condition_embd=dim,
                    n_head=1,
                    seq_len=1*24*24,
                    attn_pdrop=0.1,
                    resid_pdrop=0.1,
                    )
        self.self_attn = FullAttention(
                n_embd=dim,
                n_head=1,
                seq_len=1*24*24,
                attn_pdrop=0.1,
                resid_pdrop=0.1,
            )
    def forward(self, act, time, cond, proprio, predicted_traj, img, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        #img: (bs, num_img, img_feat_dim, h, w)
        #print(act.shape)
        x_act = self.act_embeddings((act-self.obs_cls).unsqueeze(-1).to(dtype=predicted_traj.dtype))
        batch_size, seq_length = x_act.shape[0], x_act.shape[1]
        '''process img'''

        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        t = self.time_mlp(time).unsqueeze(1)
        #stacked_inputs = x_act
        stacked_inputs = torch.cat((t, prompt_embeddings, predicted_traj, x_act), dim=1)
        #stacked_inputs = torch.cat((t, prompt_embeddings, predicted_traj, image_emb, x_act), dim=1)
        #stacked_inputs = torch.cat((t, prompt_embeddings, image_emb, x_act), dim=1)
        #print("SHAPE:",t.shape, prompt_embeddings.shape, predicted_traj.shape, image_emb.shape, x_act.shape)
        stacked_inputs = prompt_embeddings * stacked_inputs# + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = t * stacked_inputs + self.position_emb#[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_inputs = self.cross_attn_1(stacked_inputs, predicted_traj, mask=mask)[0] + stacked_inputs
        #stacked_inputs = self.self_attn(stacked_inputs)[0] + stacked_inputs
        # batchify latents
        x = repeat(self.latents, 'n d -> b n d', b=batch_size)

        cross_attn, cross_ff = self.cross_attend_blocks

        for it in range(self.iterations):
            # encoder cross attention
            x = cross_attn(x, context=stacked_inputs, mask=mask) + x
            x = cross_ff(x) + x

            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # decoder cross attention
        latents = self.decoder_cross_attn(stacked_inputs, context=x)
        #x = self.pos_drop(x)
        predicted_token_act = latents[:, -self.horizon*self.action_dim:, :].reshape(batch_size, self.horizon, self.action_dim, -1)
        #predicted_act = self.predict_act(predicted_token_act)
        predicted_trans_x = self.predict_act_x(predicted_token_act[:,:,0:1])
        predicted_trans_y = self.predict_act_y(predicted_token_act[:,:,1:2])
        predicted_trans_z = self.predict_act_z(predicted_token_act[:,:,2:3])
        
        predicted_gripper = self.predict_gripper(predicted_token_act[:,:,3:])
        predicted_act = torch.cat([predicted_trans_x, predicted_trans_y,predicted_trans_z,predicted_gripper], dim=2).reshape(batch_size, self.horizon*self.action_dim, self.act_cls)
        
        return predicted_act
class MetaFinetuneModel_v1(nn.Module):
    def __init__(self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=256,
            hidden_dim=256,
            shape=(4,24,24),
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=12,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=8,
            max_ep_len=1000,
            patch_size=2,
            in_chans=4,
            act_cls=256,
            obs_cls=2048,
            vqvae=None,
            vqvae_ckpt='./lightning_logs/version_45/checkpoints/last.ckpt',
        num_latents=2048,):
        super().__init__()
        num_latens=512
        self.action_dim =action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.prompt_embed = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.propri_embed = nn.Sequential(
            nn.LayerNorm(39),
            nn.Linear(39, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # self.process_tokens = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),  # TODO
        #     nn.Mish(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Mish(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )
        self.horizon = horizon
        self.vqvae=vqvae
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_act_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_size, self.act_cls),
            
        )
        self.predict_act_y = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_size, self.act_cls),
        )
        self.predict_act_z = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_size, self.act_cls),
        )
        
        self.predict_gripper = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_size, self.act_cls),
        )#torch.nn.Linear(hidden_size, self.act_cls)
        self.act_embeddings =  nn.Sequential(
            #nn.LayerNorm(self.act_cls),
            nn.Linear(1, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+24*24+4*self.action_dim+20*20, hidden_size)) #T, language, history, pred_act, pred_traj
        import transformers
        from .GPT2 import GPT2Model
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=6,
            n_head=2,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.transformer = GPT2Model(config)
        '''images blocks'''
        activation = "lrelu"
        self.img_patch_size = 13
        self.im_channels = 64
        # self.proprio_preprocess = DenseBlock(
        #        4, self.im_channels, norm=None, activation=activation,
        #     )
        # self.input_preprocess = Conv2DBlock(
        #     3,
        #     self.im_channels,
        #     kernel_sizes=1,
        #     strides=1,
        #     norm=None,
        #     activation=activation,
        # )
        # inp_pre_out_dim = self.im_channels
        # self.patchify = Conv2DBlock(
        #     inp_pre_out_dim,
        #     self.im_channels,
        #     kernel_sizes=self.img_patch_size,
        #     strides=self.img_patch_size,
        #     norm="group",
        #     activation=activation,
        #     padding=0,
        # )
        # self.img_embed = nn.Sequential(
        #     nn.LayerNorm(self.im_channels),
        #     nn.Linear(self.im_channels, hidden_dim),  # TODO
        #     nn.Mish(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )
        self.cross_attn_1 = CrossAttention(
                    1*24*24,
                    n_embd=dim,
                    condition_embd=dim,
                    n_head=1,
                    seq_len=1*24*24,
                    attn_pdrop=0.1,
                    resid_pdrop=0.1,
                    )
        self.self_attn = FullAttention(
                n_embd=dim,
                n_head=1,
                seq_len=1*24*24,
                attn_pdrop=0.1,
                resid_pdrop=0.1,
            )
    def forward(self, act, time, cond, proprio, predicted_traj, img, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        #img: (bs, num_img, img_feat_dim, h, w)
        #print(act.shape)
        x_act = self.act_embeddings((act-self.obs_cls).unsqueeze(-1).to(dtype=torch.float32))
        # x_act_x = self.act_embeddings_x((act-self.obs_cls).reshape(predicted_traj.shape[0], 4, 4)[:,:, 0:1].to(dtype=predicted_traj.dtype))
        # x_act_y = self.act_embeddings_y((act-self.obs_cls).reshape(predicted_traj.shape[0], 4, 4)[:,:, 1:2].to(dtype=predicted_traj.dtype))
        # x_act_z = self.act_embeddings_z((act-self.obs_cls).reshape(predicted_traj.shape[0], 4, 4)[:,:, 2:3].to(dtype=predicted_traj.dtype))
        # x_act_g = self.act_embeddings_g((act-self.obs_cls).reshape(predicted_traj.shape[0], 4, 4)[:,:, 3:4].to(dtype=predicted_traj.dtype))
        # x_act = torch.cat([x_act_x,x_act_y,x_act_z,x_act_g], dim=-2)
        batch_size, seq_length = x_act.shape[0], x_act.shape[1]
        '''process img'''

        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        t = self.time_mlp(time).unsqueeze(1)
        # predicted_traj = self.process_tokens(predicted_traj)
        #stacked_inputs = x_act
        #stacked_inputs = torch.cat((t, prompt_embeddings, image_emb, proprio, x_act), dim=1)
        #stacked_inputs = torch.cat((t, prompt_embeddings, predicted_traj, image_emb, x_act), dim=1)
        stacked_inputs = torch.cat((t, prompt_embeddings, predicted_traj, x_act), dim=1)
        #stacked_inputs = torch.cat((t, prompt_embeddings, image_emb, x_act), dim=1)
        #print("SHAPE:",t.shape, prompt_embeddings.shape, predicted_traj.shape, image_emb.shape, x_act.shape)
        stacked_inputs = prompt_embeddings * stacked_inputs# + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        #stacked_inputs = self.cross_attn_1(stacked_inputs, predicted_traj, mask=mask)[0] + stacked_inputs
        #stacked_inputs = self.self_attn(stacked_inputs)[0] + stacked_inputs
        # batchify latents
        
        stacked_attention_mask = torch.ones((stacked_inputs.shape[0], stacked_inputs.shape[1])).to(stacked_inputs.device)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        latents = transformer_outputs['last_hidden_state']
        predicted_token_act = latents[:, -self.horizon*self.action_dim:, :].reshape(batch_size, self.horizon, self.action_dim, -1)
        #predicted_act = self.predict_act(predicted_token_act)
        predicted_trans_x = self.predict_act_x(predicted_token_act[:,:,0:1])
        predicted_trans_y = self.predict_act_y(predicted_token_act[:,:,1:2])
        predicted_trans_z = self.predict_act_z(predicted_token_act[:,:,2:3])
        
        predicted_gripper = self.predict_gripper(predicted_token_act[:,:,3:])
        predicted_act = torch.cat([predicted_trans_x, predicted_trans_y,predicted_trans_z,predicted_gripper], dim=2).reshape(batch_size, self.horizon*self.action_dim, self.act_cls)
        
        return predicted_act
class MultiviewFinetuneModel(nn.Module):
    def __init__(self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=256,
            hidden_dim=256,
            shape=(4,24,24),
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=12,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=8,
            max_ep_len=1000,
            patch_size=2,
            in_chans=4,
            act_cls=256,
            obs_cls=2048,
            vqvae=None,
            vqvae_ckpt='./lightning_logs/version_45/checkpoints/last.ckpt',
        num_latents=2048,):
        super().__init__()
        self.action_dim =action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.prompt_embed = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.horizon = horizon
        self.vqvae=vqvae
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.im_channels = 256
        activation = "lrelu"
        self.proprio_preprocess = DenseBlock(
               4, self.im_channels, norm=None, activation=activation,
            )

        self.predict_trans = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size, self.act_cls),
        )
        self.predict_quat = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size, self.act_cls),
        )
        self.process_tokens = nn.Sequential(
            nn.Linear(1024, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predict_gripper = torch.nn.Linear(hidden_size, self.act_cls)
        self.act_embeddings = nn.Embedding(self.act_cls+1, hidden_size)
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+2*12*12*8+horizon*self.action_dim+4*16*16, hidden_size)) #T, language, history, pred_act, pred_traj
        import transformers
        from .GPT2 import GPT2Model
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=6,
            n_head=2,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.transformer = GPT2Model(config)
        
        '''images blocks'''
        activation = "lrelu"
        self.img_patch_size = 13
        self.im_channels = 64
        self.input_preprocess = Conv2DBlock(
            3,
            self.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=activation,
        )
        inp_pre_out_dim = self.im_channels
        self.patchify = Conv2DBlock(
            inp_pre_out_dim,
            self.im_channels,
            kernel_sizes=self.img_patch_size,
            strides=self.img_patch_size,
            norm="group",
            activation=activation,
            padding=0,
        )
        self.img_embed = nn.Sequential(
            nn.LayerNorm(self.im_channels),
            nn.Linear(self.im_channels, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # self.cross_attn_1 = CrossAttention(
        #             4*24*24,
        #             n_embd=dim,
        #             condition_embd=dim,
        #             n_head=1,
        #             seq_len=4*24*24,
        #             attn_pdrop=0.1,
        #             resid_pdrop=0.1,
        #             )
        # self.self_attn = FullAttention(
        #         n_embd=dim,
        #         n_head=1,
        #         seq_len=4*24*24,
        #         attn_pdrop=0.1,
        #         resid_pdrop=0.1,
        #     )
    def forward(self, act, time, cond, proprio, predicted_traj, img, view_condition=None, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        #img: (bs, num_img, img_feat_dim, h, w)
        x_act = self.act_embeddings(act-self.obs_cls)
        batch_size, seq_length = x_act.shape[0], x_act.shape[1]
        '''process img'''
        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.img_patch_size
        img = img.view(bs * num_img, img_feat_dim, h, w)
        d0 = self.input_preprocess(img)
        ins = self.patchify(d0)
        ins = (
            ins.view(
                bs,
                num_img,
                self.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2)
            .clone()
        )
        _, _, _d, _h, _w = ins.shape
        #ins = torch.cat([ins, p], dim=1)        
        ins = rearrange(ins, "b d ... -> b (...) d")  # [B, num_img * np * np, 64]
        image_emb = self.img_embed(ins)
        p = self.proprio_preprocess(proprio).unsqueeze(1).repeat(batch_size//proprio.shape[0], 1, 1)             # [B,4] -> [B,64]
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        #obs_embeddings = self.embed_history(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        predicted_traj = self.process_tokens(predicted_traj)
        #predicted_traj = self.process_tokens(predicted_traj)
        stacked_inputs = torch.cat((t, prompt_embeddings, p, predicted_traj, image_emb, x_act), dim=1)
        #print("SHAPE:",t.shape, prompt_embeddings.shape, predicted_traj.shape, image_emb.shape, x_act.shape)
        stacked_inputs = prompt_embeddings * stacked_inputs# + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        # stacked_inputs = self.cross_attn_1(stacked_inputs, predicted_traj, mask=mask)[0] + stacked_inputs
        # stacked_inputs = self.self_attn(stacked_inputs)[0] + stacked_inputs
        #stacked_inputs = self.cross_attn_2(stacked_inputs, prompt_embeddings, mask=mask)[0] + stacked_inputs
        # batchify latents
        # x = repeat(self.latents, 'n d -> b n d', b=batch_size)

        # cross_attn, cross_ff = self.cross_attend_blocks

        # for it in range(self.iterations):
        #     # encoder cross attention
        #     x = cross_attn(x, context=stacked_inputs, mask=mask) + x
        #     x = cross_ff(x) + x

        #     # self-attention layers
        #     for self_attn, self_ff in self.layers:
        #         x = self_attn(x) + x
        #         x = self_ff(x) + x

        # # decoder cross attention
        # latents = self.decoder_cross_attn(stacked_inputs, context=x)
        # #x = self.pos_drop(x)
        stacked_attention_mask = torch.ones((stacked_inputs.shape[0], stacked_inputs.shape[1])).to(stacked_inputs.device)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        latents = transformer_outputs['last_hidden_state']
        predicted_token_act = latents[:, -self.horizon*self.action_dim:, :].reshape(batch_size, self.horizon, self.action_dim, -1)
        predicted_trans = self.predict_trans(predicted_token_act[:,:,:3])
        predicted_quat = self.predict_quat(predicted_token_act[:,:,3:6])
        predicted_gripper = self.predict_gripper(predicted_token_act[:,:,-1:])
        predicted_act = torch.cat([predicted_trans,predicted_quat,predicted_gripper], dim=2).reshape(batch_size, self.horizon*self.action_dim, self.act_cls)
        return predicted_act
class VideoDiffuserModel(nn.Module):
    "Video-based MT-Diffuser with a PerceiverIO architecture"

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=256,
            hidden_dim=256,
            shape=(4,24,24),
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=12,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=8,
            max_ep_len=1000,
            patch_size=2,
            in_chans=4,
            act_cls=256,
            obs_cls=2048,
            vqvae=None,
            vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
            num_latents=2048,
            pretrain=True,
            multiview=False,
            meta=False,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.multiview=multiview
        self.horizon = horizon*4
        self.hidden_size = hidden_size
        self.state_dim = transition_dim - 1
        self.action_dim = action_dim
        self.prompt_trajectories = prompt_trajectories
        self.task_list = task_list
        #self.vqvae = vqvae
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.traj_model = pretrainModel(self.horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim,
            hidden_dim,
            shape,
            dim_mults,
            attention,
            depth,
            mlp_ratio,
            hidden_size,
            num_heads,
            train_device,
            prompt_trajectories,
            task_list,
            action_dim,
            max_ep_len,
            patch_size,
            in_chans,
            act_cls,
            obs_cls,
            vqvae=vqvae,
            vqvae_ckpt='./lightning_logs/version_45/checkpoints/last.ckpt',
            num_latents=2048)
        
        if not pretrain:
            if multiview:
                self.act_model = MultiviewFinetuneModel(self.horizon,
                            transition_dim,
                            cond_dim,
                            num_tasks,
                            dim,
                            hidden_dim,
                            shape,
                            dim_mults,
                            attention,
                            depth,
                            mlp_ratio,
                            hidden_size,
                            num_heads,
                            train_device,
                            prompt_trajectories,
                            task_list,
                            action_dim,
                            max_ep_len,
                            patch_size,
                            in_chans,
                            act_cls,
                            obs_cls,
                            vqvae=vqvae,
                            vqvae_ckpt='./lightning_logs/version_45/checkpoints/last.ckpt',
                            num_latents=2048)
            elif meta:
                self.act_model = MetaFinetuneModel_v1(self.horizon,
                            transition_dim,
                            cond_dim,
                            num_tasks,
                            dim,
                            hidden_dim,
                            shape,
                            dim_mults,
                            attention,
                            depth,
                            mlp_ratio,
                            hidden_size,
                            num_heads,
                            train_device,
                            prompt_trajectories,
                            task_list,
                            action_dim,
                            max_ep_len,
                            patch_size,
                            in_chans,
                            act_cls,
                            obs_cls,
                            vqvae=vqvae,
                            vqvae_ckpt='./lightning_logs/version_90/checkpoints/last.ckpt',
                            num_latents=2048)
            else:
                self.act_model = finetuneModel(self.horizon,
                            transition_dim,
                            cond_dim,
                            num_tasks,
                            dim,
                            hidden_dim,
                            shape,
                            dim_mults,
                            attention,
                            depth,
                            mlp_ratio,
                            hidden_size,
                            num_heads,
                            train_device,
                            prompt_trajectories,
                            task_list,
                            action_dim,
                            max_ep_len,
                            patch_size,
                            in_chans,
                            act_cls,
                            obs_cls,
                            vqvae=vqvae,
                            vqvae_ckpt='./lightning_logs/version_45/checkpoints/last.ckpt',
                            num_latents=2048)
            self.traj_model.eval()
    # def split(self, x, act_dim):
    #     #T, H, W = 4, 24, 24 #TODO
    #     T, H, W = 1, 24, 24
    #     z_dim = T * H * W
    #     act_dim = 4*act_dim
    #     z, a = x.split([z_dim, act_dim], dim=2) #TODO:action_dim
    #     z = einops.rearrange(z, 'B K (T H W) -> B K T H W', T=T, H=H, W=W)
    #     #a = einops.rearrange(a, 'B (L D) -> B L D', L=act_dim, D=1)
    #     return z, a
    def forward(self, x, time, cond, value, imgs=None, context_mask=None, view_condition=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        '''
        x : batch * (4*horizon) * 24 * 24; cond:  batch * (2*4) * 24 * 24; mask: batch * (2*16) * 7(action_dim)
        '''
        
        traj_obs, act = split(x, self.action_dim)
        # if view_condition:
        #     left, _ = split(view_condition['left'], self.action_dim)
        #     right, _ = split(view_condition['right'], self.action_dim)
        #     wrist, _ = split(view_condition['wrist'], self.action_dim)
        batch_size = traj_obs.shape[0]
        # act =x
        # batch_size = act.shape[0]
        #traj_obs, act = self.split(x, self.action_dim)
        if pretrain:
            predicted_traj, predicted_token_traj = self.traj_model(traj_obs, time, cond, value, x_condition=x_condition, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
            predicted_act = torch.zeros((x.shape[0], self.horizon*self.action_dim, self.act_cls), device=x.device)
            # predicted_act = self.act_model(act, time, cond, value, None, imgs, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
            #predicted_act = self.act_model(act, time, cond, value, predicted_token_traj, torch.zeros((x.shape[0], 4, 3, 128, 128), device=x.device, dtype=torch.float32), force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
        elif self.multiview:
            with torch.no_grad():
                predicted_traj, predicted_token_traj = self.traj_model(traj_obs, time, cond, value, x_condition=x_condition, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
                if view_condition:
                    # left_traj, left_token_traj = self.traj_model(left, time, cond, value, x_condition=view_condition['left_'], force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
                    # right_traj, right_token_traj = self.traj_model(right, time, cond, value, x_condition=view_condition['right_'], force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
                    # wrist_traj, wrist_token_traj = self.traj_model(wrist, time, cond, value, x_condition=view_condition['wrist_'], force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
                    left = self.traj_model.vqvae.codebook.dictionary_lookup(view_condition['left'].long())
                    left = shift_dim(left, -1, 1).flatten(2).transpose(1, 2)
                    right = self.traj_model.vqvae.codebook.dictionary_lookup(view_condition['right'].long())
                    right = shift_dim(right, -1, 1).flatten(2).transpose(1, 2)
                    wrist = self.traj_model.vqvae.codebook.dictionary_lookup(view_condition['wrist'].long())
                    wrist = shift_dim(wrist, -1, 1).flatten(2).transpose(1, 2)
                    condition = torch.cat([wrist,left,right,wrist], dim=-1)
            #predicted_traj, predicted_token_traj = self.traj_model(traj_obs, time, cond, value, x_condition=x_condition, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
            # multiview_representation  = torch.zeros((batch_size//4, 24*24, 4*256))
            # for i in range(batch_size//4):
            #     multiview_representation[i] = einops.rearrange(predicted_token_traj[4*i:4*(i+1)],'K L D -> L (K D)', K=4)#.reshape(4*24*24, 256)
            # multiview_representation  = multiview_representation.repeat(4,1,1).to(predicted_token_traj.device)
            #multiview_representation  = multiview_representation.to(predicted_token_traj.device)
            #act = act[:batch_size//4]
            # predicted_act = self.act_model(act, time, cond, value, multiview_representation, imgs, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
            predicted_act = self.act_model(act, time, cond, value, condition, imgs, view_condition=condition, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
            #predicted_act = predicted_act.repeat(4,1,1)
        else:
            with torch.no_grad():
                predicted_traj, predicted_token_traj = self.traj_model(traj_obs, time, cond, value, x_condition=x_condition, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
                
            predicted_traj, predicted_token_traj = predicted_traj.detach(), predicted_token_traj.detach()
            # if force:
            #     predicted_token_traj = 0 * predicted_token_traj
            # predicted_traj, predicted_token_traj = self.traj_model(traj_obs, time, cond, value, x_condition=x_condition, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
            # out = einops.rearrange(predicted_traj, 'B X c-> B c X').argmax(1)
            # out = einops.rearrange(out, 'B (T H W) -> B T H W', T=1, H=24, W=24)
            # out = self.traj_model.vqvae.codebook.dictionary_lookup(traj_obs.long())
            # out = einops.rearrange(out, 'B T H W C -> B (T H W) C', T=1, H=24, W=24)
            predicted_act = self.act_model(act, time, cond, value, predicted_token_traj, imgs, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
            # predicted_act = self.act_model(act, time, cond, value, None, imgs, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
            # predicted_act = self.act_model(act, time, cond, value, out, imgs, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
        return predicted_traj, predicted_act
        # return predicted_act
class R3MModel(nn.Module):
    def __init__(self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=256,
            hidden_dim=256,
            shape=(4,24,24),
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=12,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=8,
            max_ep_len=1000,
            patch_size=2,
            in_chans=4,
            act_cls=256,
            obs_cls=2048,
            vqvae=None,
            vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
            num_latents=2048,
            pretrain=True,
            multiview=False,
            meta=False,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.multiview=multiview
        self.horizon = horizon*4
        self.hidden_size = hidden_size
        self.state_dim = transition_dim - 1
        self.action_dim = action_dim
        self.prompt_trajectories = prompt_trajectories
        self.task_list = task_list
        #self.vqvae = vqvae
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.action_dim =action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.prompt_embed = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.propri_embed = nn.Sequential(
            nn.LayerNorm(39),
            nn.Linear(39, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.process_tokens = nn.Sequential(
            nn.Linear(2048, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        #self._prompt_trajectories = [np.load(f"./metaworld_prompts/{task_id}_prompt.npy", allow_pickle=True) for task_id in ['bin-picking-v2','box-close-v2','door-lock-v2','door-unlock-v2','hand-insert-v2']]
        #prompt_data = torch.tensor([get_prompt(prompt_trajectories, 1, 20, True) for ind in cond])
        # note: we don't predict states or returns for the paper
        self.predict_act_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_size, self.act_cls),
            
        )
        self.predict_act_y = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_size, self.act_cls),
        )
        self.predict_act_z = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_size, self.act_cls),
        )
        
        self.predict_gripper = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_size, self.act_cls),
        )#torch.nn.Linear(hidden_size, self.act_cls)
        self.act_embeddings =  nn.Sequential(
            #nn.LayerNorm(self.act_cls),
            nn.Linear(1, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )#nn.Embedding(self.act_cls+1, hidden_size)
        # self.act_embeddings_y =  nn.Sequential(
        #     #nn.LayerNorm(self.act_cls),
        #     nn.Linear(1, hidden_dim),  # TODO
        #     nn.Mish(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )
        # self.act_embeddings_z =  nn.Sequential(
        #     #nn.LayerNorm(self.act_cls),
        #     nn.Linear(1, hidden_dim),  # TODO
        #     nn.Mish(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )
        # self.act_embeddings_g =  nn.Sequential(
        #     #nn.LayerNorm(self.act_cls),
        #     nn.Linear(1, hidden_dim),  # TODO
        #     nn.Mish(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+24*24+4*self.action_dim+20*20, hidden_size)) #T, language, history, pred_act, pred_traj
        import transformers
        from .GPT2 import GPT2Model
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=6,
            n_head=2,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.transformer = GPT2Model(config)
        '''images blocks'''
        activation = "lrelu"
        self.img_patch_size = 13
        self.im_channels = 64
        # self.proprio_preprocess = DenseBlock(
        #        4, self.im_channels, norm=None, activation=activation,
        #     )
        # self.input_preprocess = Conv2DBlock(
        #     3,
        #     self.im_channels,
        #     kernel_sizes=1,
        #     strides=1,
        #     norm=None,
        #     activation=activation,
        # )
        # inp_pre_out_dim = self.im_channels
        # self.patchify = Conv2DBlock(
        #     inp_pre_out_dim,
        #     self.im_channels,
        #     kernel_sizes=self.img_patch_size,
        #     strides=self.img_patch_size,
        #     norm="group",
        #     activation=activation,
        #     padding=0,
        # )
        # self.img_embed = nn.Sequential(
        #     nn.LayerNorm(self.im_channels),
        #     nn.Linear(self.im_channels, hidden_dim),  # TODO
        #     nn.Mish(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )
        from r3m import load_r3m
        self.r3m = load_r3m("resnet50") # resnet18, resnet34
        self.r3m.eval()
        #self.r3m.to(self.act_embeddings.device)
        self.cross_attn_1 = CrossAttention(
                    1*24*24,
                    n_embd=dim,
                    condition_embd=dim,
                    n_head=1,
                    seq_len=1*24*24,
                    attn_pdrop=0.1,
                    resid_pdrop=0.1,
                    )
        self.self_attn = FullAttention(
                n_embd=dim,
                n_head=1,
                seq_len=1*24*24,
                attn_pdrop=0.1,
                resid_pdrop=0.1,
            )
    def forward(self, act, time, cond, pri, imgs, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        #img: (bs, num_img, img_feat_dim, h, w)
        #print(act.shape)
        x_act = self.act_embeddings(act.unsqueeze(-1).to(dtype=torch.float32))
        batch_size, seq_length = x_act.shape[0], x_act.shape[1]
        '''process img'''
        # x_condition = F.interpolate(x_condition, size=(96,96), mode='bilinear',
        #                   align_corners=False)
        x_condition = F.interpolate(x_condition, size=(224,224), mode='bilinear',
                          align_corners=False)
        with torch.no_grad():
            x_embeddings = self.r3m(x_condition)
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        t = self.time_mlp(time).unsqueeze(1)
        x_embeddings = self.process_tokens(x_embeddings).unsqueeze(1)
        #stacked_inputs = x_act
        #stacked_inputs = torch.cat((t, prompt_embeddings, image_emb, proprio, x_act), dim=1)
        #stacked_inputs = torch.cat((t, prompt_embeddings, predicted_traj, image_emb, x_act), dim=1)
        stacked_inputs = torch.cat((t, prompt_embeddings, x_embeddings, x_act), dim=1)
        #stacked_inputs = torch.cat((t, prompt_embeddings, image_emb, x_act), dim=1)
        #print("SHAPE:",t.shape, prompt_embeddings.shape, predicted_traj.shape, image_emb.shape, x_act.shape)
        stacked_inputs = prompt_embeddings * stacked_inputs# + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        #stacked_inputs = self.cross_attn_1(stacked_inputs, predicted_traj, mask=mask)[0] + stacked_inputs
        #stacked_inputs = self.self_attn(stacked_inputs)[0] + stacked_inputs
        # batchify latents
        
        stacked_attention_mask = torch.ones((stacked_inputs.shape[0], stacked_inputs.shape[1])).to(stacked_inputs.device)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        latents = transformer_outputs['last_hidden_state']
        predicted_token_act = latents[:, -self.horizon*self.action_dim:, :].reshape(batch_size, self.horizon, self.action_dim, -1)
        #predicted_act = self.predict_act(predicted_token_act)
        predicted_trans_x = self.predict_act_x(predicted_token_act[:,:,0:1])
        predicted_trans_y = self.predict_act_y(predicted_token_act[:,:,1:2])
        predicted_trans_z = self.predict_act_z(predicted_token_act[:,:,2:3])
        
        predicted_gripper = self.predict_gripper(predicted_token_act[:,:,3:])
        predicted_act = torch.cat([predicted_trans_x, predicted_trans_y,predicted_trans_z,predicted_gripper], dim=2).reshape(batch_size, self.horizon*self.action_dim, self.act_cls)
        
        return predicted_act

class Tasksmeta(nn.Module):
    "MT-Diffuser with a Transformer backbone"

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=56,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=None,
            max_ep_len=1000,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.state_dim = transition_dim - 1
        self.action_dim = action_dim
        self.prompt_trajectories = prompt_trajectories
        self.task_list = task_list
        import transformers
        from .GPT2 import GPT2Model
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=4,
            n_head=2,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(2*dim),
            nn.Linear(2*dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, 2*dim),
        )
        from r3m import load_r3m
        self.r3m = load_r3m("resnet50") # resnet18, resnet34
        self.r3m.eval()
        # self.return_mlp = nn.Sequential(
        #     nn.Linear(1, dim * 2),
        #     nn.Mish(),
        #     nn.Linear(dim * 2, 4*dim),
        #     nn.Mish(),
        #     nn.Linear(dim * 4, 2 * dim),
        # )
        # self.prompt_embed = nn.Sequential(
        #     nn.LayerNorm(self.state_dim+self.action_dim),
        #     nn.Linear(self.state_dim + self.action_dim, hidden_size*2),  # TODO
        #     nn.Mish(),
        #     nn.Linear(hidden_size * 2, 4 * hidden_size),
        #     nn.Mish(),
        #     nn.Linear(4 * hidden_size, hidden_size),
        # )
        self.prompt_embed = nn.Sequential(
            #nn.LayerNorm(1024),
            nn.Linear(1024, hidden_size),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.embed_obs = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 2 * hidden_size),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size * 2, 4 * hidden_size),
            nn.Mish(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        
        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_act = nn.Linear(self.action_dim, hidden_size)
        self.position_emb = nn.Parameter(torch.zeros(1, 24+horizon, hidden_size))

        # note: we don't predict states or returns for the paper
        self.predict_act = torch.nn.Linear(hidden_size, self.action_dim)
        #self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, time, cond, context_mask=None, value=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        #cond = cond.long()
        #x_condition = einops.rearrange(x_condition, 'i j k -> i (j k)')
        # prompt_data = get_prompt_batchs(self.prompt_trajectories, cond, num_episodes=1, num_steps=20, is_meta=True)
        # prompt_data = prompt_data.to(device=x.device, dtype=torch.float32).reshape(cond.shape[0], -1, self.state_dim+self.action_dim)#TODO
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        # obs_embeddings = self.embed_obs(x_condition)
        x_condition = F.interpolate(x_condition, size=(224,224), mode='bilinear',
                          align_corners=False)
        with torch.no_grad():
            x_embeddings = self.r3m(x_condition)
        obs_embeddings = self.embed_obs(x_embeddings).unsqueeze(1)
        t = self.time_mlp(time).unsqueeze(1)
        
        batch_size, seq_length = x.shape[0], x.shape[1]
        # value = value.view(-1, 1)
        # value = self.return_mlp(value)
        # cond_return = (value * mask).unsqueeze(1) + 1e-8
        act_embeddings = self.embed_act(x)
        # addition_length = 20 + 1 + 1 + 2  #TODO
        # addition_attention_mask = torch.ones((batch_size, addition_length), dtype=torch.long, device=x.device)
        # if attention_mask is None:
        #     # attention mask for GPT: 1 if can be attended to, 0 if not
        #     attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=x.device)
        #     '''set attention mask'''
        stacked_inputs = torch.cat((t, prompt_embeddings, obs_embeddings, act_embeddings), dim=1)
        stacked_attention_mask = torch.ones((stacked_inputs.shape[0], stacked_inputs.shape[1])).to(stacked_inputs.device)
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        #return_preds = self.predict_return(x[:, 1])  # predict next return given state and action
        act_preds = self.predict_act(x[:, -seq_length:, :])  # predict next state given state and action
        return act_preds

class Tasksmeta(nn.Module):
    "MT-Diffuser with a Transformer backbone"

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=56,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=None,
            max_ep_len=1000,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.state_dim = transition_dim - 1
        self.action_dim = action_dim
        self.prompt_trajectories = prompt_trajectories
        self.task_list = task_list
        import transformers
        from .GPT2 import GPT2Model
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=4,
            n_head=2,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(2*dim),
            nn.Linear(2*dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, 2*dim),
        )
        from r3m import load_r3m
        self.r3m = load_r3m("resnet50") # resnet18, resnet34
        self.r3m.eval()
        # self.return_mlp = nn.Sequential(
        #     nn.Linear(1, dim * 2),
        #     nn.Mish(),
        #     nn.Linear(dim * 2, 4*dim),
        #     nn.Mish(),
        #     nn.Linear(dim * 4, 2 * dim),
        # )
        # self.prompt_embed = nn.Sequential(
        #     nn.LayerNorm(self.state_dim+self.action_dim),
        #     nn.Linear(self.state_dim + self.action_dim, hidden_size*2),  # TODO
        #     nn.Mish(),
        #     nn.Linear(hidden_size * 2, 4 * hidden_size),
        #     nn.Mish(),
        #     nn.Linear(4 * hidden_size, hidden_size),
        # )
        self.prompt_embed = nn.Sequential(
            #nn.LayerNorm(1024),
            nn.Linear(1024, hidden_size),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.embed_obs = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, 2 * hidden_size),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size * 2, 4 * hidden_size),
            nn.Mish(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        
        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_act = nn.Linear(self.action_dim, hidden_size)
        self.position_emb = nn.Parameter(torch.zeros(1, 24+horizon, hidden_size))

        # note: we don't predict states or returns for the paper
        self.predict_act = torch.nn.Linear(hidden_size, self.action_dim)
        #self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, time, cond, context_mask=None, value=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        #cond = cond.long()
        #x_condition = einops.rearrange(x_condition, 'i j k -> i (j k)')
        # prompt_data = get_prompt_batchs(self.prompt_trajectories, cond, num_episodes=1, num_steps=20, is_meta=True)
        # prompt_data = prompt_data.to(device=x.device, dtype=torch.float32).reshape(cond.shape[0], -1, self.state_dim+self.action_dim)#TODO
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        # obs_embeddings = self.embed_obs(x_condition)
        x_condition = F.interpolate(x_condition, size=(224,224), mode='bilinear',
                          align_corners=False)
        with torch.no_grad():
            x_embeddings = self.r3m(x_condition)
        obs_embeddings = self.embed_obs(x_embeddings).unsqueeze(1)
        t = self.time_mlp(time).unsqueeze(1)
        
        batch_size, seq_length = x.shape[0], x.shape[1]
        # value = value.view(-1, 1)
        # value = self.return_mlp(value)
        # cond_return = (value * mask).unsqueeze(1) + 1e-8
        act_embeddings = self.embed_act(x)
        # addition_length = 20 + 1 + 1 + 2  #TODO
        # addition_attention_mask = torch.ones((batch_size, addition_length), dtype=torch.long, device=x.device)
        # if attention_mask is None:
        #     # attention mask for GPT: 1 if can be attended to, 0 if not
        #     attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=x.device)
        #     '''set attention mask'''
        stacked_inputs = torch.cat((t, prompt_embeddings, obs_embeddings, act_embeddings), dim=1)
        stacked_attention_mask = torch.ones((stacked_inputs.shape[0], stacked_inputs.shape[1])).to(stacked_inputs.device)
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        #return_preds = self.predict_return(x[:, 1])  # predict next return given state and action
        act_preds = self.predict_act(x[:, -seq_length:, :])  # predict next state given state and action
        return act_preds
class DT(nn.Module):
    "MT-Diffuser with a Transformer backbone"

    def __init__(
            self,
            horizon,
            transition_dim,
            cond_dim,
            num_tasks,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            attention=False,
            depth=56,
            mlp_ratio=4.0,
            hidden_size=256,
            num_heads=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=None,
            max_ep_len=1000,
            vqvae=None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.state_dim = transition_dim - 1
        self.action_dim = action_dim
        self.prompt_trajectories = prompt_trajectories
        self.task_list = task_list
        self.vqvae=vqvae.eval()
        import transformers
        from .GPT2 import GPT2Model
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            n_layer=6,
            n_head=2,
            n_inner=4 * 256,
            activation_function='mish',
            n_positions=1024,
            n_ctx=1023,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(2*dim),
            nn.Linear(2*dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, 2*dim),
        )
        # self.process_tokens = nn.Sequential(
        #     nn.Linear(256, hidden_dim),  # TODO
        #     nn.Mish(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Mish(),
        #     nn.Linear(hidden_dim, hidden_dim),
        # )
        # from r3m import load_r3m
        # self.r3m = load_r3m("resnet50") # resnet18, resnet34
        # self.r3m.eval()
        # self.return_mlp = nn.Sequential(
        #     nn.Linear(1, dim * 2),
        #     nn.Mish(),
        #     nn.Linear(dim * 2, 4*dim),
        #     nn.Mish(),
        #     nn.Linear(dim * 4, 2 * dim),
        # )
        # self.prompt_embed = nn.Sequential(
        #     nn.LayerNorm(self.state_dim+self.action_dim),
        #     nn.Linear(self.state_dim + self.action_dim, hidden_size*2),  # TODO
        #     nn.Mish(),
        #     nn.Linear(hidden_size * 2, 4 * hidden_size),
        #     nn.Mish(),
        #     nn.Linear(4 * hidden_size, hidden_size),
        # )
        self.prompt_embed = nn.Sequential(
            #nn.LayerNorm(1024),
            nn.Linear(1024, hidden_size),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.embed_obs = nn.Sequential(
            #nn.LayerNorm(2048),
            nn.Linear(hidden_size, 2 * hidden_size),  # TODO
            nn.Mish(),
            nn.Linear(hidden_size * 2, 4 * hidden_size),
            nn.Mish(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        
        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_act = nn.Linear(self.action_dim, hidden_size)
        self.position_emb = nn.Parameter(torch.zeros(1, 582, hidden_size))

        # note: we don't predict states or returns for the paper
        self.predict_act = torch.nn.Linear(hidden_size, self.action_dim)
        #self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, time, cond, context_mask=None, value=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        #cond = cond.long()
        #x_condition = einops.rearrange(x_condition, 'i j k -> i (j k)')
        # prompt_data = get_prompt_batchs(self.prompt_trajectories, cond, num_episodes=1, num_steps=20, is_meta=True)
        # prompt_data = prompt_data.to(device=x.device, dtype=torch.float32).reshape(cond.shape[0], -1, self.state_dim+self.action_dim)#TODO
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        # obs_embeddings = self.embed_obs(x_condition)
        # x_condition = F.interpolate(x_condition, size=(224,224), mode='bilinear',
        #                   align_corners=False)
        # with torch.no_grad():
        #     x_embeddings = self.r3m(x_condition)
        with torch.no_grad():
            x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        x_condition = shift_dim(x_condition, -1, 1).flatten(2).transpose(1, 2)
        obs_embeddings = self.embed_obs(x_condition)#.unsqueeze(1)
        t = self.time_mlp(time).unsqueeze(1)
        
        batch_size, seq_length = x.shape[0], x.shape[1]
        # value = value.view(-1, 1)
        # value = self.return_mlp(value)
        # cond_return = (value * mask).unsqueeze(1) + 1e-8
        act_embeddings = self.embed_act(x)
        # addition_length = 20 + 1 + 1 + 2  #TODO
        # addition_attention_mask = torch.ones((batch_size, addition_length), dtype=torch.long, device=x.device)
        # if attention_mask is None:
        #     # attention mask for GPT: 1 if can be attended to, 0 if not
        #     attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=x.device)
        #     '''set attention mask'''
        stacked_inputs = torch.cat((t, prompt_embeddings, obs_embeddings, act_embeddings), dim=1)
        stacked_attention_mask = torch.ones((stacked_inputs.shape[0], stacked_inputs.shape[1])).to(stacked_inputs.device)
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        #return_preds = self.predict_return(x[:, 1])  # predict next return given state and action
        act_preds = self.predict_act(x[:, -seq_length:, :])  # predict next state given state and action
        return act_preds