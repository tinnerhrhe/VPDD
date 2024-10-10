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
from .encoder import *
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
class LayerNorm(nn.Module):
    def __init__(self, embd_dim, class_cond_dim):
        super().__init__()
        self.conditional = class_cond_dim is not None

        if self.conditional:
            self.w = nn.Linear(class_cond_dim, embd_dim, bias=False)
            nn.init.constant_(self.w.weight.data, 1. / np.sqrt(class_cond_dim))
            self.wb = nn.Linear(class_cond_dim, embd_dim, bias=False)
        else:
            self.g = nn.Parameter(torch.ones(embd_dim, dtype=torch.float32), requires_grad=True)
            self.b = nn.Parameter(torch.zeros(embd_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, x, cond):
        if self.conditional:  # (b, cond_dim)
            g = 1 + self.w(cond['class_cond']).view(x.shape[0], *(1,)*(len(x.shape)-2), x.shape[-1]) # (b, ..., embd_dim)
            b = self.wb(cond['class_cond']).view(x.shape[0], *(1,)*(len(x.shape)-2), x.shape[-1])
        else:
            g = self.g  # (embd_dim,)
            b = self.b

        x_float = x.float()

        mu = x_float.mean(dim=-1, keepdims=True)
        s = (x_float - mu).square().mean(dim=-1, keepdims=True)
        x_float = (x_float - mu) * (1e-5 + s.rsqrt())  # (b, ..., embd_dim)
        x_float = x_float * g + b

        x = x_float.type_as(x)
        return x


def unpatchify(x, dim):
    patch_size = int((x.shape[2]//dim))
    h = w = int((x.shape[1] / 2) ** .5)
    #assert h * w == x.shape[1] and patch_size ** 2 * in_chans == x.shape[2]
    x = einops.rearrange(x, 'B (p2 h w) (p1 C) -> B (p2 p1) h w C', h=h, w=w, p1=patch_size, p2=2, C=dim, B=x.shape[0])
    return x
def split(x, act_dim):
        T, H, W = 4, 24, 24
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
            pretrain=True,
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
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+2*12*12*8, hidden_size)) #T, language, history, pred_act, pred_traj
        """ Perceiver Transformer """
         # latent vectors (that are randomly initialized)
        #self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))
        self.encoder = encoder(
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
            act_cls=act_cls,
            obs_cls=obs_cls,
            vqvae=self.vqvae,
            vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
            num_latents=2048,
        )
        # encoder cross attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(hidden_dim, PerAttention(hidden_dim,
                                          hidden_dim,
                                          heads=1,
                                          dim_head=64,
                                          dropout=0.1),
                    context_dim=hidden_dim),
            PreNorm(hidden_dim, FeedForward(hidden_dim))
        ])

        get_latent_attn = lambda: PreNorm(hidden_dim,
                                          PerAttention(hidden_dim, heads=1,
                                                    dim_head=64, dropout=0.1))
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
            #x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        x_condition = self.encoder(cond, x_condition=x_condition)
        traj_obs = shift_dim(traj_obs, -1, 1).flatten(2).transpose(1, 2)
        batch_size, seq_length = traj_obs.shape[0], traj_obs.shape[1]
        #x_condition = shift_dim(x_condition, -1, 1).flatten(2).transpose(1, 2)
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        #obs_embeddings = self.embed_history(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        traj_embeddings = self.embed_traj(traj_obs)
        stacked_inputs = torch.cat((t, prompt_embeddings, traj_embeddings), dim=1)
        stacked_inputs = prompt_embeddings * stacked_inputs
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        # batchify latents
        #x = repeat(self.latents, 'n d -> b n d', b=batch_size)

        cross_attn, cross_ff = self.cross_attend_blocks

        for it in range(self.iterations):
            # encoder cross attention
            x_condition = cross_attn(x_condition, context=stacked_inputs, mask=mask) + x_condition
            x_condition = cross_ff(x_condition) + x_condition

            # self-attention layers
            for self_attn, self_ff in self.layers:
                x_condition = self_attn(x_condition) + x_condition
                x_condition = self_ff(x_condition) + x_condition

        # decoder cross attention
        latents = self.decoder_cross_attn(stacked_inputs, context=x_condition)
        #x = self.pos_drop(x)

        predicted_token_traj = latents[:, -2*12*12*8:, :]
        predicted_traj = self.predict_traj(predicted_token_traj)
        return predicted_traj
class pretrainModel_v1(nn.Module):
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
            pretrain=True,
            meta=True,
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
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+2*12*12*8, hidden_size)) #T, language, history, pred_act, pred_traj
        """ Perceiver Transformer """
         # latent vectors (that are randomly initialized)
        #self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))
        self.encoder = encoder_v1(
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
            act_cls=act_cls,
            obs_cls=obs_cls,
            vqvae=self.vqvae,
            vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
            num_latents=2048,
        )
        # encoder cross attention
        # self.cross_attend_blocks = nn.ModuleList([
        #     PreNorm(hidden_dim, PerAttention(hidden_dim,
        #                                   hidden_dim,
        #                                   heads=1,
        #                                   dim_head=64,
        #                                   dropout=0.1),
        #             context_dim=hidden_dim),
        #     PreNorm(hidden_dim, FeedForward(hidden_dim))
        # ])

        # get_latent_attn = lambda: PreNorm(hidden_dim,
        #                                   PerAttention(hidden_dim, heads=1,
        #                                             dim_head=64, dropout=0.1))
        # get_latent_ff = lambda: PreNorm(hidden_dim, FeedForward(hidden_dim))
        # get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # # self attention layers
        # self.layers = nn.ModuleList([])
        # cache_args = {'_cache': False}

        # for i in range(depth//2):
        #     self.layers.append(nn.ModuleList([
        #         get_latent_attn(**cache_args),
        #         get_latent_ff(**cache_args)
        #     ]))

        # # decoder cross attention
        # self.decoder_cross_attn = PreNorm(hidden_dim, PerAttention(hidden_dim,
        #                                                         hidden_dim,
        #                                                         heads=1,
        #                                                         dim_head=32,
        #                                                         dropout=0.0),
        #                                   context_dim=hidden_dim)
        # self.iterations = 1
        self.cross_attn_1 = CrossAttention(
                    4*16*16,
                    n_embd=dim,
                    condition_embd=dim,
                    n_head=1,
                    seq_len=4*16*16,
                    attn_pdrop=0.1,
                    resid_pdrop=0.1,
                    )
        self.full_attn_1 = FullAttention(
                    n_embd=dim,
                    n_head=1,
                    seq_len=4*16*16,
                    attn_pdrop=0.1,
                    resid_pdrop=0.1,
                    )
        self.ln = nn.LayerNorm(hidden_size)
    def forward(self, traj_obs, time, cond, value, context_mask=None, x_condition=None, force=False, imgs=None,return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        with torch.no_grad():
            traj_obs = self.vqvae.codebook.dictionary_lookup(traj_obs.long())
            #x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        x_condition = self.encoder(cond, x_condition=x_condition)
        traj_obs = shift_dim(traj_obs, -1, 1).flatten(2).transpose(1, 2)
        batch_size, seq_length = traj_obs.shape[0], traj_obs.shape[1]
        #x_condition = shift_dim(x_condition, -1, 1).flatten(2).transpose(1, 2)
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        #obs_embeddings = self.embed_history(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        traj_embeddings = self.embed_traj(traj_obs)
        stacked_inputs = torch.cat((t, prompt_embeddings, traj_embeddings), dim=1)
        stacked_inputs = prompt_embeddings * stacked_inputs
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        # batchify latents
        #x = repeat(self.latents, 'n d -> b n d', b=batch_size)
        stacked_inputs = self.cross_attn_1(stacked_inputs, x_condition, mask=mask)[0] + stacked_inputs
        stacked_inputs = self.full_attn_1(self.ln(stacked_inputs), x_condition, mask=mask)[0] + stacked_inputs
        # cross_attn, cross_ff = self.cross_attend_blocks

        # for it in range(self.iterations):
        #     # encoder cross attention
        #     x_condition = cross_attn(x_condition, context=stacked_inputs, mask=mask) + x_condition
        #     x_condition = cross_ff(x_condition) + x_condition

        #     # self-attention layers
        #     for self_attn, self_ff in self.layers:
        #         x_condition = self_attn(x_condition) + x_condition
        #         x_condition = self_ff(x_condition) + x_condition

        # # decoder cross attention
        # latents = self.decoder_cross_attn(stacked_inputs, context=x_condition)
        # #x = self.pos_drop(x)
        
        # predicted_token_traj = latents[:, -2*12*12*8:, :]
        # predicted_traj = self.predict_traj(predicted_token_traj)
        predicted_token_traj = stacked_inputs[:, -1*24*24:, :]
        predicted_traj = self.predict_traj(predicted_token_traj)
        return predicted_traj

class sodaActModel(nn.Module):
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
            pretrain=True,
            meta=True,
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
        self.horizon=4
        self.action_dim=4
        self.vqvae=vqvae.eval()
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.embed_ln = nn.LayerNorm(hidden_size)
       
        self.prompt_embed = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act_embeddings =  nn.Sequential(
            #nn.LayerNorm(self.act_cls),
            nn.Linear(1, hidden_dim),  # TODO
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
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
        
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+2*12*12*8, hidden_size)) #T, language, history, pred_act, pred_traj
        
        self.encoder = encoder_v1(
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
            act_cls=act_cls,
            obs_cls=obs_cls,
            vqvae=self.vqvae,
            vqvae_ckpt='./lightning_logs/version_52/checkpoints/last.ckpt',
            num_latents=2048,
        )
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
    def forward(self, act, time, cond, value, context_mask=None, x_condition=None, force=False, imgs=None,return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        x_act = self.act_embeddings(act.unsqueeze(-1).to(dtype=torch.float32))
        with torch.no_grad():
            x_condition = self.encoder(cond, x_condition=x_condition)
        
        batch_size, seq_length = x_act.shape[0], x_act.shape[1]
        #x_condition = shift_dim(x_condition, -1, 1).flatten(2).transpose(1, 2)
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        #obs_embeddings = self.embed_history(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        
        stacked_inputs = torch.cat((t, prompt_embeddings, x_condition,x_act), dim=1)
        stacked_inputs = prompt_embeddings * stacked_inputs
        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = torch.ones((stacked_inputs.shape[0], stacked_inputs.shape[1])).to(stacked_inputs.device)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        latents = transformer_outputs['last_hidden_state']
        predicted_token_act = latents[:, -self.horizon*self.action_dim:, :].reshape(batch_size, self.horizon, self.action_dim, -1)
        predicted_trans_x = self.predict_act_x(predicted_token_act[:,:,0:1])
        predicted_trans_y = self.predict_act_y(predicted_token_act[:,:,1:2])
        predicted_trans_z = self.predict_act_z(predicted_token_act[:,:,2:3])
        
        predicted_gripper = self.predict_gripper(predicted_token_act[:,:,3:])
        predicted_act = torch.cat([predicted_trans_x, predicted_trans_y,predicted_trans_z,predicted_gripper], dim=2).reshape(batch_size, self.horizon*self.action_dim, self.act_cls)
        
        return predicted_act
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
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.horizon = horizon*4
        self.hidden_size = hidden_size
        self.state_dim = transition_dim - 1
        self.action_dim = action_dim
        self.prompt_trajectories = prompt_trajectories
        self.task_list = task_list
        
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
    def forward(self, x, time, cond, value, imgs=None, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        '''
        x : batch * (4*horizon) * 24 * 24; cond:  batch * (2*4) * 24 * 24; mask: batch * (2*16) * 7(action_dim)
        '''
        traj_obs, act = split(x, self.action_dim)
        if pretrain:
            predicted_traj, predicted_token_traj = self.traj_model(traj_obs, time, cond, value, x_condition=x_condition, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
            predicted_act = torch.zeros((x.shape[0], self.horizon*self.action_dim, self.act_cls), device=x.device)
            #predicted_act = self.act_model(act, time, cond, value, predicted_token_traj, torch.zeros((x.shape[0], 4, 3, 128, 128), device=x.device, dtype=torch.float32), force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
        else:
            with torch.no_grad():
                predicted_traj, predicted_token_traj = self.traj_model(traj_obs, time, cond, value, x_condition=x_condition, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
            predicted_traj, predicted_token_traj = predicted_traj.detach(), predicted_token_traj.detach()
            predicted_act = self.act_model(act, time, cond, value, predicted_token_traj, imgs, force=force, flag=flag, attention_mask=attention_mask, pretrain=pretrain, mask=mask, clf=clf)
        return predicted_traj, predicted_act
