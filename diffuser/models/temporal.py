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
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
    AttentionBlock
)
from torch.distributions import Bernoulli
from videogpt import VideoData, VideoGPT, load_videogpt, VQVAE
import os
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

def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0., training=True):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn) # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v) # b x n_head x d1 x ... x dn x d

    return a
class FullAttention(nn.Module):
    def __init__(self, shape, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        seq_len = np.prod(shape)
        if self.causal:
            self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, q, k, v, decode_step, decode_idx):
        mask = self.mask if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(q, k, v, mask=mask,
                                           attn_dropout=self.attn_dropout,
                                           training=self.training)

        return view_range(out, 2, 3, old_shape)
class MultiHeadAttention(nn.Module):
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer,
                 causal, attn_type, attn_kwargs):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False) # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False) # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False) # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True) # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))
        self.attn = FullAttention(shape, causal, **attn_kwargs)

        self.cache = None

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        """ Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None

        Returns
            The output after performing attention
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (q.shape[0], n_head, *self.shape, self.d_k)
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                                    v=torch.zeros(v_shape, dtype=v.dtype, device=q.device))
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.clone(), v=v.clone())
            if self.causal:
                idx = (slice(None, None), slice(None, None), *[slice(i, i+ 1) for i in decode_idx])
                self.cache['k'][idx] = k
                self.cache['v'][idx] = v
            k, v = self.cache['k'], self.cache['v']

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a) # (b x seq_len x embd_dim)

        return a
class AttentionBlock(nn.Module):
    def __init__(self, shape, embd_dim, n_head, n_layer, dropout,
                 attn_type, attn_dropout, class_cond_dim, frame_cond_shape):
        super().__init__()
        self.use_frame_cond = frame_cond_shape is not None
        self.pre_attn_norm = LayerNorm(embd_dim, class_cond_dim)
        self.post_attn_dp = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(shape, embd_dim, embd_dim, n_head,
                                       n_layer, causal=True, attn_type=attn_type,
                                       attn_kwargs=dict(attn_dropout=attn_dropout))

        # if frame_cond_shape is not None:
        #     enc_len = np.prod(frame_cond_shape[:-1])
        #     self.pre_enc_norm = LayerNorm(embd_dim, class_cond_dim)
        #     self.post_enc_dp = nn.Dropout(dropout)
        #     self.enc_attn = MultiHeadAttention(shape, embd_dim, frame_cond_shape[-1],
        #                                        n_head, n_layer, attn_type='full',
        #                                        attn_kwargs=dict(attn_dropout=0.), causal=False)

        self.pre_fc_norm = LayerNorm(embd_dim, class_cond_dim)
        self.post_fc_dp = nn.Dropout(dropout)
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=embd_dim, out_features=embd_dim * 4),
            GeLU2(),
            nn.Linear(in_features=embd_dim * 4, out_features=embd_dim),
        )

    def forward(self, x, cond, decode_step, decode_idx):
        h = self.pre_attn_norm(x, cond)
        h = self.attn(h, h, h, decode_step, decode_idx)
        h = self.post_attn_dp(h)
        x = x + h

        # if self.use_frame_cond:
        #     h = self.pre_enc_norm(x, cond)
        #     if self.training:
        #         h = checkpoint(self.enc_attn, h, cond['frame_cond'], cond['frame_cond'],
        #                        decode_step, decode_idx)
        #     else:
        #         h = self.enc_attn(h, cond['frame_cond'], cond['frame_cond'],
        #                           decode_step, decode_idx)
        #     h = self.post_enc_dp(h)
        #     x = x + h

        h = self.pre_fc_norm(x, cond)
        h = self.fc_block(h)
        h = self.post_fc_dp(h)
        x = x + h

        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, 'B H L D -> B L (H D)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim, class_cond_dim, hidden_dim, skip=False, drop=0, drop_path=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim) if skip else None
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        # self.atten = AttentionBlock(
        #             shape=(4, 24, 24),
        #             embd_dim=embed_dim,
        #             n_head=4,
        #             n_layer=8,
        #             dropout=0.2,
        #             attn_dropout=0.3,
        #             class_cond_dim=class_cond_dim,
        #         )
        self.atten = Attention(
            dim=embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=drop, proj_drop=drop)
        self.skip_linear = nn.Linear(2 * embed_dim, embed_dim) if skip else None
        self.mlp = Mlp(in_features=embed_dim, hidden_features=1024, act_layer=nn.GELU, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x, cond=None, skip = None):
        #x: b * 4*24*24 *hidden_dim
        x = x*cond + cond
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
            x = self.norm1(x)
        x = x + self.drop_path(self.atten(x))
        x = self.norm2(x)

        x = x + self.drop_path(self.mlp(x))
        x = self.norm3(x)
        #x = self.atten(x, cond)
        return x
# class AddBroadcastPosEmbed(nn.Module):
#     def __init__(self, shape, embd_dim, dim=-1):
#         super().__init__()
#         assert dim in [-1, 1] # only first or last dim supported
#         self.shape = shape
#         self.n_dim = n_dim = len(shape)
#         self.embd_dim = embd_dim
#         self.dim = dim

#         assert embd_dim % n_dim == 0, f"{embd_dim} % {n_dim} != 0"
#         self.emb = nn.ParameterDict({
#              f'd_{i}': nn.Parameter(torch.randn(shape[i], embd_dim // n_dim) * 0.01
#                                     if dim == -1 else
#                                     torch.randn(embd_dim // n_dim, shape[i]) * 0.01)
#              for i in range(n_dim)
#         })

#     def forward(self, x, decode_step=None, decode_idx=None):
#         embs = []
#         for i in range(self.n_dim):
#             e = self.emb[f'd_{i}']
#             if self.dim == -1:
#                 # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
#                 e = e.view(1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1)
#                 e = e.expand(1, *self.shape, -1)
#             else:
#                 e = e.view(1, -1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)))
#                 e = e.expand(1, -1, *self.shape)
#             embs.append(e)

#         embs = torch.cat(embs, dim=self.dim)
#         if decode_step is not None:
#             embs = tensor_slice(embs, [0, *decode_idx, 0],
#                                 [x.shape[0], *(1,) * self.n_dim, x.shape[-1]])

#         return x + embs
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=2, in_chans=256, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        #B, D, H, W, C = x.shape
        #assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(shift_dim(x, -1, 1)).flatten(2).transpose(1, 2)
        return x
def unpatchify(x, dim):
    patch_size = int((x.shape[2]//dim))
    h = w = int((x.shape[1] / 2) ** .5)
    #assert h * w == x.shape[1] and patch_size ** 2 * in_chans == x.shape[2]
    x = einops.rearrange(x, 'B (p2 h w) (p1 C) -> B (p2 p1) h w C', h=h, w=w, p1=patch_size, p2=2, C=dim, B=x.shape[0])
    return x
def split(x):
        T, H, W = 4, 24, 24
        z_dim = T * H * W
        act_dim = 4*8
        z, a = x.split([z_dim, act_dim], dim=1) #TODO:action_dim
        z = einops.rearrange(z, 'B (T H W) -> B T H W', T=T, H=H, W=W)
        #a = einops.rearrange(a, 'B (L D) -> B L D', L=act_dim, D=1)
        return z, a
class VideoModel(nn.Module):
    "MT-Diffuser with a Transformer backbone"

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
            vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
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
        self.load_vqvae(vqvae_ckpt)
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=hidden_dim)
        self.in_blocks = nn.ModuleList([
            Block(
                embed_dim=self.dim, class_cond_dim=2, hidden_dim=hidden_dim)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                embed_dim=self.dim, class_cond_dim=2, hidden_dim=hidden_dim)

        self.out_blocks = nn.ModuleList([
            Block(
                embed_dim=self.dim, class_cond_dim=2, hidden_dim=hidden_dim)
            for _ in range(depth // 2)])
        
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

        self.mask_dist = Bernoulli(probs=0.8)
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_act = nn.Linear(self.action_dim, hidden_size)
        self.embed_traj = nn.Linear(dim, hidden_dim)

        self.predict_act = torch.nn.Linear(hidden_size, self.act_cls)
        self.patch_dim = patch_size * patch_size * patch_size * self.obs_cls
        self.predict_traj = nn.Sequential(
            nn.Linear(hidden_size, self.patch_dim),
        )
        #self.pos_drop = nn.Dropout(p=pos_drop_rate)
        #self.predict_return = torch.nn.Linear(hidden_size, 1)
        # self.position_emb = nn.Parameter(torch.zeros(1, 1+1+32+2*12*12+self.horizon+2*12*12, hidden_size)) #T, language, prev_action, history, 
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+2*12*12+self.horizon*self.action_dim+2*12*12, hidden_size)) #T, language, history, pred_act, pred_traj
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt)
        self.vqvae.eval()
    def forward(self, x, time, cond, value, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True):
        '''
            x : [ batch x horizon x transition ]
        '''
        '''
        x : batch * (4*horizon) * 24 * 24; cond:  batch * (2*4) * 24 * 24; mask: batch * (2*16) * 7(action_dim)
        '''
        #b,h = x.shape[0],x.shape[1]
        # print(x.shape)
        # print(x_condition.shape)
        # print(cond.shape)
        #cond = cond.long()
        #x = x.flatten(start_dim=1, end_dim=-2)
        traj_obs, act = split(x)
        if pretrain:
            #print(traj_obs.shape)
            #print(act.max(), act.min())
            traj_obs = self.vqvae.codebook.dictionary_lookup(traj_obs.long())

            #x = einops.rearrange(x, 'i j h k b c-> i (j h) k b c')
            x = self.patch_embed(traj_obs)
            x = torch.cat([x, torch.zeros((x.shape[0], self.horizon*self.action_dim, self.hidden_size)).to(x.device)], dim=1)
        else:
            x_obs = self.vqvae.codebook.dictionary_lookup(traj_obs.long())
            x_obs = self.patch_embed(x_obs)
            x_act = self.vqvae.codebook.dictionary_lookup(act.long())
            x_act = self.embed_act(x_act)
            x = torch.cat([x, x_act], dim=1)
        #x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        #x_condition = einops.rearrange(x_condition, 'i j h k b c-> i (j h) k b c')
        x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        x_condition = self.patch_embed(x_condition)
        batch_size, seq_length = x.shape[0], x.shape[1]
        #x_condition = x_condition.flatten(start_dim=1, end_dim=-2)
        #x_condition = einops.rearrange(x_condition, 'i j k -> i (j k)')
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        #print(prompt_embeddings.shape)
        obs_embeddings = self.embed_history(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        traj_embeddings = self.embed_traj(x)
        # if pretrain:
        #     mask = torch.zeros((traj_embeddings.shape[0], 32, self.hidden_size)).to(x.device)
        #     #print(obs_embeddings.shape,mask.shape)
        #     obs_embeddings = torch.cat([obs_embeddings, mask], dim=1)
        # if flag:
        #     traj_embeddings = torch.cat([traj_embeddings, torch.zeros((batch_size, seq_length, self.hidden_size))], dim=1)
        """
        addition_length = 1 + 1 + 8  #TODO
        addition_attention_mask = torch.ones((batch_size, addition_length), dtype=torch.long, device=x.device)
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=x.device)
            '''set attention mask'''
        """
        #print(t.shape,obs_embeddings.shape,traj_embeddings.shape)
        stacked_inputs = torch.cat((t, prompt_embeddings, obs_embeddings, traj_embeddings), dim=1)
        # stacked_attention_mask = torch.cat(
        #     (addition_attention_mask, attention_mask), dim=1)

        stacked_inputs = t * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
        stacked_inputs = self.embed_ln(stacked_inputs)
        #x = self.pos_drop(x)

        skips = []
        for blk in self.in_blocks:
            x = blk(x, cond = t)
            skips.append(x)

        x = self.mid_block(x, cond=t)

        for blk in self.out_blocks:
            x = blk(x, cond=t, skip=skips.pop())
        predicted_token_traj, predicted_token_act = x[:, -2*12*12-self.horizon*self.action_dim:-self.horizon*self.action_dim, :], x[:, -self.horizon*self.action_dim:, :]
        predicted_traj = self.predict_traj(predicted_token_traj)
        predicted_traj = unpatchify(predicted_traj, self.obs_cls)
        predicted_act = self.predict_act(predicted_token_act)
        # stacked_inputs = self.pos_embd(stacked_inputs)
        # transformer_outputs = self.transformer(
        #     inputs_embeds=stacked_inputs,
        #     attention_mask=stacked_attention_mask,
        # )
        #x = transformer_outputs['last_hidden_state']
        #return_preds = self.predict_return(x[:, 1])  # predict next return given state and action
        #act_preds = self.predict_act(x[:, -seq_length:, :])  # predict next state given state and action
        #print(predicted_traj.shape, predicted_act.shape)
        return predicted_traj, predicted_act
    #NOTE: output one-hot prediction


# helper classes
from functools import wraps
from einops import rearrange, repeat, reduce
from torch import nn, einsum
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class PerAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
class PerceiverVideoModel(nn.Module):
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
        self.vqvae = vqvae
        self.vqvae.eval()
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=hidden_dim)
        self.in_blocks = nn.ModuleList([
            Block(
                embed_dim=self.dim, class_cond_dim=2, hidden_dim=hidden_dim)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                embed_dim=self.dim, class_cond_dim=2, hidden_dim=hidden_dim)

        self.out_blocks = nn.ModuleList([
            Block(
                embed_dim=self.dim, class_cond_dim=2, hidden_dim=hidden_dim)
            for _ in range(depth // 2)])
        
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
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_act = nn.Linear(self.action_dim, hidden_size)
        self.embed_traj = nn.Linear(dim, hidden_dim)

        self.predict_act = torch.nn.Linear(hidden_size, self.act_cls)
        self.patch_dim = patch_size * patch_size * patch_size * self.obs_cls
        self.predict_traj = nn.Sequential(
            nn.Linear(hidden_size, self.obs_cls),
        )
        #self.pos_drop = nn.Dropout(p=pos_drop_rate)
        #self.predict_return = torch.nn.Linear(hidden_size, 1)
        # self.position_emb = nn.Parameter(torch.zeros(1, 1+1+32+2*12*12+self.horizon+2*12*12, hidden_size)) #T, language, prev_action, history, 
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+2*12*12*8+self.horizon*self.action_dim+2*12*12*8, hidden_size)) #T, language, history, pred_act, pred_traj
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
        self.act_embeddings = nn.Embedding(self.act_cls+1, hidden_size)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt)
        self.vqvae.eval()
    def forward(self, x, time, cond, value, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        '''
        x : batch * (4*horizon) * 24 * 24; cond:  batch * (2*4) * 24 * 24; mask: batch * (2*16) * 7(action_dim)
        '''
        traj_obs, act = split(x)
        #x = einops.rearrange(x, 'B (T H W) -> B T H W', T=4, H=24, W=24)
        traj_obs = self.vqvae.codebook.dictionary_lookup(traj_obs.long())
        traj_obs = shift_dim(traj_obs, -1, 1).flatten(2).transpose(1, 2)
        #traj_obs = einops.rearrange(x, 'B (T H W) -> B T H W', T=4, H=24, W=24)
        if pretrain:
            #print(traj_obs.max(),traj_obs.min())
            # traj_obs = self.vqvae.codebook.dictionary_lookup(traj_obs.long())
            # x = self.patch_embed(traj_obs)
            x = torch.cat([traj_obs, torch.zeros((x.shape[0], self.horizon*self.action_dim, self.hidden_size)).to(x.device)], dim=1)
        else:
            x_act = self.act_embeddings(act-self.obs_cls)
            #print(x.shape,x_act.shape)
            x = torch.cat([traj_obs, x_act], dim=1)
        x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        #x_condition = self.patch_embed(x_condition)
        x_condition = shift_dim(x_condition, -1, 1).flatten(2).transpose(1, 2)
        batch_size, seq_length = x.shape[0], x.shape[1]
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        obs_embeddings = self.embed_history(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        traj_embeddings = self.embed_traj(x)
        stacked_inputs = torch.cat((t, prompt_embeddings, obs_embeddings, traj_embeddings), dim=1)

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

        predicted_token_traj, predicted_token_act = latents[:, -2*12*12*8-self.horizon*self.action_dim:-self.horizon*self.action_dim, :], latents[:, -self.horizon*self.action_dim:, :]
        predicted_traj = self.predict_traj(predicted_token_traj)
        #predicted_traj = unpatchify(predicted_traj, self.obs_cls)
        predicted_act = self.predict_act(predicted_token_act)
        #predicted_traj = einops.rearrange(predicted_traj, 'B (T H W) c -> B T H W c', T=4, H=24, W=24)
        #predicted_act = einops.rearrange(predicted_act, 'B (T H W) c -> B T H W c', T=4, H=24, W=24)
        if pretrain:
            predicted_act = predicted_act.detach()
        else:
            predicted_traj = predicted_traj.detach()
        return predicted_traj, predicted_act
        #return predicted_traj
    #NOTE: output one-hot prediction
class VQVideoModel(nn.Module):
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
            vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
            num_latents=2048,
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
        self.load_vqvae(vqvae_ckpt)
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=hidden_dim)
        self.in_blocks = nn.ModuleList([
            Block(
                embed_dim=self.dim, class_cond_dim=2, hidden_dim=hidden_dim)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                embed_dim=self.dim, class_cond_dim=2, hidden_dim=hidden_dim)

        self.out_blocks = nn.ModuleList([
            Block(
                embed_dim=self.dim, class_cond_dim=2, hidden_dim=hidden_dim)
            for _ in range(depth // 2)])
        
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
        self.act_cls = act_cls
        self.obs_cls = obs_cls
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.embed_act = nn.Linear(self.action_dim, hidden_size)
        self.embed_traj = nn.Linear(dim, hidden_dim)

        self.predict_act = torch.nn.Linear(hidden_size, self.act_cls)
        self.patch_dim = patch_size * patch_size * patch_size * self.obs_cls
        # self.predict_traj = nn.Sequential(
        #     nn.Linear(hidden_size, self.patch_dim),
        # )
        self.predict_traj = nn.Sequential(
             nn.Linear(hidden_size, self.obs_cls),
         )
        #self.pos_drop = nn.Dropout(p=pos_drop_rate)
        #self.predict_return = torch.nn.Linear(hidden_size, 1)
        # self.position_emb = nn.Parameter(torch.zeros(1, 1+1+32+2*12*12+self.horizon+2*12*12, hidden_size)) #T, language, prev_action, history, 
        #self.position_emb = nn.Parameter(torch.zeros(1, 1+1+2*12*12+self.horizon*self.action_dim+2*12*12, hidden_size)) #T, language, history, pred_act, pred_traj
        self.position_emb = nn.Parameter(torch.zeros(1, 1+1+2*12*12*8+2*12*12*8, hidden_size))
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
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt)
        self.vqvae.eval()
    def forward(self, x, time, cond, value, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        '''
        x : batch * (4*horizon) * 24 * 24; cond:  batch * (2*4) * 24 * 24; mask: batch * (2*16) * 7(action_dim)
        '''
        x = einops.rearrange(x, 'B (T H W) -> B T H W', T=4, H=24, W=24)
        traj_obs = self.vqvae.codebook.dictionary_lookup(x.long())
        x = shift_dim(traj_obs, -1, 1).flatten(2).transpose(1, 2)
        #x = self.patch_embed(traj_obs)
        # traj_obs, act = split(x)
        # if pretrain:
        #     #print(traj_obs.max(),traj_obs.min())
        #     traj_obs = self.vqvae.codebook.dictionary_lookup(traj_obs.long())
        #     x = self.patch_embed(traj_obs)
        #     x = torch.cat([x, torch.zeros((x.shape[0], self.horizon*self.action_dim, self.hidden_size)).to(x.device)], dim=1)
        # else:
        #     x_obs = self.vqvae.codebook.dictionary_lookup(traj_obs.long())
        #     x_obs = self.patch_embed(x_obs)
        #     x_act = self.vqvae.codebook.dictionary_lookup(act.long())
        #     x_act = self.embed_act(x_act)
        #     x = torch.cat([x, x_act], dim=1)
        x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        #x_condition = self.patch_embed(x_condition)
        x_condition = shift_dim(x_condition, -1, 1).flatten(2).transpose(1, 2)
        batch_size, seq_length = x.shape[0], x.shape[1]
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        obs_embeddings = self.embed_history(x_condition)
        t = self.time_mlp(time).unsqueeze(1)
        traj_embeddings = self.embed_traj(x)
        stacked_inputs = torch.cat((t, prompt_embeddings, obs_embeddings, traj_embeddings), dim=1)

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

        #predicted_token_traj = latents[:, -2*12*12:, :]
        predicted_token_traj = latents[:, -2*12*12*8:, :]
        predicted_traj = self.predict_traj(predicted_token_traj)
        #predicted_traj = unpatchify(predicted_traj, self.obs_cls)
        predicted_traj = einops.rearrange(predicted_traj, 'B (T H W) c -> B T H W c', T=4, H=24, W=24)
        return predicted_traj
