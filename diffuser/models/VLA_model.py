import math
import torch
from torch import nn
import torch.nn.functional as F


import numpy as np
from einops import rearrange

from inspect import isfunction
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
from videogpt.utils import view_range, shift_dim, trunc_normal_, tensor_slice
from torch.distributions import Bernoulli
from videogpt import VideoData, VideoGPT, load_videogpt, VQVAE
import os
import einops
from torch.distributions import Bernoulli
from einops.layers.torch import Rearrange
class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x
class AdaInsNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adainsnorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1,-2) * (1 + scale) + shift
        return x
class Conv_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1, padding=1)
        self.act = act
        self.conv2 = nn.Conv2d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        n =  x.size()[1]
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(n)))
        x = self.conv2(self.act(self.conv1(x)))
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.dropout(x)
class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 seq_len=None, # the max length of sequence
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 causal=True,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.causal = causal

    def forward(self, x, encoder_output=None, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        att = None
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        # att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

class CrossAttention(nn.Module):
    def __init__(self,
                 condition_seq_len,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 seq_len=None, # the max length of sequence
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 causal=False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                                        .view(1, 1, seq_len, seq_len))

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        att = None
        
        y = y.transpose(1, 2).reshape(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        # att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att
class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 class_type='adalayernorm',
                 class_number=1000,
                 condition_seq_len=77,
                 n_embd=1024,
                 n_head=16,
                 seq_len=256,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU2',
                 attn_type='selfcross',
                 if_upsample=False,
                 condition_dim=1024,
                 diffusion_step=100,
                 timestep_type='adalayernorm',
                 mlp_type = 'fc',
                 ):
        super().__init__()
        self.if_upsample = if_upsample
        self.attn_type = attn_type

        if attn_type in ['selfcross', 'selfcondition', 'self']: 
            if 'adalayernorm' in timestep_type:
                self.ln1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")
        else:
            self.ln1 = nn.LayerNorm(n_embd)
        
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        # self.if_selfcross = False
        if attn_type in ['self', 'selfcondition']:
            self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                seq_len=seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
            if attn_type == 'selfcondition':
                if 'adalayernorm' in class_type:
                    self.ln2 = AdaLayerNorm(n_embd, class_number, class_type)
                else:
                    self.ln2 = AdaInsNorm(n_embd, class_number, class_type)
        elif attn_type == 'selfcross':
            self.self_attn = FullAttention(
                    n_embd=n_embd,
                    n_head=n_head,
                    seq_len=seq_len,
                    attn_pdrop=attn_pdrop, 
                    resid_pdrop=resid_pdrop,
                    )
            self.cross_attn = CrossAttention(
                    condition_seq_len,
                    n_embd=n_embd,
                    condition_embd=condition_dim,
                    n_head=n_head,
                    seq_len=seq_len,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    )
            if 'adalayernorm' in timestep_type:
                self.ln1_1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
                self.ln1_2 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")
        else:
            print("attn_type error")
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        if mlp_type == 'conv_mlp':
            self.mlp = Conv_MLP(n_embd, mlp_hidden_times, act, resid_pdrop)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )

        self.instancenorm = nn.InstanceNorm1d(n_embd)
    
    def _self_cross_blocks(self,x,timestep,cond,mask=None):
        a, att = self.cross_attn(self.ln1_1(x, timestep), cond, mask=mask)
        a = self.instancenorm(a)
        return a

    def forward(self, x,  timestep, label=None, mask=None):    
        if self.attn_type == "selfcross":
            #x = self.ln1(x, timestep)
            #a = self._self_cross_blocks(x, timestep, label, mask=mask)

            a, _ = self.cross_attn(self.ln1(x, timestep), label, mask=mask)
            x = x + a
            #a, attn = self.self_attn(self.ln2(x), None, mask=mask)
            a, attn = self.self_attn(self.ln1_1(x, timestep), label, mask=mask)
            x = x + a
        # elif self.attn_type == "selfcondition":
        #     a, att = self.attn(self.ln1(x, timestep), cond, mask=mask)
        #     x = x + a
        #     x = x + self.mlp(self.ln2(x, cond.long()))   # only one really use encoder_output
        #     return x, att
        # else:  # 'self'
        #     a, att = self.attn(self.ln1(x, timestep), cond, mask=mask)
        #     x = x + a 

        x = x + self.mlp(self.ln3(x))

        return x, None

class VLATransformer(nn.Module):
    "A transformer which incoporate language, video and actions for modeling"
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
            n_layer=12,
            mlp_ratio=4.0,
            hidden_size=256,
            n_head=8,
            train_device=None,
            prompt_trajectories=None,
            task_list=None,
            action_dim=8,
            max_ep_len=1000,
            patch_size=2,
            in_chans=4,
            act_cls=256,
            obs_cls=2048,
            attn_pdrop=0,
            resid_pdrop=0,
            mlp_hidden_times=4,
            block_activate='GELU2',
            attn_type='selfcross',
            content_spatial_size=[16,16], # H , W
            diffusion_step=100,
            timestep_type='adalayernorm',
            mlp_type="fc",
            embedding = None,
            pretrain = True,
            vqvae=None,
            vqvae_ckpt='./lightning_logs/version_40/checkpoints/last.ckpt',
            num_latents=2048,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.pretrain = pretrain
        self.dim = dim
        self.horizon = horizon*4
        self.hidden_size = hidden_size
        self.state_dim = transition_dim - 1
        self.action_dim = action_dim
        self.prompt_trajectories = prompt_trajectories
        self.task_list = task_list
        """embedding layers"""
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
        #self.embed_act = nn.Linear(self.action_dim, hidden_size)
        self.embed_act = nn.Embedding(self.act_cls, hidden_size)
        self.embed_traj = nn.Linear(dim, hidden_dim)
        
        all_attn_type = [attn_type] * n_layer
        self.state_seq_len = 4*24*24
        self.action_seq_len = self.horizon * self.action_dim
        self.traj_blocks = nn.Sequential(*[Block(
                n_embd=hidden_size,
                n_head=n_head,
                seq_len=self.state_seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                attn_type=all_attn_type[n],
                diffusion_step = diffusion_step,
                timestep_type = timestep_type,
                condition_dim = hidden_size,
                mlp_type = mlp_type,
        ) for n in range(n_layer)])
        self.act_blocks = nn.Sequential(*[Block(
                n_embd=hidden_size,
                n_head=n_head,
                seq_len=self.action_seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                attn_type=all_attn_type[n],
                diffusion_step = diffusion_step,
                timestep_type = timestep_type,
                condition_dim = hidden_size,
                mlp_type = mlp_type,
        ) for n in range(n_layer-4)])
        # final prediction head
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.embedding = embedding
        self.to_logits_act = nn.Sequential(
            nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, self.act_cls),
        )
        self.to_logits_traj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.obs_cls),
        )
        self.mask_dist = Bernoulli(probs=0.8)
        self.apply(self._init_weights)
        #self.load_vqvae(vqvae_ckpt)
        self.vqvae = vqvae
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.transformer.named_parameters()}# if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        print('CKPT: ',ckpt)
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt)
        self.vqvae.eval()
    # def forward(
    #         self, 
    #         input, 
    #         t,
    #         cond='txt'):
    def forward(self, x, time, cond, value, context_mask=None, x_condition=None, force=False, clf=False, mask=None):
        '''
            x : [ batch x horizon x transition ]
        '''
        '''
        x : batch * (4*horizon) * 24 * 24; cond:  batch * (2*4) * 24 * 24; mask: batch * (2*16) * 7(action_dim)
        '''
        traj_obs, act = split(x)
        batch_size, seq_length = x.shape[0], x.shape[1]
        #x = einops.rearrange(x, 'B (T H W) -> B T H W', T=4, H=24, W=24)
        with torch.no_grad():
            #print(traj_obs,traj_obs.min(),traj_obs.max())
            traj_obs = self.vqvae.codebook.dictionary_lookup(traj_obs.long())
            x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        traj_obs = shift_dim(traj_obs, -1, 1).flatten(2).transpose(1, 2) #B x L x D
        x_condition = shift_dim(x_condition, -1, 1).flatten(2).transpose(1, 2)
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        obs_embeddings = self.embed_history(x_condition)
        x_embed = self.embed_traj(traj_obs)
        cond_embd = torch.cat((prompt_embeddings, obs_embeddings), dim=1)
        # if not force:
        #     mask = self.mask_dist.sample(sample_shape=(cond_embd.shape[0], 1, 1)).to(x.device)
        # else:
        #     mask = 0 if clf else 1
        # cond_embd = cond_embd * mask
        for block_idx in range(len(self.traj_blocks)):   
            x_embed, att_weight = self.traj_blocks[block_idx](x_embed, time, cond_embd) # B x (Ld+Lt) x D, B x (Ld+Lt) x (Ld+Lt)
        #t = self.time_mlp(time).unsqueeze(1)
        if self.pretrain:
            #h = self.norm1(x_embed)
            #return self.to_logits_traj(h), None
            x_embed_ = x_embed.detach()
            #print(act)
            #act[:,:] = torch.tensor(2049).to(act.device).long()
            #x_act = self.embed_act(act)
            x_act = self.vqvae.codebook.dictionary_lookup(act.long())
            #print(act.shape,x_act.shape)
            for block_idx in range(len(self.act_blocks)):   
                x_act, att_weight = self.act_blocks[block_idx](x_act, time, x_embed_)
            # h_act = self.norm1(x_act)
            # h_traj = self.norm2(x_embed)
            h_act = x_act
            h_traj = x_embed
        return self.to_logits_traj(h_traj), self.to_logits_act(h_act)
def split(x):
        T, H, W = 4, 24, 24
        z_dim = T * H * W
        act_dim = 4*8
        z, a = x.split([z_dim, act_dim], dim=1) #TODO:action_dim
        z = einops.rearrange(z, 'B (T H W) -> B T H W', T=T, H=H, W=W)
        #a = einops.rearrange(a, 'B (L) -> B L', L=act_dim)
        return z, a