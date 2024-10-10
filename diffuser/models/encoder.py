from .utils import *
from videogpt.utils import shift_dim
from .helpers import (
    SinusoidalPosEmb,
)
class encoder(nn.Module):
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
        self.position_emb = nn.Parameter(torch.zeros(1, 1+2*12*12*8, hidden_size)) #T, language, history, pred_act, pred_traj
        """ Perceiver Transformer """
         # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

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
        # self.decoder_cross_attn = PreNorm(hidden_dim, PerAttention(hidden_dim,
        #                                                         hidden_dim,
        #                                                         heads=1,
        #                                                         dim_head=64,
        #                                                         dropout=0.0),
        #                                   context_dim=hidden_dim)
        self.iterations = 1
    def forward(self, cond, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        with torch.no_grad():
            x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        
        x_condition = shift_dim(x_condition, -1, 1).flatten(2).transpose(1, 2)
        batch_size, seq_length = x_condition.shape[0], x_condition.shape[1]
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        obs_embeddings = self.embed_history(x_condition)
        stacked_inputs = torch.cat((prompt_embeddings, obs_embeddings), dim=1)
        #print(prompt_embeddings.shape, stacked_inputs.shape)
        stacked_inputs = prompt_embeddings * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
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
        # latents = self.decoder_cross_attn(stacked_inputs, context=x)
        # x = self.pos_drop(x)

        # predicted_token_traj = latents[:, -2*12*12*8:, :]
        # predicted_traj = self.predict_traj(predicted_token_traj)
        # return predicted_traj, predicted_token_traj
        return x 
class encoder_v1(nn.Module):
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

        self.position_emb = nn.Parameter(torch.zeros(1, 1+2*12*12*8, hidden_size)) #T, language, history, pred_act, pred_traj
        """ Perceiver Transformer """
         # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

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
                                                                dim_head=64,
                                                                dropout=0.0),
                                          context_dim=hidden_dim)
        self.iterations = 1
    def forward(self, cond, context_mask=None, x_condition=None, force=False, return_cond=False, flag=False, attention_mask=None, pretrain=True, mask=None, clf=None):
        with torch.no_grad():
            x_condition = self.vqvae.codebook.dictionary_lookup(x_condition.long())
        
        x_condition = shift_dim(x_condition, -1, 1).flatten(2).transpose(1, 2)
        batch_size, seq_length = x_condition.shape[0], x_condition.shape[1]
        prompt_embeddings = self.prompt_embed(cond).unsqueeze(1)
        obs_embeddings = self.embed_history(x_condition)
        stacked_inputs = torch.cat((prompt_embeddings, obs_embeddings), dim=1)
        #print(prompt_embeddings.shape, stacked_inputs.shape)
        stacked_inputs = prompt_embeddings * stacked_inputs + self.position_emb[:, :stacked_inputs.shape[1], :]
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

        predicted_token_traj = latents[:, -24*24:, :]
        #predicted_traj = self.predict_traj(predicted_token_traj)
        return predicted_token_traj