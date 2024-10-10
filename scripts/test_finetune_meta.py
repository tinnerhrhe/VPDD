import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import diffuser.utils as utils
from pathlib import Path
import numpy as np
from numpy.random import randn
import matplotlib as mpl
from scipy import stats
import torch
import einops
import math
import torch.nn.functional as F
import json
from scipy.spatial.transform import Rotation
from diffuser.utils.arrays import batch_to_device
from videogpt.utils import save_video_grid
##Environment##
from tqdm import tqdm
from einops import rearrange, repeat, reduce
# from rlbench import environment
# from rlbench.action_modes.action_mode import MoveArmThenGripper
# from rlbench.action_modes.arm_action_modes import *
# from rlbench.action_modes.gripper_action_modes import Discrete
# from rlbench.observation_config import ObservationConfig
# from rlbench.tasks import *
from helpers.clip.core.clip import build_model, load_clip, tokenize
from diffuser.utils import action_tokenizer
from PIL import Image
import gym
import imageio
import importlib 
import matplotlib.pyplot as plt

from metaworld.envs.mujoco.env_dict import MT50_V2, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
task_prompts = [
 'Dunk the basketball into the basket',
 'Grasp the puck from one bin and place it into another bin',
 'Press a button from the top',
 'Press a button',
 'Bypass a wall and press a button',
 'Push a button on the coffee machine',
 'Pull a mug from a coffee machine',
 'Push a mug under a coffee machine',
 'Rotate a dial 180 degrees',
 'pick a nut out of the a peg',
 'Close a door with a revolving joint',
 'Lock the door by rotating the lock clockwise',
 'Open a door with a revolving joint',
 'Unlock the door by rotating the lock counter-clockwise',
 'Insert the gripper into a hole',
 'Push and close a drawer',
 'Open a drawer',
 'Rotate the faucet counter-clockwise',
 'Rotate the faucet clockwise',
 'Press a handle down sideways',
 'Press a handle down',
 'Pull a handle up sideways',
 'Pull a handle up',
 'Pull a lever down 90 degrees',
 'Insert a peg sideways',
 'Pick a puck, bypass a wall and place the puck',
 'Pick up a puck from a hole',
 'reach a goal position',
 'Pull a puck to a goal',
 'Push the puck to a goal',
 'Pick and place a puck to a goal',
 'Slide a plate into a cabinet',
 'Slide a plate into a cabinet sideways',
 'Get a plate from the cabinet',
 'Get a plate from the cabinet sideways',
 'Kick a soccer into the goal',
 'Bypass a wall and push a puck to a goal',
 'pick and place a puck onto a shelf',
 'Sweep a puck into a hole',
 'Sweep a puck off the table',
 'Push and open a window',
 'Push and close a window',
 'Pick up a nut and place it onto a peg',
 'Bypass a wall and press a button from the top',
 'Hammer a screw on the wall',
 'Unplug a peg sideways',
 'Bypass a wall and reach a goal',
 'Grasp a stick and push a box using the stick',
 'Grasp a stick and pull a box with the stick',
 'Grasp the cover and close the box with it',
]

# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#
DTYPE = torch.float
DEVICE = 'cuda'
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
class Parser(utils.Parser):
    dataset: str = 'meta'
    config: str = 'config.locomotion'




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

def _norm_rgb(x):
    return (x.float() / 255.0) - 0.5
def create_env_and_policy(env_name, seed=None):
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + '-goal-observable']
    env = env_cls(seed=seed, render_mode='rgb_array')
    env.camera_name="corner2"
    #print(env.camera_name)
    # env.width=128
    # env.height=128
    #print(env.width)
    #print(env.resolution)
    #print(env.offscreen)
    #env.offscreen = False
    # def initialize(env, seed=None):
    #     if seed is not None:
    #         st0 = np.random.get_state()
    #         np.random.seed(seed)
    #     # super(type(env), env).__init__()
    #     # env._partially_observable = True
    #     env._freeze_rand_vec = False
    #     # env._set_task_called = True
    #     env.reset()
    #     env._freeze_rand_vec = True
    #     if seed is not None:
    #         env.seed(seed)
    #         np.random.set_state(st0)

    # initialize(env, seed=seed)

    

    return env
@torch.no_grad()
def evaluate(task):
    #import pdb; pdb.set_trace()
    max_episode_length = 200
    num_evals = 20
    for env_id, env_ in enumerate([task]):
        #imgs = []
        print(env_)
        
        #description = task_prompts[env_id]
        description = prompt_dict[task]
        tokens = tokenize(description).numpy()
        token_tensor = torch.from_numpy(tokens).to(args.device)
        lang_feats, lang_embs = clip_model.encode_text_with_embeddings(token_tensor)
        env_list = [create_env_and_policy(env_, seed=args.seed+idx) for idx in range(num_evals)]
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        for each in env_list:
            each.render()
        obs_list = [env.reset()[0][None] for env in env_list]
        condition = torch.zeros([num_evals, 4, 260, 260, 3], device=args.device)
        #import pdb;pdb.set_trace()
        for i, env in enumerate(env_list):
            condition[i,-1] = torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(args.device)
        
        #imgs = [np.rot90(env.render(), 2)[110:370,110:370] for env in env_list]
        # obs = np.concatenate(obs_list, axis=0)
        for step in tqdm(range(0, max_episode_length),desc="Episode timestep ", total=max_episode_length):
            x_condition = vqvae.encode(torch.stack([preprocess(condition[ind], 96) for ind in range(condition.shape[0])])).unsqueeze(1)
            #import pdb;pdb.set_trace()
            _, act = diffusion.sample_mask(num_evals, task=lang_feats.unsqueeze(0).repeat(num_evals,1,1).to(dtype=torch.float32), x_condition=x_condition.unsqueeze(0))
            
            trans = action_tokenizer.reconstruct(act[:,0,:3]-2048)
            grippers = gripper_list[(act[:,0,-1:]-2048).cpu().numpy()]
            action = np.concatenate([trans, grippers], axis=-1)
            for i in range(len(env_list)):
                obs, reward, done, truncate, info = env_list[i].step(action[i])
                condition[i] = torch.cat([condition[i,1:], torch.from_numpy(np.rot90(env_list[i].render(), 2)[110:370,110:370].copy()).to(args.device).unsqueeze(0)],dim=0)
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
        print(f"task:{env_},success rate:{sum(env_success_rate)/len(env_success_rate)}, mean episodic return:{sum(episode_rewards)/len(episode_rewards)}")
        with open(os.path.join(f'./eval_meta_{args.seed}.txt'), 'a+') as f:
            f.write(f"task:{env_},success rate:{sum(env_success_rate)/len(env_success_rate)}, mean episodic return:{sum(episode_rewards)/len(episode_rewards)}\n")
        #          f"std:{statistics.stdev(tmp)}")
        # for i in range(len(self.env_list)):
        #     tmp = []
        #     tmp_suc = 0
        #     for j in range(num_eval):
        #         tmp.append(episode_rewards[i+j*50])
        #         tmp_suc += env_success_rate[i+j*50]
        #     this_score = statistics.mean(tmp)
        #     success = tmp_suc / num_eval
        #     total_success += success
        #     score += this_score
        #     print(f"task:{self.task_list[i]},success rate:{success}, mean episodic return:{this_score}, "
        #           f"std:{statistics.stdev(tmp)}")
        # print('Total success rate:', total_success / len(self.envs))
    
if __name__ == '__main__':
    args = Parser().parse_args('plan')
    # -----------------------------------------------------------------------------#
    # ---------------------------------- loading ----------------------------------#
    # -----------------------------------------------------------------------------#
    # # load diffusion model and value function from disk
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    diffusion_experiment = utils.load_diffusion(
        args.loadbase, args.dataset, args.diffusion_loadpath,
        epoch=args.diffusion_epoch, device=args.device, seed=args.seed,
        )
    diffusion = diffusion_experiment.ema.to(args.device)
    #diffusion = diffusion_experiment.model.to(args.device)
    diffusion.clip_denoised = True
    dataset = diffusion_experiment.dataset
    vqvae = diffusion.model.traj_model.vqvae
    clip_model = dataset.clip_model
    gripper_list = np.array([-1.0, 0.0, 0.10000000149011612, 0.5, 0.6000000238418579, 0.6499999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0])
    meta_tasks = ['basketball-v2', 'bin-picking-v2',  'button-press-topdown-v2',
 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2',
'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2',
'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2',
 'faucet-close-v2',  'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2',
 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2',
 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2',
 'plate-slide-back-side-v2',  'soccer-v2',
 'push-wall-v2',  'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2','assembly-v2',
 'button-press-topdown-wall-v2','hammer-v2','peg-unplug-side-v2',
                               'reach-wall-v2', 'stick-push-v2', 'stick-pull-v2', 'box-close-v2']
    prompt_dict = {}
    for i in range(50):
        prompt_dict[meta_tasks[i]]=task_prompts[i]
    # np.save('./meta_prompts.npy',prompt_dict)
    #prompt_dict = np.load('./meta_prompts.npy',allow_pickle=True)
    action_tokenizer = dataset.discretizer
    #for task in meta_tasks:
    #    evaluate(task)
    evaluate(args.meta_task)

