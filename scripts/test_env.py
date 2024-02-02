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
from diffuser.utils.arrays import batch_to_device
from videogpt.utils import save_video_grid
##Environment##
from rlbench import environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import OpenDrawer
# -----------------------------------------------------------------------------#
# ----------------------------------- setup -----------------------------------#
# -----------------------------------------------------------------------------#
ASSET_DIR = "./data"
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
    dataset: str = 'test'
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
def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image
def _resize_if_needed(image, size):
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
def get_task(task_class, arm_action_mode, obs_config=None):
        if obs_config is None:
            obs_config = ObservationConfig()
            obs_config.set_all(False)
            obs_config.set_all_low_dim(True)
            obs_config.front_camera.rgb = True
            #obs_config.left_shoulder_camera.point_cloud = True
        mode = MoveArmThenGripper(arm_action_mode, Discrete())
        env = environment.Environment(
            mode, ASSET_DIR, obs_config, headless=True)
        env.launch()
        return env.get_task(task_class)
def evaluate():
    #import pdb; pdb.set_trace()
    task = get_task(
            OpenDrawer, EndEffectorPoseViaPlanning())
    done = False
    history = torch.zeros([16, 128, 128], device=args.device)
    while not done:
        desc, obs = task.reset()
        condition = torch.cat([history[1:], to_torch(_resize_if_needed(Image.fromarray(obs.front_rgb, 'RGB'),(128,128)).convert('RGB'), device=args.device).unsqueeze(0)],dim=0)
        self.vqvae.encode(preprocesss(_robot_obs, resolution).unsqueeze(0))

   
if __name__ == '__main__':
    args = Parser().parse_args('plan')
    # -----------------------------------------------------------------------------#
    # ---------------------------------- loading ----------------------------------#
    # -----------------------------------------------------------------------------#
    # # load diffusion model and value function from disk
    # diffusion_experiment = utils.load_diffusion(
    #     args.loadbase, args.dataset, args.diffusion_loadpath,
    #     epoch=args.diffusion_epoch, device=args.device, seed=args.seed,
    #     )
    # diffusion = diffusion_experiment.ema.to(args.device)
    # diffusion.clip_denoised = True
    # dataset = diffusion_experiment.dataset
    evaluate()

"""
python scripts/diff_test.py --diffusion_loadpath -Mar15_11-36-43 --diffusion_epoch 0_2220.887145190953_1 --device cuda:1
"""
