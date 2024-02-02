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
import torch.nn.functional as F
from torchvision.utils import save_image
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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args = Parser().parse_args('plan')

# -----------------------------------------------------------------------------#
# ---------------------------------- loading ----------------------------------#
# -----------------------------------------------------------------------------#

# load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, device=args.device, seed=args.seed,
)
#diffusion_experiment = utils.load_diffusion(
#    "/mnt/petrelfs/hehaoran","MTdiffuser_remote/logs_",
#    epoch=args.diffusion_epoch, device=args.device, seed=args.seed,
#)
## ensure that the diffusion model and value function are compatible with each other
# utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema.to(args.device)
diffusion.clip_denoised = True
dataset = diffusion_experiment.dataset
## initialize value guide
# value_function = value_experiment.ema
def evaluate(ind):
    # batch = next(iter(diffusion_experiment.renderer))
    # batch = batch_to_device(batch,args.device)
    # loss, info = diffusion.loss(*batch)
    
    #print(dataset.indices[46000])
    #import pdb;pdb.set_trace()
    #diffuion = diffusion.to('cpu')
    #vqvae = diffusion.model.traj_model.vqvae

    vqvae = dataset.vqvae
    desc=dataset.task_desc[dataset.indices[ind+10][0]//20]
    #desc=dataset.task_desc[dataset.indices[ind][0]]
    #import pdb;pdb.set_trace()
    for k in range(5):
        data = dataset[ind+k*10]
        obs, _ = diffusion.sample_mask(2, task=to_torch(data.task).unsqueeze(0), x_condition=to_torch(data.conditions).unsqueeze(0))
        video_recon = vqvae.decode(obs)
        samples = torch.clamp(video_recon, -0.5, 0.5) + 0.5
        sample = samples[0,:,0]#.copy()
        img = F.interpolate(sample.unsqueeze(0), size=(128,128),mode='bilinear',align_corners=False)
        save_image(img, f'./images/{desc}_{k*10}.jpg')
        #print(samples.shape)
        # for i in range(4):
        #     sample = samples[0,:,i]#.copy()
        #     img = F.interpolate(sample.unsqueeze(0), size=(128,128),mode='bilinear',align_corners=False)
        #     save_image(img, f'./imgs/{desc}_front_{k*10}_{i}.jpg')
        # for i in range(4):
        #     sample = samples[1,:,i]#.copy()
        #     img = F.interpolate(sample.unsqueeze(0), size=(128,128),mode='bilinear',align_corners=False)
        #     save_image(img, f'./imgs/{desc}_left_{k*10}_{i}.jpg')
        # for i in range(4):
        #     sample = samples[2,:,i]#.copy()
        #     img = F.interpolate(sample.unsqueeze(0), size=(128,128),mode='bilinear',align_corners=False)
        #     save_image(img, f'./imgs/{desc}_right_{k*10}_{i}.jpg')
        # for i in range(4):
        #     sample = samples[3,:,i]#.copy()
        #     img = F.interpolate(sample.unsqueeze(0), size=(128,128),mode='bilinear',align_corners=False)
        #     save_image(img, f'./imgs/{desc}_wrist_{k*10}_{i}.jpg')
    
    
if __name__ == '__main__':
    vqvae = dataset.vqvae
    data=dataset[4003]
    #import pdb;pdb.set_trace()
    evaluate(46800)
    import pdb;pdb.set_trace()

"""
python scripts/diff_test.py --diffusion_loadpath -Mar15_11-36-43 --diffusion_epoch 0_2220.887145190953_1 --device cuda:1
"""
