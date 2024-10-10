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

diffusion = diffusion_experiment.ema.to(args.device)
diffusion.clip_denoised = True
dataset = diffusion_experiment.dataset
## initialize value guide
# value_function = value_experiment.ema
def evaluate():
    # batch = next(iter(diffusion_experiment.renderer))
    # batch = batch_to_device(batch,args.device)
    # loss, info = diffusion.loss(*batch)
    data = dataset[47000]
    #print(dataset.indices[46000])
    #import pdb;pdb.set_trace()
    #diffuion = diffusion.to('cpu')
    #vqvae = diffusion.model.traj_model.vqvae

    vqvae = dataset.vqvae
    #import pdb;pdb.set_trace()
    hist = to_torch(data.conditions).unsqueeze(0).long()
    hist = einops.rearrange(hist, 'i j h b k c -> (i j) (h b) k c')
    print(hist.shape[0])
    video_recon = vqvae.decode(hist)
    samples = torch.clamp(video_recon, -0.5, 0.5) + 0.5
    save_video_grid(samples.detach(), './samples_origin.mp4')
    #import pdb;pdb.set_trace()
    obs = diffusion.sample_mask(2, task=to_torch(data.task).unsqueeze(0), x_condition=to_torch(data.conditions))#obs, _ = diffusion.sample_mask(2, task=to_torch(data.task).unsqueeze(0), x_condition=to_torch(data.conditions).unsqueeze(0))
    print(obs.shape)
    hist = to_torch(data.trajectories['obs']).unsqueeze(0).long()
    hist = einops.rearrange(hist, 'i j h b k c -> (i j) (h b) k c')
    gt = hist
    print("Ground Truth:", gt)
    print("Denoised:", obs)
    print("Error:", (obs-gt)**2/np.prod(obs.shape))
    video_recon = vqvae.decode(hist)
    samples = torch.clamp(video_recon, -0.5, 0.5) + 0.5
    save_video_grid(samples.detach(), './samples_origin_traj.mp4')
    #import pdb; pdb.set_trace()
    video_recon = vqvae.decode(obs)
    samples = torch.clamp(video_recon, -0.5, 0.5) + 0.5
    save_video_grid(samples.detach(), './samples_pretrain.mp4')
if __name__ == '__main__':
    evaluate()

