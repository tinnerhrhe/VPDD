import os
import argparse
import torch
import os
import random
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from videogpt import VideoData, VideoGPT, load_videogpt, VQVAE
from videogpt.utils import save_video_grid
from torchvision.utils import save_image
"""
python dataset_generator.py --save_path=/mnt/petrelfs/hehaoran/video_diff/data/RLBench --image_size=128,128 --renderer=opengl --episodes_per_task=100 --all_variations=True --processes=10
"""
#40,52,71,83,90
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./lightning_logs/version_90/checkpoints/last.ckpt')
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_path', type=str, default='/mnt/data/optimal/hehaoran/data')
parser.add_argument('--sequence_length', type=int, default=4)
parser.add_argument('--resolution', type=int, default=96)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--n_cond_frames', type=int, default=1)
parser.add_argument('--class_cond', action='store_true')
parser.add_argument('--hidden_dim', type=int, default=576)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--layers', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--attn_type', type=str, default='full',
                            choices=['full', 'sparse'])
parser.add_argument('--attn_dropout', type=float, default=0.3)
args = parser.parse_args()
n = args.n
#args.ckpt = 'kinetics_stride4x4x4'
args.vqvae = args.ckpt
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

#args.class_cond = False
from videogpt.download import load_vqvae
if not os.path.exists(args.vqvae):
        vqvae = load_vqvae(args.vqvae)
else:
        vqvae =  VQVAE.load_from_checkpoint(args.vqvae)
#vqvae.codebook._need_init = False
vqvae.eval()
#args.class_cond_dim = 2
#vqvae = load_vqvae('kinetics_stride4x4x4')
# if not os.path.exists(args.ckpt):
#     gpt = load_videogpt(args.ckpt)
# else:
#     gpt = VideoGPT.load_from_checkpoint(args.ckpt)
# gpt = gpt.cuda()
# gpt.eval()
# args = gpt.hparams['args']
#gpt = VideoGPT(args)
args.batch_size = 8
data = VideoData(args)
loader = data.test_dataloader()
print("loading dataloader...")
step=0
for batch in iter(loader):
        batch = batch['video']
        images = torch.clamp(batch,-0.5,0.5)+0.5
        for j in range(4):
                image = images[:,:,j]
                for i in range(image.shape[0]):
                        save_image(image[i:i+1],f'./image_{i}_{j}.jpg') 
        encodings, embeddings = vqvae.encode(batch,include_embeddings=True)
        video_recon = vqvae.decode(encodings)
        samples = torch.clamp(video_recon, -0.5, 0.5) + 0.5
        for j in range(4):
                image = samples[:,:,j]
                for i in range(image.shape[0]):
                        save_image(image[i:i+1],f'./vqave_image_{i}_{j}.jpg') 
        import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        # batch_rec = torch.clamp(batch, -0.5, 0.5) + 0.5
        # save_video_grid(batch_rec, f'samples/batch_v1{step}.mp4')
        # encodings, embeddings = vqvae.encode(batch,include_embeddings=True)
        # print(encodings.shape)
        # #print(encodings.shape, embeddings.shape)
        # video_recon = vqvae.decode(encodings)
        # samples = torch.clamp(video_recon, -0.5, 0.5) + 0.5
        # save_video_grid(samples.detach(), f'samples/samples_v1_{step}.mp4')
        # step+=1