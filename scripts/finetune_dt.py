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
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset:str = 'meta'
    config: str = 'config.locomotion'

args = Parser().parse_args('diffusion')
#'''
if args.single:
    args.tasks = ['close_jar']
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#prompt_trajectories = [np.load(f"./metaworld_prompts/{task_list[ind]}_prompt.npy", allow_pickle=True) for ind in range(len(task_list))]
dataset_config = utils.Config(
    args.loader,
    tasks = args.meta_tasks,
    savepath=(args.savepath, 'dataset_config.pkl'),
    data_folder = args.data_folder,
    sequence_length = 4, #args.sequence_length, #TODO
    devices = args.device,
    horizon = args.horizon,
)

#render_config = utils.Config(
#    args.renderer,
#    savepath=(args.savepath, 'render_config.pkl'),
#    env=args.dataset,
#)

dataset = dataset_config()

#"""
#renderer = render_config()
#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#
#Batch_size x 4 x 24 x 24

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=4,
    transition_dim=4,# + action_dim,# + reward_dim,
    cond_dim=4,
    num_tasks=50,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
    train_device=args.device,
    #prompt_trajectories=prompt_trajectories,
    verbose=False,
    task_list=args.meta_tasks,
    action_dim=4,
    vqvae=dataset.vqvae,
)
diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=4,
    observation_dim=4,
    action_dim=4,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = utils.Config(
    utils.DTTrainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    envs=args.meta_tasks,
    task_list=args.meta_tasks,
    is_unet=False,
    trainer_device=args.device,
    horizon=4,
)


#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)
renderer=None

trainer = trainer_config(diffusion, dataset, renderer)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)
# print('Testing forward...', end=' ', flush=True)
# batch = utils.batchify(dataset[0])
# #loss, _ = diffusion.loss(*batch, device=args.device)
# #loss.backward()
# print('✓')

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
#args.n_train_steps = 5e4
#args.n_steps_per_epoch = 1000
n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)
#"""
"""
nohup python -u scripts/train_diffusion_meta.py --model models.Tasksmeta --diffusion models.GaussianActDiffusion --loss_type statehuber --loader datasets.RTGActDataset --device cuda:1 --batch_size 8 --discount 1.0 --horizon 16 
"""