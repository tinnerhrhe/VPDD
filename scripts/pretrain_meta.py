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

dataset_config = utils.Config(
    args.loader,
    tasks = args.meta_tasks,
    savepath=(args.savepath, 'dataset_config.pkl'),
    data_folder = args.data_folder,
    sequence_length = 4, #args.sequence_length, #TODO
    devices = args.device,
    horizon = args.horizon,
    num_demos=args.num_demos,
)


dataset = dataset_config()



model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=24,
    obs_cls=2048,
    act_cls=args.act_classes,
    cond_dim=512,
    num_tasks=50,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
    train_device=args.device,
    verbose=False,
    action_dim=4,
    vqvae=dataset.vqvae,
    pretrain=args.pretrain,
    meta=True,
)
diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon, #TODO
    observation_dim=24*24,
    obs_classes=2048,
    act_classes=args.act_classes,
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
    pretrain=args.pretrain,
    focal=args.focal,
    force=args.force,
)
if args.pretrain and args.concat:
     dataset = dataset()
trainer_config = utils.Config(
    utils.MetaworldTrainer,
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
    trainer_device=args.device,
    horizon=args.horizon,
    distributed=False,
    pretrain=args.pretrain,
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
