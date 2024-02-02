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
from diffuser.utils.distributed.launch import launch
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
# environment variables
NODE_RANK = os.environ['AZ_BATCHAI_TASK_INDEX'] if 'AZ_BATCHAI_TASK_INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = os.environ['AZ_BATCH_MASTER_NODE'].split(':') if 'AZ_BATCH_MASTER_NODE' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#
class Parser(utils.Parser):
        dataset:str = 'test'
        config: str = 'config.locomotion'
def main_worker(local_rank, args):
    #prompt_trajectories = [np.load(f"./metaworld_prompts/{task_list[ind]}_prompt.npy", allow_pickle=True) for ind in range(len(task_list))]
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node
    args.distributed = args.world_size > 1
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        data_folder = args.data_folder,
        sequence_length = args.sequence_length,
        devices = args.device,
        horizon = args.horizon,
        )

    dataset = dataset_config()
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=24,
        cond_dim=512,
        num_tasks=50,
        dim_mults=args.dim_mults,
        attention=args.attention,
        device=args.device,
        train_device=args.device,
        verbose=False,
        action_dim=8,
        )
    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon, #TODO
        observation_dim=24*24,
        action_dim=8,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        ## loss weighting
        action_weight=args.action_weight,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        #device=args.device,
        )
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
    gpuid=local_rank,
    )
    model = model_config()
    diffusion = diffusion_config(model)
    renderer=None
    trainer = trainer_config(diffusion, dataset, renderer)
    
    utils.report_parameters(model)

    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} | {args.savepath}')
        trainer.train(n_train_steps=args.n_steps_per_epoch)
def main():
    args = Parser().parse_args('diffusion')
    #Distributional Training
    args.dist_url = DIST_URL
    args.node_rank = NODE_RANK
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.num_node
    launch(main_worker, args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args=(args,))
#"""
"""
nohup python -u scripts/train_diffusion_meta.py --model models.Tasksmeta --diffusion models.GaussianActDiffusion --loss_type statehuber --loader datasets.RTGActDataset --device cuda:1 --batch_size 8 --discount 1.0 --horizon 16 
"""
if __name__ == '__main__':
    main()
    