import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from videogpt import VQVAE, VideoData
import torch

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--n_codes', type=int, default=2048)
    parser.add_argument('--ckpt', type=str, default='./lightning_logs/version_83/checkpoints/last.ckpt')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)#16
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--save_path", default='./results', type=str, help="path to save checkpoints")
    parser.add_argument("--save_topk", default=5, type=int, help="save topk checkpoint")
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()
    model = VQVAE(args)
    #from videogpt.download import load_vqvae
    
    #vqvae =  VQVAE.load_from_checkpoint(args.ckpt)
    model = model.load_from_checkpoint(args.ckpt)
    logger = TensorBoardLogger(
        save_dir=args.save_path,
        name='log'
    )
    callbacks = []
    callbacks.append(ModelCheckpoint(filename=os.path.join(args.save_path, '{val/recon_loss:.4f}'),
        save_top_k=args.save_topk,
        save_last=True,monitor='val/recon_loss', mode='min'))

    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(strategy="ddp", accelerator="gpu", gpus=args.gpus, devices=-1)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=200000000, **kwargs)
    print("Executing training!")
    trainer.fit(model, data)


if __name__ == '__main__':
    main()

