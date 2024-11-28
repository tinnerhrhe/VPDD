# :rocket: VPDD | NeurIPS 2024

## Learning an Actionable Discrete Diffusion Policy via Large-Scale Actionless Video Pre-Training

This is the official code for the paper "Learning an Actionable Discrete Diffusion Policy via Large-Scale Actionless Video Pre-Training".
We introduce a novel framework that leverages a unified discrete diffusion to combine generative pre-training on human videos and policy fine-tuning on a small number of action-labeled robot videos. We aim to incorporate foresight from predicted videos to facilitate efficient policy learning.

📝 [Paper](https://arxiv.org/abs/2402.14407) \|  [中文blog@知乎](https://zhuanlan.zhihu.com/p/684830185) \| [公众号@量子位](https://mp.weixin.qq.com/s/bFVwWpjFQpTTWkbpaEqYCQ)
## Environment Configurations
```
conda env create -f environment.yml
conda activate VPDD
```
## Dataset
- For experiments on RLBench, you can use [pre-generated dataset](https://drive.google.com/drive/folders/0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE?resourcekey=0-jRw5RaXEYRLe2W6aNrNFEQ) provided by [PerAct](https://github.com/peract/peract).
- For experiments on MetaWorld, we use the script policy provided in [MetaWorld](https://github.com/Farama-Foundation/Metaworld) to collect image-based data. You can refer to [generate_metaworld_dataset.py](https://github.com/pairlab/QueST/blob/main/scripts/generate_metaworld_dataset.py) for implementation.

## Model
The pre-trained VQ-VAE models and discrete diffusion models are available at https://huggingface.co/haoranhe/VPDD-pretrain. You can download them and change the `path` in the corresponding code.
## Pre-training

We first train a VQ-VAE to learn a unified discrete latent codebook:

`torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=4 --nnodes=$WORLD_SIZE --node_rank=$RANK scripts/train_vqvae.py --gpus=4 --max_epoch=10 --resolution 96 --sequence_length 8 --batch_size 32`

We then pre-train VPDD on Meta-World:

`python scripts/pretrain_meta.py --seed 1 --model models.VideoDiffuserModel --diffusion models.GaussianVideoDiffusion --loss_type video --device cuda:0 --batch_size 10 --loader datasets.MetaDataset --act_classes 48`

or on RLBench which requires multi-view videos prediction:

`python scripts/pretrain_video_diff.py --seed 1 --model models.VideoDiffuserModel --diffusion models.MultiviewVideoDiffusion --loss_type video --device cuda:0 --batch_size 3 --loader datasets.MultiViewDataset --act_classes 360 --n_diffusion_steps 100`
## Fine-Tuning
After pre-training, we fine-tune VPDD with a limited set of robot data:

`python scripts/pretrain_meta.py --seed 1 --model models.VideoDiffuserModel --diffusion models.GaussianVideoDiffusion --loss_type video --device cuda:0 --batch_size 1 --loader datasets.MetaFinetuneDataset --pretrain False`

`python scripts/pretrain_video_diff.py --seed 1 --model models.VideoDiffuserModel --diffusion models.MultiviewVideoDiffusion --loss_type video --device cuda:0 --batch_size 10 --loader datasets.MultiviewFinetuneDataset --pretrain False --act_classes 360`


## Acknowledgment 
Our code for VPDD is partially based on the following awesome projects:
- MTDiff from https://github.com/tinnerhrhe/MTDiff/
- UniD3 from https://github.com/mhh0318/UniD3
## Citation
```bibtex
@inproceedings{
he2024learning,
title={Learning an Actionable Discrete Diffusion Policy via Large-Scale Actionless Video Pre-Training},
author={He, Haoran and Bai, Chenjia and Pan, Ling and Zhang, Weinan and Zhao, Bin and Li, Xuelong},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tinnerhrhe/VPDD&type=Date)](https://star-history.com/#hpcaitech/Open-Sora&Date)
