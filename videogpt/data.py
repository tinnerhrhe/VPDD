import os
import os.path as osp
import math
import random
import pickle
import warnings

import glob
import h5py
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.datasets.video_utils import VideoClips
import pytorch_lightning as pl


class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        # self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'RLBench')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])  # [:4]
        print(len(files))
        # print(folder)
        exts = ['png']
        cams = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
        self.files_obs = []
        li = os.listdir(folder_robot)
        print(li)
        self.files_wrist = []
        for task in li:
            for x in range(10):
                # self.files_wrist.append(sum([sorted(glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "wrist_rgb", f'*.{ext}'), recursive=True),key=lambda name:int(name.split('/')[-1][:-4])) for ext in exts], []))
                self.files_obs.append(sum([sorted(glob.glob(
                    osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "front_rgb", f'*.{ext}'),
                    recursive=True), key=lambda name: int(name.split('/')[-1][:-4])) for ext in exts], []))
                self.files_obs.append(sum([sorted(glob.glob(
                    osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "left_shoulder_rgb",
                             f'*.{ext}'), recursive=True), key=lambda name: int(name.split('/')[-1][:-4])) for ext in
                                           exts], []))
                self.files_obs.append(sum([sorted(glob.glob(
                    osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "right_shoulder_rgb",
                             f'*.{ext}'), recursive=True), key=lambda name: int(name.split('/')[-1][:-4])) for ext in
                                           exts], []))
                self.files_obs.append(sum([sorted(glob.glob(
                    osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "wrist_rgb", f'*.{ext}'),
                    recursive=True), key=lambda name: int(name.split('/')[-1][:-4])) for ext in exts], []))

        self.indices = []
        for i, term in enumerate(self.files_obs):
            # for i, term in enumerate(self.files_right):
            max_start = len(term)  # min(len(term) - 1, len(term) - self.sequence_length)
            for start in range(max_start - self.sequence_length + 1):
                self.indices.append((i, start))
                # f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "variation_descriptions.pkl"), 'rb')
                # self.task_desc.append(pickle.load(f))

        clips = VideoClips(files, sequence_length, frames_between_clips=8, num_workers=32)

        self._clips = clips
        self.classes = ['human_hands_interaction', 'robot_arm_manipulation']
        print("WILD_NUMS:", self._clips.num_clips())
        print("OBS_NUMS:", len(self.files_obs))


    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        # return self._clips.num_clips()+len(self.indices)# - self.sequence_length#self.len
        return len(self.indices)

    def __getitem__(self, idx):
        resolution = self.resolution

        path_ind, start = self.indices[idx]
        _robot_obs = np.array(
            [np.array(Image.open(id)) for id in self.files_obs[path_ind][start:start + self.sequence_length]])
        video = torch.from_numpy(_robot_obs).to(dtype=torch.float32)
        label = self.classes[1]

        return dict(video=preprocess(video, resolution), label=label)


class MultiviewDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        # self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'RLBench')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])  # [:4]
        print(len(files))
        # print(folder)
        exts = ['png']
        cams = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
        self.files_obs = []
        li = os.listdir(folder_robot)
        print(li)
        self.task_desc = []
        self.files_obs = []
        self.files_left = []
        self.files_right = []
        self.files_wrist = []
        self.key_points = []
        for task in li:
            for x in range(10):
                self.files_obs.append(sum([sorted(glob.glob(
                    osp.join('./data/RLBench', 'put_item_in_drawer', "all_variations",
                             "episodes", f'episode{0}', "left_shoulder_rgb", f'*.png'), recursive=True),
                                                  key=lambda name: int(name.split('/')[-1][:-4])) for ext in exts], []))
                self.files_left.append(sum([sorted(glob.glob(
                    osp.join('./data/RLBench', 'put_item_in_drawer', "all_variations",
                             "episodes", f'episode{0}', "left_shoulder_rgb", f'*.png'), recursive=True),
                                                   key=lambda name: int(name.split('/')[-1][:-4])) for ext in exts],
                                           []))
                self.files_right.append(sum([sorted(glob.glob(
                    osp.join('./data/RLBench', 'put_item_in_drawer', "all_variations",
                             "episodes", f'episode{0}', "left_shoulder_rgb", f'*.png'), recursive=True),
                                                    key=lambda name: int(name.split('/')[-1][:-4])) for ext in exts],
                                            []))
                self.files_wrist.append(sum([sorted(glob.glob(
                    osp.join('./data/RLBench', 'put_item_in_drawer', "all_variations",
                             "episodes", f'episode{0}', "left_shoulder_rgb", f'*.png'), recursive=True),
                                                    key=lambda name: int(name.split('/')[-1][:-4])) for ext in exts],
                                            []))
                f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}',
                                  "variation_descriptions.pkl"), 'rb')
                description = pickle.load(f)[0]  # REVIEW
                if len(description) > 77: description = description[:77]
                self.task_desc.append(description)
                f_low_dim = open(
                    osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "low_dim_obs.pkl"), 'rb')
                _low_dim = pickle.load(f_low_dim)
                self.key_points.append(keypoint_discovery(_low_dim))

        self.indices = []
        for i, term in enumerate(self.files_obs):
            # for i, term in enumerate(self.files_right):
            # max_start = min(len(term) - 1, len(term) - 1)
            for start in range(len(term)):
                self.indices.append((i, start))
                # f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "variation_descriptions.pkl"), 'rb')
                # self.task_desc.append(pickle.load(f))

        clips = VideoClips(files, sequence_length, frames_between_clips=8, num_workers=32)

        self._clips = clips
        self.classes = ['human_hands_interaction', 'robot_arm_manipulation']
        print("WILD_NUMS:", self._clips.num_clips())
        print("OBS_NUMS:", len(self.files_obs))

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips() + len(self.indices)  # - self.sequence_length#self.len

    def __getitem__(self, idx):
        resolution = self.resolution
        # idx = self._clips.num_clips() + len(self.indices)-100
        if idx < self._clips.num_clips():
            # video, _, _, idx = self._clips.get_clip(idx)
            # label = self.classes[0]
            path_ind, start = self.indices[idx % 100]
            _robot_obs = np.array(
                [np.array(Image.open(id)) for id in self.files_obs[path_ind][start:start + self.sequence_length]])
            video = torch.from_numpy(_robot_obs).to(dtype=torch.float32)
            label = self.classes[1]
        else:
            path_ind, start = self.indices[idx - self._clips.num_clips()]
            _robot_obs = np.array(
                [np.array(Image.open(id)) for id in self.files_obs[path_ind][start:start + self.sequence_length]])
            video = torch.from_numpy(_robot_obs).to(dtype=torch.float32)
            label = self.classes[1]

        return dict(video=preprocess(video, resolution), label=label)


class MetaDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        # self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'metaworld_image')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])  # [:4]
        print(len(files))
        # print(folder)
        exts = ['png']
        cams = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
        self.files_obs = []
        li = os.listdir(folder_robot)
        print(li)
        for task in li:
            data = pickle.load(open(osp.join(folder_robot, task), 'rb'))
            # print(len(data))
            data = data[:20]
            # import pdb; pdb.set_trace()
            self.files_obs.append([data[i][j][0] for i in range(len(data)) for j in range(len(data[0]))])
            # import pdb; pdb.set_trace()

        self.indices = []
        for i, term in enumerate(self.files_obs):
            max_start = len(term) - self.sequence_length
            for start in range(max_start):
                self.indices.append((i, start))
                # f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "variation_descriptions.pkl"), 'rb')
                # self.task_desc.append(pickle.load(f))

        clips = VideoClips(files, sequence_length, frames_between_clips=8, num_workers=32)
        # if not osp.exists(cache_file):
        #     clips = VideoClips(files, sequence_length, frames_between_clips=8, num_workers=32)
        #     pickle.dump(clips.metadata, open(cache_file, 'wb'))
        # else:
        #     metadata = pickle.load(open(cache_file, 'rb'))
        #     clips = VideoClips(files, sequence_length,
        #                        _precomputed_metadata=metadata)
        self._clips = clips
        self.classes = ['human_hands_interaction', 'robot_arm_manipulation']
        print("WILD_NUMS:", self._clips.num_clips())
        print("OBS_NUMS:", len(self.files_obs))


    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips() // 2 + len(self.indices)  # - self.sequence_length#self.len

    def __getitem__(self, idx):
        resolution = self.resolution
        # idx = self._clips.num_clips() + len(self.indices)-100
        if idx < self._clips.num_clips() // 2:
            video, _, _, idx = self._clips.get_clip(idx + 2500)
            label = self.classes[0]
            # path_ind, start = self.indices[idx%100]
            # _robot_obs = np.array(self.files_obs[path_ind][start:start + self.sequence_length])
            # video = torch.from_numpy(_robot_obs).to(dtype=torch.float32)
            # label = self.classes[1]
        else:
            path_ind, start = self.indices[idx - self._clips.num_clips() // 2]
            _robot_obs = np.array(self.files_obs[path_ind][start:start + self.sequence_length])
            video = torch.from_numpy(_robot_obs).to(dtype=torch.float32)
            label = self.classes[1]

        return dict(video=preprocess(video, resolution), label=label)


class RobotDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png', 'jpg']

    def __init__(self, data_folder, sequence_length, train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        folder_robot = osp.join(data_folder, 'RLBench')
        li = os.listdir(folder_robot)
        print(li)
        exts = ['png']
        self.descs = []
        self.files_obs = []
        for task in li:
            for x in range(10):
                self.files_obs.append(sum([glob.glob(
                    osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "front_rgb", f'*.{ext}'),
                    recursive=True) for ext in exts], []))
                f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}',
                                  "variation_descriptions.pkl"), 'rb')
                description = pickle.load(f)[0]  # REVIEW
                self.descs.append(description)
        self.indices = []
        for i, term in enumerate(self.files_obs):
            max_start = min(len(term) - 1, len(term) - self.sequence_length)
            for start in range(max_start):
                self.indices.append((i, start))
        self.classes = ['robot manipulation']


    @property
    def n_classes(self):
        return len(self.descs)

    def __len__(self):
        return len(self.indices)  # len(self.obs) - self.sequence_length#self.len

    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
        _robot_obs = np.array(
            [np.array(Image.open(id)) for id in self.files_obs[path_ind][start:start + self.sequence_length]])
        video = torch.from_numpy(_robot_obs).to(dtype=torch.float32)
        label = self.descs[path_ind]
        resolution = self.resolution
        return dict(video=preprocess(video, resolution), label=label)


def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    # print(video.shape)
    video = video.permute(0, 3, 1, 2).float() / 255.  # TCHW
    # video = video.float() / 255. # TCHW
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
    video = video.permute(1, 0, 2, 3).contiguous()  # CTHW

    video -= 0.5

    return video


class HDF5Dataset(data.Dataset):
    """ Generic dataset for data stored in h5py as uint8 numpy arrays.
    Reads videos in {0, ..., 255} and returns in range [-0.5, 0.5] """

    def __init__(self, data_file, sequence_length, train=True, resolution=64):
        """
        Args:
            data_file: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64
                }
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution

        # read in data
        self.data_file = data_file
        self.data = h5py.File(data_file, 'r')
        self.prefix = 'train' if train else 'test'
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx']
        self.size = len(self._idx)

    @property
    def n_classes(self):
        raise Exception('class conditioning not support for HDF5Dataset')

    def __getstate__(self):
        state = self.__dict__
        state['data'] = None
        state['_images'] = None
        state['_idx'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_file, 'r')
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = self._idx[idx]
        end = self._idx[idx + 1] if idx < len(self._idx) - 1 else len(self._images)
        assert end - start >= 0

        start = start + np.random.randint(low=0, high=end - start - self.sequence_length)
        assert start < start + self.sequence_length <= end
        video = torch.tensor(self._images[start:start + self.sequence_length])
        return dict(video=preprocess(video, self.resolution))


class VideoData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def sample_dataloader(self):
        # Dataset = VideoDataset if osp.isdir(self.args.data_path) else VideoDataset
        # Dataset = RobotDataset
        Dataset = MetaDataset
        dataset = Dataset(self.args.data_path, self.args.sequence_length, resolution=self.args.resolution)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None
        )
        return dataloader

    def _dataset(self, train):
        # Dataset = VideoDataset if osp.isdir(self.args.data_path) else VideoDataset
        # Dataset = RobotDataset
        Dataset = MetaDataset
        dataset = Dataset(self.args.data_path, self.args.sequence_length, resolution=self.args.resolution)
        return dataset

    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()
