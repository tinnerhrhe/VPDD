"empty_dishwasher, get_ice_from_fridge"

import os
import os.path as osp
import math
import random
import pickle
import warnings
from collections import namedtuple
import glob
import h5py
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import ConcatDataset
from torchvision.datasets.video_utils import VideoClips
import pytorch_lightning as pl
from videogpt import VideoData, VideoGPT, load_videogpt, VQVAE
from videogpt.utils import shift_dim, save_video_grid
TaskBatch = namedtuple('TaskBatch', 'trajectories conditions task value')
from helpers.clip.core.clip import build_model, load_clip, tokenize
from diffuser.utils import action_tokenizer
import bisect
from einops import rearrange, repeat, reduce
import pprint
import json
from scipy.spatial.transform import Rotation
#from helpers import utils
METAWORLD_VQVAE = "Your saved path here"
RLBENCH_VQVAE = "Your saved path here"
 def keypoint_discovery(demo,
                       stopping_delta=0.1,
                       method='heuristic'):
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        return episode_keypoints
def _is_stopped(demo, i, stopped_buffer, delta=0.1):
    obs = demo[i]
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped
def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    #return euler# / 360. * 2. - 1.
    return disc

def discrete_euler_to_quaternion(discrete_euler, resolution):
    #euler = (discrete_euler + 1.) / 2. * 360.
    euluer = discrete_euler*resolution - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()
def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)
def _norm_rgb(x):
    return (x / 255.0) - 0.5
def normalize_pos(pos,gripper_loc_bounds=[[-0.3, -0.5, 0.6], [0.7, 0.5, 1.6]]):
    pos_min = np.array(gripper_loc_bounds[0])
    pos_max = np.array(gripper_loc_bounds[1])
    return ((pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0)
def point_to_voxel_index(point: np.ndarray, voxel_size: np.ndarray, coord_bounds=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(np.int32), dims_m_one
    )
    return voxel_indicy
def voxel_index_to_point(index: np.ndarray, voxel_size: np.ndarray, coord_bounds=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    point = index * (res + 1e-12) + bb_mins
    
    return point
def unnormalize_pos(pos,gripper_loc_bounds=[[-0.3, -0.5, 0.6], [0.7, 0.5, 1.6]]):
    pos_min = np.array(gripper_loc_bounds[0])
    pos_max = np.array(gripper_loc_bounds[1])
    return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

'''
logs/test/-Dec15_21-04-40/state_0.pt #TODO
logs/test/-Dec16_17-10-31
-Dec18_22-54-52
'''

class FinetuneDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae='./lightning_logs/version_40/checkpoints/last.ckpt', train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = devices
        self.load_vqvae(vqvae)
        self.actions = []
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'RLBench')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                      for ext in self.exts], [])[:100]
        print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        self.files_obs = []
        self.key_points = []
        self.propri = []
        self.multi_imgs = {'left_shoulder':[],'right_shoulder':[],'wrist':[]}
        li = tasks if tasks is not None else os.listdir(folder_robot)
        gripper_loc_bounds = json.load(open("./data_new/bounds.json",'r'))
        for task in li:
            for x in range(10):
                self.files_obs.append(sum([glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "front_rgb", f'*.{ext}'), recursive=True) for ext in exts], []))
                f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "variation_descriptions.pkl"), 'rb')
                description = pickle.load(f)[0]
                if len(description) >77: description = description[:77]
                self.task_desc.append(description)
                for c in cams[1:]:
                    self.multi_imgs[c].append((sum([glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', f"{c}_rgb", f'*.{ext}'), recursive=True) for ext in exts], [])))
                f_low_dim = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "low_dim_obs.pkl"), 'rb')
                _low_dim = pickle.load(f_low_dim)
                self.key_points.append(keypoint_discovery(_low_dim))
                tmp_action = np.zeros((len(_low_dim), 7))
                tmp_low_state = np.zeros((len(_low_dim), 4))
                for k in range(len(_low_dim)):
                    time = (1. - (k / 1000. - 1)) * 2. - 1.
                    tmp_low_state[k] = np.array([_low_dim[k].gripper_open, *_low_dim[k].gripper_joint_positions, time])
                    pose = _low_dim[k].gripper_pose[:3]
                    #import pdb;pdb.set_trace()
                    pose = point_to_voxel_index(pose, 256)
                    #pose = normalize_pos(gripper_loc_bounds[task], pose)
                    quat = normalize_quaternion(_low_dim[k].gripper_pose[3:])
                    rot = quaternion_to_discrete_euler(quat, 2)
                    tmp_action[k] =np.concatenate([pose, rot, [_low_dim[k].gripper_open]])
                self.actions.append(tmp_action)
                self.propri.append(tmp_low_state)
        #self.discretizer = action_tokenizer.QuantileDiscretizer(np.concatenate(self.actions, axis=0).reshape(-1, 7), 256)
        self.disc_actions = self.actions
        print("OBS_NUMS:", len(self.files_obs))
        self.horizon = horizon
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model
        self.process_data()
       
    # @property
    # def device(self):
    #     return self.vqvae.device
    def process_data(self):
        
        self.robot_datas = [np.load(f'./data/robot_latents_v1_{idr}.npz')['robot'] for idr in range(540)]
        self.finetune_ind = []
        error_id = []
        for i, term in enumerate(self.key_points):
            if len(self.files_obs[i]) != len(self.multi_imgs['left_shoulder'][i]):
                print(f'Image Error at step {i}')
                error_id.append(i)
            tmp = []
            # tmp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            # prev = 16
            for start in term:
                # if start < 16:
                #     continue
                # if prev > start:
                #     tmp.append(start)
                #     prev = start + 10
                #     continue
                # for x in range(prev, start, 10):
                #     tmp.append(x)
                tmp.append(start)
                # prev = start + 10
            if tmp[-1] != self.robot_datas[i].shape[0]+15:
                print(f"Error at step {i}")
                error_id.append(i)
            self.finetune_ind.append(tmp)
        print(len(self.finetune_ind))
        self.indices = []
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas):
            #if i % 10: continue
            if term is None:
                continue
            tokens = tokenize(self.task_desc[i]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
            if i in error_id:
                continue
            max_start = self.actions[i].shape[0] - 1 #min(term.shape[0] - 1, term.shape[0] - self.horizon)
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        
                #self.finetune_ind.append((i, start))
        #self.indices = np.array(indices)
        
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len
        #return len(self.finetune_ind)

    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
        #path_ind, start = self.finetune_ind[idx]
    #     #history  = np.zeros(self.wild_obs[0].shape)
        traj_latents = np.zeros((1,1,4,24,24),dtype=np.float64)
        propri = self.propri[path_ind][start]
        if start < self.robot_datas[path_ind].shape[0]:
            traj_latents = np.stack([self.robot_datas[path_ind][start:start+self.horizon],],axis=0, dtype=np.float64)
        history = np.zeros((1,1,)+traj_latents.shape[2:],dtype=np.float64)
        k = 16
        if start-k >= 0:
            #print(start,k,idx_w)
            history[:, 0] = np.stack([self.robot_datas[path_ind][start-k],],axis=0)
        imgs = np.zeros([4, 3, 128, 128])
        if start - 1 >= 0:
            front = rearrange(np.array(Image.open(self.files_obs[path_ind][start-1]), dtype=np.float64), " h w c -> c h w")
            left = rearrange(np.array(Image.open(self.multi_imgs['left_shoulder'][path_ind][start-1]), dtype=np.float64), " h w c -> c h w")
            right = rearrange(np.array(Image.open(self.multi_imgs['right_shoulder'][path_ind][start-1]), dtype=np.float64), " h w c -> c h w")
            wrist = rearrange(np.array(Image.open(self.multi_imgs['wrist'][path_ind][start-1]), dtype=np.float64), " h w c -> c h w")
            imgs = np.stack([_norm_rgb(front), _norm_rgb(left), _norm_rgb(right), _norm_rgb(wrist)], axis=0)

        task = np.array([self.robot_task_desc[path_ind],])
        action = np.zeros((4,7), dtype=np.float64)
        act_id = self.finetune_ind[path_ind][bisect.bisect_left(self.finetune_ind[path_ind], start)]
        #print(self.disc_actions[path_ind].shape, act_id, self.finetune_ind[path_ind], start)
        action[0] = self.disc_actions[path_ind][act_id]
        j = 1
        while act_id+j < self.disc_actions[path_ind].shape[0] and j < 4:
            action[j] = self.disc_actions[path_ind][act_id+j]
            j += 1

        dic_traj = {
                'obs':traj_latents,
                'act':action,
                'imgs': imgs,
            }
        #print(traj_latents.max(),traj_latents.min())
        batch = TaskBatch(dic_traj, history, task, propri)
        #batch = TaskBatch(traj_latents, history, task, 1)
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch

class MultiviewFinetuneDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae=RLBENCH_VQVAE, train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = devices
        self.load_vqvae(vqvae)
        self.poses = []
        self.quats = []
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'RLBench')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                      for ext in self.exts], [])[:100]
        print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        self.files_obs = []
        self.key_points = []
        self.propri = []
        self.files_left = []
        self.files_right = []
        self.files_wrist = []
        self.multi_imgs = {'left_shoulder':[],'right_shoulder':[],'wrist':[]}
        li = tasks if tasks is not None else os.listdir(folder_robot)
        self.trained_tasks = ['put_money_in_safe', 'reach_and_drag', 'put_groceries_in_cupboard',
        'close_jar', 'slide_block_to_color_target', 'place_shape_in_shape_sorter', 'put_item_in_drawer', 'stack_blocks', 'place_cups', 'place_wine_at_rack_location',
        'sweep_to_dustpan_of_size', 'light_bulb_in','insert_onto_square_peg', 'meat_off_grill', 'stack_cups','turn_tap','push_buttons']
        # self.trained_tasks = ['push_buttons']
        self.skip_id = []
        gripper_loc_bounds = json.load(open("./data_new/bounds.json",'r'))
        skips=0
        for task in li:
            for x in range(10):
                if task not in self.trained_tasks:
                    self.skip_id.append(skips)
                skips+=1
                self.files_obs.append(sum([sorted(glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "front_rgb", f'*.{ext}'), recursive=True),key=lambda name:int(name.split('/')[-1][:-4])) for ext in exts], []))
                self.files_left.append(sum([sorted(glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "left_shoulder_rgb", f'*.{ext}'), recursive=True),key=lambda name:int(name.split('/')[-1][:-4])) for ext in exts], []))
                self.files_right.append(sum([sorted(glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "right_shoulder_rgb", f'*.{ext}'), recursive=True),key=lambda name:int(name.split('/')[-1][:-4])) for ext in exts], []))
                self.files_wrist.append(sum([sorted(glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "wrist_rgb", f'*.{ext}'), recursive=True),key=lambda name:int(name.split('/')[-1][:-4])) for ext in exts], []))
                f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "variation_descriptions.pkl"), 'rb')
                description = pickle.load(f)[0]
                if len(description) >77: description = description[:77]
                self.task_desc.append(description)
                for c in cams[1:]:
                    self.multi_imgs[c].append(sum([sorted(glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', f"{c}_rgb", f'*.{ext}'), recursive=True),key=lambda name:int(name.split('/')[-1][:-4])) for ext in exts], []))
                f_low_dim = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "low_dim_obs.pkl"), 'rb')
                _low_dim = pickle.load(f_low_dim)
                self.key_points.append(keypoint_discovery(_low_dim))
                tmp_pose = np.zeros((len(_low_dim), 3))
                tmp_quat = np.zeros((len(_low_dim), 4))
                tmp_low_state = np.zeros((len(_low_dim), 4))
                for k in range(len(_low_dim)):
                    time = (1. - (k / 2000. - 1)) * 2. - 1.
                    tmp_low_state[k] = np.array([_low_dim[k].gripper_open, *_low_dim[k].gripper_joint_positions, 1.])
                    pose = _low_dim[k].gripper_pose[:3]
                    #import pdb;pdb.set_trace()
                    #pose = point_to_voxel_index(pose, 256)
                    pose = normalize_pos(pose)
                    quat = normalize_quaternion(_low_dim[k].gripper_pose[3:])
                    rot = quaternion_to_discrete_euler(quat, 1)
                    tmp_quat[k] = np.concatenate([rot, [_low_dim[k].gripper_open]])
                    tmp_pose[k] = pose
                self.poses.append(tmp_pose)
                self.quats.append(tmp_quat)
                self.propri.append(tmp_low_state)
        self.discretizer = action_tokenizer.QuantileDiscretizer(np.concatenate(self.poses, axis=0).reshape(-1, 3), 360)
        #import pdb;pdb.set_trace()
        self.disc_actions = [np.concatenate([self.discretizer.discretize(self.poses[cut]).reshape(-1, 3), self.quats[cut]], axis=1) for cut in range(len(self.poses))]
        print("OBS_NUMS:", len(self.files_obs))
        self.horizon = horizon
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model
        self.process_data()
       
    # @property
    # def device(self):
    #     return self.vqvae.device
    def process_data(self):
        
        self.robot_datas_front = [np.load(f'./data_multi/robot_latents_{idr}.npz')['robot'] for idr in range(540)]
        self.robot_datas_left = [np.load(f'./data_multi/robot_latents_left_{idr}.npz')['robot'] for idr in range(540)]
        self.robot_datas_right = [np.load(f'./data_multi/robot_latents_right_{idr}.npz')['robot'] for idr in range(540)]
        self.robot_datas_wrist = [np.load(f'./data_multi/robot_latents_wrist_{idr}.npz')['robot'] for idr in range(540)]
        # self.robot_datas_front_key = [np.load(f'./data_multi/robot_latents_key_{idr}.npz')['robot'] for idr in range(540)]
        # self.robot_datas_left_key = [np.load(f'./data_multi/robot_latents_left_key_{idr}.npz')['robot'] for idr in range(540)]
        # self.robot_datas_right_key = [np.load(f'./data_multi/robot_latents_right_key_{idr}.npz')['robot'] for idr in range(540)]
        # self.robot_datas_wrist_key = [np.load(f'./data_multi/robot_latents_wrist_key_{idr}.npz')['robot'] for idr in range(540)]
        self.finetune_ind = []
        error_id = []
        for i, term in enumerate(self.key_points):
           
            tmp = []
            # tmp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            # prev = 16
            for start in term:
                # if start < 16:
                #     continue
                # if prev > start:
                #     tmp.append(start)
                #     prev = start + 10
                #     continue
                # for x in range(prev, start, 10):
                #     tmp.append(x)
                tmp.append(start)
                # prev = start + 10
            if tmp[-1] != self.robot_datas_front[i].shape[0]-1:
                print(f"Error at step {i}")
                error_id.append(i)
            self.finetune_ind.append(tmp)
        print(len(self.finetune_ind))
        self.indices = []
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas_front):
            #if i % 10: continue

            if term is None:
                continue
            tokens = tokenize(self.task_desc[i]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
            if i in error_id:
                continue
            if i in self.skip_id:
                continue
            max_start = self.disc_actions[i].shape[0] #min(term.shape[0] - 1, term.shape[0] - self.horizon)
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        #import pdb;pdb.set_trace()
                #self.finetune_ind.append((i, start))
        #self.indices = np.array(indices)
        
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len
        #return len(self.finetune_ind)
# logs/test/-Jan19_14-59-42
    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
        propri = self.propri[path_ind][start]
        pred = start+20 if start+20 < len(self.files_obs[path_ind]) else len(self.files_obs[path_ind])-1
       

        if start >= 0:
            front = rearrange(np.array(Image.open(self.files_obs[path_ind][start]), dtype=np.float64), " h w c -> c h w")
            left = rearrange(np.array(Image.open(self.multi_imgs['left_shoulder'][path_ind][start]), dtype=np.float64), " h w c -> c h w")
            right = rearrange(np.array(Image.open(self.multi_imgs['right_shoulder'][path_ind][start]), dtype=np.float64), " h w c -> c h w")
            wrist = rearrange(np.array(Image.open(self.multi_imgs['wrist'][path_ind][start]), dtype=np.float64), " h w c -> c h w")
            imgs = np.stack([_norm_rgb(front), _norm_rgb(left), _norm_rgb(right), _norm_rgb(wrist)], axis=0)
        traj_latents = np.stack([self.robot_datas_front[path_ind][pred:pred+self.horizon],])
        history = np.zeros((1,1)+traj_latents.shape[2:])
        history[:, 0] = np.stack([self.robot_datas_front[path_ind][start],])
        task = np.array([self.robot_task_desc[path_ind]])
        action = np.zeros((4, 7), dtype=np.float64)
        key_ind = bisect.bisect_right(self.finetune_ind[path_ind], start)
        key_ind = len(self.finetune_ind[path_ind])-1 if key_ind >= len(self.finetune_ind[path_ind]) else key_ind
        act_id = self.finetune_ind[path_ind][key_ind]
        action[0] = self.disc_actions[path_ind][act_id]
        # for i in range(4):
        #     action[i] = self.disc_actions[path_ind][act_id]
        j = 1
        while key_ind+j < len(self.finetune_ind[path_ind]) and j < 4:
            action[j] = self.disc_actions[path_ind][self.finetune_ind[path_ind][key_ind+j]]
            j += 1
        # while act_id+j < self.disc_actions[path_ind].shape[0] and j < 4:
        #     action[j] = self.disc_actions[path_ind][act_id]
        #     j += 1
        left = np.stack([self.robot_datas_left[path_ind][start:start+1],])
        right=np.stack([self.robot_datas_right[path_ind][start:start+1],])
        wrist=np.stack([self.robot_datas_wrist[path_ind][start:start+1],])
        dic_traj = {
                'obs':traj_latents,
                'act':action,
                'left':left,
                'right':right,
                'wrist':wrist,
                'imgs':imgs,
                
            }
        batch = TaskBatch(dic_traj, history, task, propri)
        #batch = TaskBatch(traj_latents, history, task, 1)
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch
    # -Jan19_23-09-21
class VideoDataset_(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae='./lightning_logs/version_45/checkpoints/last.ckpt', train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = 'cuda:0'#devices
        self.load_vqvae(vqvae)
        self.actions = []
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'RLBench')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                      for ext in self.exts], [])
        print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        self.files_obs = []
        for task in tasks:
            for x in range(10):
                self.files_obs.append(sum([glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "front_rgb", f'*.{ext}'), recursive=True) for ext in exts], []))
                f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "variation_descriptions.pkl"), 'rb')
                self.task_desc.append(pickle.load(f))

        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")


        clips = VideoClips(files, sequence_length,  frames_between_clips=8, num_workers=32)
        pickle.dump(clips.metadata, open(cache_file, 'wb'))
        #import pdb;pdb.set_trace()
        self._clips = clips
        self.classes = ['human_hands_interaction']
        print("NUMS:", self._clips.num_clips())
        print("OBS_NUMS:", len(self.files_obs))
        self.horizon = horizon
        self.process_data()

    def process_data(self):
        print("BEGIN PROCESS DATA")
        self.wild_obs = []
        self.robot_obs = []
        self.robot_datas = []
        self.robot_len = 0
        resolution = self.resolution
        batch = torch.zeros((32, 3, 16, 96, 96),dtype=torch.float32,device=self.device)
        count = 0
        print("Begin to save")

        self.wild_obs = np.concatenate([np.load(f'./data_new_/wild_latents_v1_{idw}.npz')['wild'] for idw in range(1,10)], axis=0)


        self.wild_len = self.wild_obs.shape[0]
        #self.robot_datas = [np.load(f'./data/robot_latents_v1_{idr}.npz')['robot'] for idr in range(540)]
        self.robot_datas = [np.load(f'./data_new_/robot_latents_{idr}.npz')['robot'] for idr in range(540)]
        

        self.indices = []
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas):
            #if i % 10: continue
            if term is None:
                continue
            tokens = tokenize(self.task_desc[i]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
            max_start = min(term.shape[0] - 1, term.shape[0] - self.horizon)
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        #self.indices = np.array(indices)
        
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len

    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
    #     #history  = np.zeros(self.wild_obs[0].shape)
        idx_w = random.randint(0, self.wild_len - self.horizon)
        traj_latents = np.stack([self.robot_datas[path_ind][start:start+self.horizon], self.wild_obs[idx_w:idx_w+self.horizon]],axis=0)
        history = np.zeros((2,1,)+traj_latents.shape[2:])
        k = 14
        if start-k >= 0 and idx_w-1 >= 0:
            #print(start,k,idx_w)
            history[:, 0] = np.stack([self.robot_datas[path_ind][start-k], self.wild_obs[idx_w-1]],axis=0)
        elif start-k >= 0:
            history[0, 0] = self.robot_datas[path_ind][start-k]
        elif idx_w-1 >= 0:
            history[1, 0] = self.wild_obs[idx_w-1]
        task = np.array([self.robot_task_desc[path_ind],self.wild_embeds])
        
        if not self.pretrain:
            dic_traj = {
                'obs':traj_latents,
                'act':actions,
            }
        else:
            dic_traj = {
                'obs':traj_latents,
            }
        #print(traj_latents.max(),traj_latents.min())
        batch = TaskBatch(dic_traj, history, task, np.array([1]))
        #batch = TaskBatch(traj_latents, history, task, 1)
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch
class RobotDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae='./lightning_logs/version_45/checkpoints/last.ckpt', train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = devices
        #self.load_vqvae(vqvae)
        self.actions = []
        self.sequence_length = sequence_length
        self.resolution = resolution
        folder_robot = osp.join(data_folder, 'RLBench')
        self.key_points = []
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        #self.multi_imgs = {'left_shoulder':[],'right_shoulder':[],'wrist':[]}
        self.files_obs = []
        tasks = tasks if tasks is not None else os.listdir(folder_robot)
        for task in tasks:
            for x in range(10):
                self.files_obs.append(sum([glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "front_rgb", f'*.{ext}'), recursive=True) for ext in exts], []))
                f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "variation_descriptions.pkl"), 'rb')
                self.task_desc.append(pickle.load(f)[0])
                # for c in cams[1:]:
                #     self.multi_imgs[c].append((sum([glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', f"{c}_rgb", f'*.{ext}'), recursive=True) for ext in exts], [])))
                f_low_dim = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "low_dim_obs.pkl"), 'rb')
                _low_dim = pickle.load(f_low_dim)
                self.key_points.append(keypoint_discovery(_low_dim))

       
        print("OBS_NUMS:", len(self.files_obs))
        self.horizon = horizon
        
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model
        self.process_data()
        
    # @property
    # def device(self):
    #     return self.vqvae.device
    def process_data(self):

        self.robot_datas = [np.load(f'./data_new/robot_latents_{idr}.npz')['robot'] for idr in range(540)]
        self.finetune_ind = []
        error_id = []
        for i, term in enumerate(self.key_points):
            
            tmp = []
            # tmp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            # prev = 16
            for start in term:
                # if start < 16:
                #     continue
                # if prev > start:
                #     tmp.append(start)
                #     prev = start + 10
                #     continue
                # for x in range(prev, start, 10):
                #     tmp.append(x)
                tmp.append(start)
                # prev = start + 10
            if tmp[-1] != self.robot_datas[i].shape[0]+15:
                print(f"Error at step {i}")
                error_id.append(i)
            self.finetune_ind.append(tmp)
        print(len(self.finetune_ind))

        self.indices = []
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas):
            #if i % 10: continue
            if term is None:
                continue
            tokens = tokenize(self.task_desc[i]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
            if i in error_id:
                continue
            max_start = min(term.shape[0] - 1, term.shape[0] - self.horizon)
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        #self.indices = np.array(indices)
        
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len

    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
        
        traj_latents = np.zeros((1,1,4,24,24),dtype=np.float64)
        key_id = self.finetune_ind[path_ind][bisect.bisect_left(self.finetune_ind[path_ind], start)]
        if start+8 > key_id:
            pred = start
        elif key_id-8 >= self.robot_datas[path_ind].shape[0]:
            pred = self.robot_datas[path_ind].shape[0] - 1
        else:
            pred = key_id - 8
        if pred < self.robot_datas[path_ind].shape[0]:
            traj_latents = np.stack([self.robot_datas[path_ind][pred:pred+self.horizon],],axis=0, dtype=np.float64)
        history = np.zeros((1,1,)+traj_latents.shape[2:],dtype=np.float64)
        k = 16
        if start-k >= 0:
            #print(start,k,idx_w)
            history[:, 0] = np.stack([self.robot_datas[path_ind][start-k],],axis=0)

        
        task = np.array([self.robot_task_desc[path_ind],])
        dic_traj = {
                'obs':traj_latents,
            }
        #print(traj_latents.max(),traj_latents.min())
        batch = TaskBatch(dic_traj, history, task, np.array([1]))
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch
class HumanDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae='./lightning_logs/version_45/checkpoints/last.ckpt', train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = devices
        #self.load_vqvae(vqvae)
        self.actions = []
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                      for ext in self.exts], [])
        print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        
        
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        metadata = pickle.load(open(cache_file, 'rb'))
       
        clips = VideoClips(files, sequence_length, frames_between_clips=8, num_workers=32, _precomputed_metadata=metadata)
        self._clips = clips
        self.classes = ['human_hands_interaction']
        print("NUMS:", self._clips.num_clips())
        
        self.horizon = horizon
        
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model

        self.process_data()
        tokens = tokenize(["There are observations about human hands interacting with objects"]).numpy()
        token_tensor = torch.from_numpy(tokens).to(self.device)
        self.wild_embeds, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
        self.wild_embeds = self.wild_embeds[0].float().detach().cpu().numpy()
        print(self.wild_embeds.shape)
    # @property
    # def device(self):
    #     return self.vqvae.device
    def process_data(self):
        self.video_dict = np.load('./data_new/video.npy', allow_pickle=True).flatten()[0]
        self.video_dict_ = sorted(self.video_dict.items(), key=lambda x: x[1])
        self.cumu_idx = [self.video_dict_[i][1] for i in range(len(self.video_dict_))]
        self.wild_obs = np.concatenate([np.load(f'./data_new/wild_latents_v1_{idw}.npz')['wild'] for idw in range(1,11)], axis=0)

        self.wild_len = self.wild_obs.shape[0]
        self.indices = []
        self.wild_desc = []
        file = open('./data/result/result.txt','r')  #open prompts file
        file_data = file.readlines()
        flag = False
        self.file_desc = {}
        for row in file_data:
            if 'mp4' in row:
                video_name = row.split('/')[-1][:-1]
                flag=True
                desc=''
            elif flag and not ('sorry' in row or 'Error' in row):
                desc += row
                if row == '\n': 
                    flag = False
                    tokens = tokenize(desc[:-1]).numpy()
                    token_tensor = torch.from_numpy(tokens).to(self.device)
                    lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
                    self.file_desc[video_name] = lang_feats[0].float().detach().cpu().numpy()
        #print(self.file_desc,len(self.file_desc))
        
        
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return self.wild_len #self.robot_len - self.prd_len

    def __getitem__(self, idx):
       
        traj_latents = np.stack([self.wild_obs[idx:idx+self.horizon],],axis=0, dtype=np.float64)
        history = np.zeros((1,1,)+traj_latents.shape[2:],dtype=np.float64)
        k = 2
        if idx-k >= 0:
            history[:, 0] = np.stack([self.wild_obs[idx-k],],axis=0)
        video_idx = bisect.bisect_right(self.cumu_idx, idx)
        video_name = self.video_dict_[video_idx][0].split('/')[-1]
        desc = self.file_desc[video_name] if video_name in self.file_desc.keys() else self.wild_embeds
        task = np.array([desc,])
        dic_traj = {
                'obs':traj_latents,
            }
        batch = TaskBatch(dic_traj, history, task, np.array([1]))
        return batch
class pretrain_dataset(data.Dataset):
    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae='./lightning_logs/version_40/checkpoints/last.ckpt', train=True, resolution=96):
        self.human_dataset = HumanDataset(data_folder, sequence_length, devices, horizon, tasks=tasks, pretrain=pretrain, vqvae=vqvae, resolution=96)
        self.robot_dataset = RobotDataset(data_folder, sequence_length, devices, horizon, tasks=tasks, pretrain=pretrain, vqvae=vqvae, resolution=96)
        self.device = devices
        self.load_vqvae(vqvae)
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
    def __call__(self, *args, **kwargs):
        return ConcatDataset([self.human_dataset, self.robot_dataset])
def get_pretrain_dataset(data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae='./lightning_logs/version_45/checkpoints/last.ckpt', train=True, resolution=96):
    human_dataset = HumanDataset(data_folder, sequence_length, devices, horizon, tasks=tasks, pretrain=pretrain, vqvae=vqvae, resolution=96)
    robot_dataset = RobotDataset(data_folder, sequence_length, devices, horizon, tasks=tasks, pretrain=pretrain, vqvae=vqvae, resolution=96)
    return ConcatDataset([human_dataset, robot_dataset])

class MixDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae='./lightning_logs/version_40/checkpoints/last.ckpt', train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = 'cuda:0'#devices
        self.load_vqvae(vqvae)
        self.actions = []
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'RLBench')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                      for ext in self.exts], [])
        print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        self.files_obs = []
        self.files_left = []
        self.files_right = []
        self.files_wrist = []
        self.key_points = []
        #bounds = {task: [] for task in tasks}
        for task in tasks:
            for x in range(10):
                self.files_obs.append(sum([glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "front_rgb", f'*.{ext}'), recursive=True) for ext in exts], []))

                f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "variation_descriptions.pkl"), 'rb')
                description = pickle.load(f)[0]   #REVIEW
                if len(description) > 77: description = description[:77]
                self.task_desc.append(description)
                f_low_dim = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "low_dim_obs.pkl"), 'rb')
                _low_dim = pickle.load(f_low_dim)
                self.key_points.append(keypoint_discovery(_low_dim))

        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        metadata = pickle.load(open(cache_file, 'rb'))


        clips = VideoClips(files, sequence_length, frames_between_clips=8, num_workers=32, _precomputed_metadata=metadata)
        #import pdb;pdb.set_trace()
        self._clips = clips
        self.classes = ['human_hands_interaction']
        print("NUMS:", self._clips.num_clips())
        print("OBS_NUMS:", len(self.files_obs))



        self.horizon = horizon
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model
        tokens = tokenize(["There are observations about human hands interacting with objects"]).numpy()
        token_tensor = torch.from_numpy(tokens).to(self.device)
        self.wild_embeds, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
        self.wild_embeds = self.wild_embeds[0].float().detach().cpu().numpy()
        print(self.wild_embeds.shape)
        self.process_data()

    def process_data(self):
        print("BEGIN PROCESS DATA")
        self.wild_obs = []
        self.robot_obs = []
        self.robot_datas = []
        self.robot_len = 0
        resolution = self.resolution
        batch = torch.zeros((32, 3, 16, 96, 96),dtype=torch.float32,device=self.device)
        count = 0
        print("Begin to save")


        self.wild_obs = np.concatenate([np.load(f'./data/wild_latents_v1_{idw}.npz')['wild'] for idw in range(1,11)], axis=0)
        

        self.wild_len = self.wild_obs.shape[0]
        self.video_dict = np.load('./data_multi/video.npy', allow_pickle=True).flatten()[0]
        self.video_dict_ = sorted(self.video_dict.items(), key=lambda x: x[1])
        self.cumu_idx = [self.video_dict_[i][1] for i in range(len(self.video_dict_))]
        
        self.wild_desc = []
        file = open('./data/result/result.txt','r')  #open prompts file
        file_data = file.readlines()
        flag = False
        self.file_desc = {}
        for row in file_data:
            if 'mp4' in row:
                video_name = row.split('/')[-1][:-1]
                flag=True
                desc=''
            elif flag and not ('sorry' in row or 'Error' in row):
                desc += row
                if row == '\n': 
                    flag = False
                    tokens = tokenize(desc[:-1]).numpy()
                    token_tensor = torch.from_numpy(tokens).to(self.device)
                    lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
                    self.file_desc[video_name] = lang_feats[0].float().detach().cpu().numpy()
        self.robot_datas_m = [np.load(f'./data/robot_latents_v1_{idr}.npz')['robot'] for idr in range(540)]
        self.robot_datas_ = [np.load(f'./data/robot_latents_{idr}.npz')['robot'] for idr in range(540)]
        self.robot_datas_last = [np.load(f'./data/robot_latents_last_{idr}.npz')['robot'] for idr in range(540)]
        self.robot_datas = [np.concatenate([self.robot_datas_[k], self.robot_datas_m[k], self.robot_datas_last[k]], axis=0) for k in range(len(self.robot_datas_))]
        self.robot_datas_key = [np.load(f'./data_key/robot_latents_key_{idr}.npz')['robot'] for idr in range(540)]
        #self.robot_datas = [np.load(f'./data_new/robot_latents_{idr}.npz')['robot'] for idr in range(540)]
        
        #print(self.robot_datas[0].shape)
        #np.savez('./data/robot_latents_v1.npz', robot=np.array(self.robot_datas))
        #print(np.array(self.wild_obs).shape, np.array(self.robot_obs).shape)
        #np.savez_compressed('./data/video_latents_.npz',**dic)
        #np.savez_compressed('./data/video_latents.npz',wild=np.array(self.wild_obs), robot=np.array(self.robot_obs))
        #self.robot_len = len(self.robot_obs)
        self.finetune_ind = []
        error_id = []
        for i, term in enumerate(self.key_points):
            #print(self.robot_datas[i].shape[0],len(self.files_obs[i]))
            tmp = []
            # tmp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            # prev = 16
            for start in term:
                # if start < 16:
                #     continue
                # if prev > start:
                #     tmp.append(start)
                #     prev = start + 10
                #     continue
                # for x in range(prev, start, 10):
                #     tmp.append(x)
                tmp.append(start)
                # prev = start + 10
            if tmp[-1] != self.robot_datas[i].shape[0]-1:
                print(f"Error at step {i}")
                error_id.append(i)
            self.finetune_ind.append(tmp)
        print(len(self.finetune_ind))


        self.indices = []
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas):
            #if i % 10: continue
            if term is None:
                continue
            tokens = tokenize(self.task_desc[i]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
            if i in error_id:
                continue
            #max_start = min(term.shape[0] - 1, term.shape[0] - self.horizon)
            max_start = term.shape[0]
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        #self.indices = np.array(indices)
        
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len

    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
    #     #history  = np.zeros(self.wild_obs[0].shape)
        idx_w = random.randint(2, self.wild_len - self.horizon)
        #print(idx_w)
        # key_id = self.finetune_ind[path_ind][bisect.bisect_left(self.finetune_ind[path_ind], start)]
        # pred = key_id + 8 if key_id + 8 < self.robot_datas[path_ind].shape[0] else key_id
        pred = bisect.bisect_left(self.finetune_ind[path_ind], start)
        # traj_latents = np.stack([self.robot_datas[path_ind][pred:pred+self.horizon], self.wild_obs[idx_w:idx_w+self.horizon]],axis=0, dtype=np.float64)
        traj_latents = np.stack([self.robot_datas_key[path_ind][pred:pred+self.horizon], self.wild_obs[idx_w:idx_w+self.horizon]],axis=0, dtype=np.float64)
        history = np.zeros((2,1,)+traj_latents.shape[2:])
        
        
        history[:, 0] = np.stack([self.robot_datas[path_ind][start], self.wild_obs[idx_w-2]],axis=0)
        
        
        video_idx = bisect.bisect_right(self.cumu_idx, idx_w)
        video_name = self.video_dict_[video_idx][0].split('/')[-1]
        desc = self.file_desc[video_name] if video_name in self.file_desc.keys() else self.wild_embeds
        task = np.array([self.robot_task_desc[path_ind], desc])
        
        if not self.pretrain:
            dic_traj = {
                'obs':traj_latents,
                'act':actions,
            }
        else:
            dic_traj = {
                'obs':traj_latents,
            }
        #print(traj_latents.max(),traj_latents.min())
        batch = TaskBatch(dic_traj, history, task, np.array([1]))
        #batch = TaskBatch(traj_latents, history, task, 1)
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch
class MultiViewDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae=RLBENCH_VQVAE, train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = 'cuda:0'#devices
        self.load_vqvae(vqvae)
        self.actions = []
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'RLBench')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                      for ext in self.exts], [])
        print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        self.files_obs = []
        self.files_left = []
        self.files_right = []
        self.files_wrist = []
        self.key_points = []
        #bounds = {task: [] for task in tasks}
        for task in tasks:
            for x in range(10):
                self.files_obs.append(sum([sorted(glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "front_rgb", f'*.{ext}'), recursive=True),key=lambda name:int(name.split('/')[-1][:-4])) for ext in exts], []))
                self.files_left.append(sum([sorted(glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "left_shoulder_rgb", f'*.{ext}'), recursive=True),key=lambda name:int(name.split('/')[-1][:-4])) for ext in exts], []))
                self.files_right.append(sum([sorted(glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "right_shoulder_rgb", f'*.{ext}'), recursive=True),key=lambda name:int(name.split('/')[-1][:-4])) for ext in exts], []))
                self.files_wrist.append(sum([sorted(glob.glob(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "wrist_rgb", f'*.{ext}'), recursive=True),key=lambda name:int(name.split('/')[-1][:-4])) for ext in exts], []))
                f = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "variation_descriptions.pkl"), 'rb')
                description = pickle.load(f)[0]   #REVIEW
                if len(description) > 77: description = description[:77]
                self.task_desc.append(description)
                f_low_dim = open(osp.join(folder_robot, task, "all_variations", "episodes", f'episode{x}', "low_dim_obs.pkl"), 'rb')
                _low_dim = pickle.load(f_low_dim)
                self.key_points.append(keypoint_discovery(_low_dim))


        clips = VideoClips(files, sequence_length, frames_between_clips=8, num_workers=32)
        #import pdb;pdb.set_trace()
        self._clips = clips
        self.classes = ['human_hands_interaction']
        print("NUMS:", self._clips.num_clips())
        print("OBS_NUMS:", len(self.files_obs))



        self.horizon = horizon
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model
        tokens = tokenize(["There are observations about human hands interacting with objects"]).numpy()
        token_tensor = torch.from_numpy(tokens).to(self.device)
        self.wild_embeds, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
        self.wild_embeds = self.wild_embeds[0].float().detach().cpu().numpy()
        print(self.wild_embeds.shape)
        self.process_data()

    def process_data(self):
        print("BEGIN PROCESS DATA")
        self.wild_obs = []
        self.robot_obs = []
        self.robot_datas = []
        self.robot_len = 0
        resolution = self.resolution
        batch = torch.zeros((32, 3, 8, 96, 96),dtype=torch.float32,device=self.device)
        count = 0
        print("Begin to save")
        #error_video=np.load('./error_video_id.npy')
        
        self.video_list = np.load('./video_id.npy')


        self.wild_obs = np.concatenate([np.load(f'./data_multi/wild_latents_v1_{idw}.npz')['wild'] for idw in range(1,11)], axis=0)
        #print(self.wild_obs.shape[0], self.video_list[-1])
        # for idx in [241,243,246]:
        #          np.savez(f'./data_multi/robot_latents_wrist_{idx}.npz', robot=np.array([]))
      
        self.wild_len = self.wild_obs.shape[0]  
        self.video_dict = np.load('./data_multi/video.npy', allow_pickle=True).flatten()[0]
        self.video_dict_ = sorted(self.video_dict.items(), key=lambda x: x[1])
        self.cumu_idx = [self.video_dict_[i][1] for i in range(len(self.video_dict_))]
        print(self.cumu_idx[-1],self.video_list[-1])
        self.wild_desc = []
        file = open('./data/result/result.txt','r')  #open prompts file
        file_data = file.readlines()
        flag = False
        self.file_desc = {}
        for row in file_data:
            if 'mp4' in row:
                video_name = row.split('/')[-1][:-1]
                flag=True
                desc=''
            elif flag and not ('sorry' in row or 'Error' in row):
                desc += row
                if row == '\n': 
                    flag = False
                    tokens = tokenize(desc[:-1]).numpy()
                    token_tensor = torch.from_numpy(tokens).to(self.device)
                    lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
                    self.file_desc[video_name] = lang_feats[0].float().detach().cpu().numpy()
       
        self.robot_datas_front = [np.load(f'./data_multi/robot_latents_{idr}.npz')['robot'] for idr in range(540)]
        self.robot_datas_left = [np.load(f'./data_multi/robot_latents_left_{idr}.npz')['robot'] for idr in range(540)]
        self.robot_datas_right = [np.load(f'./data_multi/robot_latents_right_{idr}.npz')['robot'] for idr in range(540)]
        self.robot_datas_wrist = [np.load(f'./data_multi/robot_latents_wrist_{idr}.npz')['robot'] for idr in range(540)]

        self.finetune_ind = []
        error_id = []
        for i, term in enumerate(self.key_points):
            #print(self.robot_datas[i].shape[0],len(self.files_obs[i]))
            tmp = []
            # tmp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            # prev = 16
            for start in term:
                # if start < 16:
                #     continue
                # if prev > start:
                #     tmp.append(start)
                #     prev = start + 10
                #     continue
                # for x in range(prev, start, 10):
                #     tmp.append(x)
                tmp.append(start)
                # prev = start + 10
            if tmp[-1] != self.robot_datas_front[i].shape[0]-1:
                print(f"Error at step {i}")
                error_id.append(i)
            self.finetune_ind.append(tmp)
        print(len(self.finetune_ind))

        self.video_list = [x for x in self.video_list if x < self.wild_obs.shape[0]//8]
        self.indices = []
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas_front):
            #if i % 10: continue
            if term is None:
                continue
            tokens = tokenize(self.task_desc[i]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
            if i in error_id:
                continue
            #max_start = min(term.shape[0] - 1, term.shape[0] - self.horizon)
            max_start = term.shape[0]
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        #self.indices = np.array(indices)
        
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len

    def __getitem__(self, idx):
        resolution=96
        path_ind, start = self.indices[idx]
    #     #history  = np.zeros(self.wild_obs[0].shape)
        #idx_w = random.randint(2485, self.wild_len - self.horizon)
        idx_w = np.random.choice(self.video_list)
        idx_w = [idx_w,idx_w+1,idx_w+2,idx_w+3]

        pred = start+20 if start+20 < len(self.files_obs[path_ind]) else len(self.files_obs[path_ind])-1
        # traj_latents = np.stack([self.robot_datas[path_ind][pred:pred+self.horizon], self.wild_obs[idx_w:idx_w+self.horizon]],axis=0, dtype=np.float64)
        #print(self.robot_datas_front[path_ind][pred:pred+self.horizon].shape,self.robot_datas_left[path_ind][pred:pred+self.horizon].shape,self.robot_datas_right[path_ind][pred:pred+self.horizon].shape,self.robot_datas_wrist[path_ind][pred:pred+self.horizon].shape)
        traj_latents = np.stack([self.robot_datas_front[path_ind][pred:pred+self.horizon], self.robot_datas_left[path_ind][pred:pred+self.horizon], \
            self.robot_datas_right[path_ind][pred:pred+self.horizon], self.robot_datas_wrist[path_ind][pred:pred+self.horizon],\
                self.wild_obs[idx_w[0]:idx_w[0]+self.horizon, 0:1],self.wild_obs[idx_w[1]:idx_w[1]+self.horizon, 0:1],self.wild_obs[idx_w[2]:idx_w[2]+self.horizon, 0:1],self.wild_obs[idx_w[3]:idx_w[3]+self.horizon, 0:1]],axis=0, dtype=np.float32)
        history = np.zeros((8,1)+traj_latents.shape[2:])
        
        
        history[:, 0] = np.stack([self.robot_datas_front[path_ind][start], self.robot_datas_left[path_ind][start], \
            self.robot_datas_right[path_ind][start], self.robot_datas_wrist[path_ind][start], \
               self.wild_obs[idx_w[0]-2, 0:1],self.wild_obs[idx_w[1]-2, 0:1],self.wild_obs[idx_w[2]-2, 0:1],self.wild_obs[idx_w[3]-2, 0:1]],axis=0, dtype=np.float32)
        #np.stack([self.robot_datas[path_ind][start], self.wild_obs[idx_w-2]],axis=0)
        descs = []
        for ind in idx_w:
            video_idx = bisect.bisect_right(self.cumu_idx, ind)
            video_name = self.video_dict_[video_idx][0].split('/')[-1]
            desc = self.file_desc[video_name] if video_name in self.file_desc.keys() else self.wild_embeds
            descs.append(desc)
        task = np.array([self.robot_task_desc[path_ind], self.robot_task_desc[path_ind],self.robot_task_desc[path_ind],self.robot_task_desc[path_ind],\
            descs[0],descs[1],descs[2],descs[3]])
        # task = np.array([self.robot_task_desc[path_ind],
        #     descs[0]])
        if not self.pretrain:
            dic_traj = {
                'obs':traj_latents,
                'act':actions,
            }
        else:
            dic_traj = {
                'obs':traj_latents,
            }
        #print(traj_latents.max(),traj_latents.min())
        batch = TaskBatch(dic_traj, history, task, np.array([1]))
        #batch = TaskBatch(traj_latents, history, task, 1)
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch


from .utils import task_prompts
class MetaDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']

    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, num_demos=20, vqvae=METAWORLD_VAVAE, train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = "cuda:0"#devices
        self.load_vqvae(vqvae)
        self.actions = []
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'metaworld_image')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                      for ext in self.exts], [])
        print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        self.files_obs = []
        
        self.key_points = []
        self.macro = []
        #bounds = {task: [] for task in tasks}
        for ind, task in enumerate(tasks):
            data = pickle.load(open(osp.join(folder_robot, 'metaworld_'+task+'.pkl'), 'rb'))
            #print(len(data))
            data = data[:20]
            #data = data[-80:]
            #self.detect([data[i] for i in range(len(data))])
            #import pdb; pdb.set_trace()
            for i in range(len(data)):
                self.files_obs.append([data[i][j][0]for j in range(len(data[0]))])
            description = task_prompts[ind]
            self.task_desc.append(description)
        #import pdb; pdb.set_trace()
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")


        clips = VideoClips(files, sequence_length, frames_between_clips=8, num_workers=32)
        #import pdb;pdb.set_trace()
        self._clips = clips
        self.classes = ['human_hands_interaction']
        print("NUMS:", self._clips.num_clips())
        print("OBS_NUMS:", len(self.files_obs))



        self.horizon = horizon
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model
        tokens = tokenize(["There are observations about human hands interacting with objects"]).numpy()
        token_tensor = torch.from_numpy(tokens).to(self.device)
        self.wild_embeds, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
        self.wild_embeds = self.wild_embeds[0].float().detach().cpu().numpy()
        print(self.wild_embeds.shape)
        self.process_data()
    def detect(self,x):
        #last=None
        #macro=[]
        for j in x:
            macro=[]
            for i, item in enumerate(j):
                if i==0: last=item[2][-1];continue
                if item[2][-1] != last:
                     macro.append(i)
                     last = item[2][-1]
            self.macro.append(macro)
    # @property
    # def device(self):
    #     return self.vqvae.device
    def process_data(self):
        print("BEGIN PROCESS DATA")
        self.wild_obs = []
        self.robot_obs = []
        self.robot_datas = []
        self.robot_len = 0
        resolution = self.resolution
        batch = torch.zeros((32, 3, 4, 96, 96),dtype=torch.float32,device=self.device)
        count = 0
        print("Begin to save")
       
        self.video_list = np.load('./video_id.npy')

        self.robot_datas = [np.load(f'./data_meta/robot_latents_{idr}.npz')['robot'] for idr in range(len(self.files_obs))]
        self.wild_obs = np.concatenate([np.load(f'./data_meta/wild_latents_{idw}.npz')['wild'] for idw in range(1,3)], axis=0)
        self.wild_len = self.wild_obs.shape[0]  
        self.video_dict = np.load('./data_meta/video.npy', allow_pickle=True).flatten()[0]
        self.video_dict_ = sorted(self.video_dict.items(), key=lambda x: x[1])
        self.cumu_idx = [self.video_dict_[i][1] for i in range(len(self.video_dict_))]
        #print(self.cumu_idx[-1],self.video_list[-1])
        self.wild_desc = []
        file = open('./data/result/result.txt','r')  #open prompts file
        file_data = file.readlines()
        flag = False
        self.file_desc = {}
        for row in file_data:
            if 'mp4' in row:
                video_name = row.split('/')[-1][:-1]
                flag=True
                desc=''
            elif flag and not ('sorry' in row or 'Error' in row):
                desc += row
                if row == '\n': 
                    flag = False
                    tokens = tokenize(desc[:-1]).numpy()
                    token_tensor = torch.from_numpy(tokens).to(self.device)
                    lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
                    self.file_desc[video_name] = lang_feats[0].float().detach().cpu().numpy()
       
        self.indices = []
        self.video_list = [x for x in self.video_list if x < self.wild_obs.shape[0]]
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas):
            #if i % 10: continue
            if term is None:
                continue
            tokens = tokenize(self.task_desc[i//20]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
           
            #max_start = min(term.shape[0] - 1, term.shape[0] - self.horizon)
            max_start = term.shape[0]
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        #self.indices = np.array(indices)
        
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len

    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
        idx_w = np.random.choice(self.video_list)
        #idx_w = [idx_w,idx_w+1,idx_w+2,idx_w+3]
    #     #history  = np.zeros(self.wild_obs[0].shape)
        #idx_w = random.randint(2, self.wild_len - self.horizon)
        #print(idx_w)
        # key_id = self.finetune_ind[path_ind][bisect.bisect_left(self.finetune_ind[path_ind], start)]
        # pred = key_id + 8 if key_id + 8 < self.robot_datas[path_ind].shape[0] else key_id
        #pred = bisect.bisect_left(self.finetune_ind[path_ind], start)
        # traj_latents = np.stack([self.robot_datas[path_ind][pred:pred+self.horizon], self.wild_obs[idx_w:idx_w+self.horizon]],axis=0, dtype=np.float64)
        pred = start+20 if start+20 < len(self.files_obs[path_ind]) else len(self.files_obs[path_ind])-1
        traj_latents = np.stack([self.robot_datas[path_ind][pred:pred+self.horizon],self.wild_obs[idx_w:idx_w+self.horizon]],axis=0, dtype=np.float32)
        #traj_latents = np.stack([self.robot_datas_key[path_ind][pred:pred+self.horizon], self.wild_obs[idx_w:idx_w+self.horizon]],axis=0, dtype=np.float64)
        # history = np.zeros((2,1,)+traj_latents.shape[2:])
        
        
        # history[:, 0] = np.stack([self.robot_datas[path_ind][start], self.wild_obs[idx_w-2]],axis=0)
        history = np.zeros((2,1,)+traj_latents.shape[2:])
        
        
        history[:, 0] = np.stack([self.robot_datas[path_ind][start],self.wild_obs[idx_w-2]],axis=0, dtype=np.float32)
        
        video_idx = bisect.bisect_right(self.cumu_idx, idx_w)
        video_name = self.video_dict_[video_idx][0].split('/')[-1]
        #print(video_name)
        desc = self.file_desc[video_name] if video_name in self.file_desc.keys() else self.wild_embeds
        task = np.array([self.robot_task_desc[path_ind], desc])

        #task = np.array([self.robot_task_desc[path_ind], desc])
        
        if not self.pretrain:
            dic_traj = {
                'obs':traj_latents,
                'act':actions,
            }
        else:
            dic_traj = {
                'obs':traj_latents,
            }
        #print(traj_latents.max(),traj_latents.min())
        batch = TaskBatch(dic_traj, history, task, np.array([1]))
        #batch = TaskBatch(traj_latents, history, task, 1)
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch
class MetaFinetuneDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']
    #./lightning_logs/version_45/checkpoints/results/val/recon_loss=0.0003-v2.ckpt
    def __init__(self, data_folder, sequence_length, devices, horizon, num_demos=20, tasks=None, pretrain=True, vqvae=METAWORLD_VAVAE, train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = devices
        self.load_vqvae(vqvae)
        self.actions = []
        self.num_demos = num_demos
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'metaworld_image')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                      for ext in self.exts], [])
        print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        self.files_obs = []
        self.files_propri =[]
        self.key_points = []
        self.macro = []
        #bounds = {task: [] for task in tasks}
        for ind, task in enumerate(tasks):
            data = pickle.load(open(osp.join(folder_robot, 'metaworld_'+task+'.pkl'), 'rb'))
            #print(len(data))
            data = data[:20]
            #self.detect([data[i] for i in range(len(data))])
            #import pdb; pdb.set_trace()
            for i in range(len(data)):
                self.files_obs.append([data[i][j][0]for j in range(len(data[0]))])
                self.files_propri.append(np.array([data[i][j][1]for j in range(len(data[0]))]))
                self.actions.append(np.array([data[i][j][2]for j in range(len(data[0]))]))
            description = task_prompts[ind]
            self.task_desc.append(description)
        #import pdb;pdb.set_trace()
        #normed_actions = self.normalize()
        #import pdb;pdb.set_trace()
        self.discretizer = action_tokenizer.QuantileDiscretizer(np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3], 48)
        self.disc_actions = [np.concatenate([self.discretizer.discretize(self.actions[cut][:, :3]).reshape(-1, 3), np.array(self.disc_gripper(self.actions[cut][:, 3:]))], axis=-1) for cut in range(len(self.actions))]
        # self.discretizer = action_tokenizer.QuantileDiscretizer(normed_actions, 256)
        # normed_actions = normed_actions.reshape(-1, 150, 3)
        # self.disc_actions = [np.concatenate([self.discretizer.discretize(normed_actions[cut]), np.array(self.disc_gripper(self.actions[cut][:, 3:]))], axis=-1) for cut in range(len(self.actions))]
        #import pdb;pdb.set_trace()
        #self.disc_actions = self.actions
        print("OBS_NUMS:", len(self.files_obs))
        self.horizon = horizon
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model
        self.process_data()
    def normalize(self):
        self.mins = np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3].min(axis=0)
        self.maxs = np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3].max(axis=0)
        return (np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3]-self.mins) / (self.maxs-self.mins)
    # @property
    # def device(self):
    #     return self.vqvae.device
    def disc_gripper(self, x):
        gripper_list = [-1.0, 0.0, 0.10000000149011612, 0.5, 0.6000000238418579, 0.6499999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]
        return [[gripper_list.index(item)] for item in x]
    def process_data(self):
        
        self.robot_datas = [np.load(f'./data_meta/robot_latents_{idr}.npz')['robot'] for idr in range(len(self.files_obs))]
        
        self.indices = []
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas):
            #if i >= 20: continue
            if term is None:
                continue
            tokens = tokenize(self.task_desc[i//20]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
            if i%20>=self.num_demos:
                continue
            max_start = self.robot_datas[i].shape[0]-1 #min(term.shape[0] - 1, term.shape[0] - self.horizon)
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        
                #self.finetune_ind.append((i, start))
        #self.indices = np.array(indices)
        #import pdb;pdb.set_trace()
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len
        #return len(self.finetune_ind)


    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
        paths = [(path_ind+20*k)%1000 for k in range(50)]
        #path_ind, start = self.finetune_ind[idx]
    #     #history  = np.zeros(self.wild_obs[0].shape)
        #traj_latents = np.zeros((1,1,1,24,24),dtype=np.float32)
        #propri = self.propri[path_ind][start]
        pred = start+20 if start+20 < len(self.files_obs[path_ind]) else len(self.files_obs[path_ind])-1
        
        traj_latents = np.stack([self.robot_datas[path_ind][pred:pred+self.horizon] for path_ind in paths],axis=0, dtype=np.float32)
        history = np.zeros((50,1,)+traj_latents.shape[2:],dtype=np.float32)
        history[:, 0] = np.stack([self.robot_datas[path_ind][start] for path_ind in paths],axis=0, dtype=np.float32)
        
        
        task = np.array([self.robot_task_desc[path_ind] for path_ind in paths])
        action = np.zeros((50, 4,4), dtype=np.float32)
        end = start+4 if start+4 <= self.disc_actions[path_ind].shape[0] else self.disc_actions[path_ind].shape[0]
        for i, path_ind in enumerate(paths):
            action[i, 0:end-start,:4] = self.disc_actions[path_ind][start:end]
        
        # actions = np.zeros(4, 8)
        # if 
        #print(self.disc_actions[path_ind].shape, start)
        #print(task.dtype,traj_latents.dtype,action.dtype)
        dic_traj = {
                'obs':traj_latents,
                'act':action,
                'imgs': 1,
            }
        #print(traj_latents.max(),traj_latents.min())
        batch = TaskBatch(dic_traj, history, task, 1)
        #batch = TaskBatch(traj_latents, history, task, 1)
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch

class R3MMetaDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']
    #./lightning_logs/version_45/checkpoints/results/val/recon_loss=0.0003-v2.ckpt
    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae=METAWORLD_VAVAE, train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = devices
        self.load_vqvae(vqvae)
        self.actions = []
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'metaworld_image')
        files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
                      for ext in self.exts], [])
        print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        self.files_obs = []
        self.files_propri =[]
        self.key_points = []
        self.macro = []
        #bounds = {task: [] for task in tasks}
        for ind, task in enumerate(tasks):
            data = pickle.load(open(osp.join(folder_robot, 'metaworld_'+task+'.pkl'), 'rb'))
            #print(len(data))
            data = data[:20]
            #self.detect([data[i] for i in range(len(data))])
            #import pdb; pdb.set_trace()
            for i in range(len(data)):
                self.files_obs.append([data[i][j][0]for j in range(len(data[0]))])
                self.files_propri.append(np.array([data[i][j][1]for j in range(len(data[0]))]))
                self.actions.append(np.array([data[i][j][2]for j in range(len(data[0]))]))
            description = task_prompts[ind]
            self.task_desc.append(description)
        #import pdb;pdb.set_trace()
        #normed_actions = self.normalize()
        #import pdb;pdb.set_trace()
        self.discretizer = action_tokenizer.QuantileDiscretizer(np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3], 48)
        self.disc_actions = [np.concatenate([self.discretizer.discretize(self.actions[cut][:, :3]).reshape(-1, 3), np.array(self.disc_gripper(self.actions[cut][:, 3:]))], axis=-1) for cut in range(len(self.actions))]
        # self.discretizer = action_tokenizer.QuantileDiscretizer(normed_actions, 256)
        # normed_actions = normed_actions.reshape(-1, 150, 3)
        # self.disc_actions = [np.concatenate([self.discretizer.discretize(normed_actions[cut]), np.array(self.disc_gripper(self.actions[cut][:, 3:]))], axis=-1) for cut in range(len(self.actions))]
        #import pdb;pdb.set_trace()
        #self.disc_actions = self.actions
        print("OBS_NUMS:", len(self.files_obs))
        self.horizon = horizon
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model
        self.process_data()
    def normalize(self):
        self.mins = np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3].min(axis=0)
        self.maxs = np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3].max(axis=0)
        return (np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3]-self.mins) / (self.maxs-self.mins)
    # @property
    # def device(self):
    #     return self.vqvae.device
    def disc_gripper(self, x):
        gripper_list = [-1.0, 0.0, 0.10000000149011612, 0.5, 0.6000000238418579, 0.6499999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]
        return [[gripper_list.index(item)] for item in x]
    def process_data(self):
        
        self.robot_datas = [np.load(f'./data_meta/robot_latents_{idr}.npz')['robot'] for idr in range(len(self.files_obs))]
        
        self.indices = []
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas):
            #if i >= 20: continue
            if term is None:
                continue
            tokens = tokenize(self.task_desc[i//20]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
            
            max_start = self.robot_datas[i].shape[0]-1 #min(term.shape[0] - 1, term.shape[0] - self.horizon)
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        
                #self.finetune_ind.append((i, start))
        #self.indices = np.array(indices)
        #import pdb;pdb.set_trace()
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len
        #return len(self.finetune_ind)

    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
        paths = [(path_ind+20*k)%1000 for k in range(50)]
        #path_ind, start = self.finetune_ind[idx]
    #     #history  = np.zeros(self.wild_obs[0].shape)
        #traj_latents = np.zeros((1,1,1,24,24),dtype=np.float32)
        #propri = self.propri[path_ind][start]
        pred = start+20 if start+20 < len(self.files_obs[path_ind]) else len(self.files_obs[path_ind])-1
        
        traj_latents = np.stack([self.robot_datas[path_ind][pred:pred+self.horizon] for path_ind in paths],axis=0, dtype=np.float32)
        history = np.zeros((50,1,)+traj_latents.shape[2:],dtype=np.float32)
        history[:, 0] = np.stack([self.robot_datas[path_ind][start] for path_ind in paths],axis=0, dtype=np.float32)
        
        
        task = np.array([self.robot_task_desc[path_ind] for path_ind in paths])
        action = np.zeros((50, 4,4), dtype=np.float32)
        end = start+4 if start+4 <= self.disc_actions[path_ind].shape[0] else self.disc_actions[path_ind].shape[0]
        for i, path_ind in enumerate(paths):
            action[i, 0:end-start,:4] = self.disc_actions[path_ind][start:end]
        imgs = np.array([rearrange(self.files_obs[path_ind][start], 'h w c->c h w') for path_ind in paths])
        # actions = np.zeros(4, 8)
        # if 
        #print(self.disc_actions[path_ind].shape, start)
        #print(task.dtype,traj_latents.dtype,action.dtype)
        dic_traj = {
                'obs':traj_latents,
                'act':action,
                'imgs': 1,
            }
        #print(traj_latents.max(),traj_latents.min())
        batch = TaskBatch(action, imgs, task, 1)
        #batch = TaskBatch(traj_latents, history, task, 1)
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch

class ContinuousDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']
    #./lightning_logs/version_45/checkpoints/results/val/recon_loss=0.0003-v2.ckpt
    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae=METAWORLD_VAVAE, train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = devices
        #self.load_vqvae(vqvae)
        self.actions = []
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'metaworld_image')
        # files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
        #               for ext in self.exts], [])
        # print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        self.files_obs = []
        self.files_propri =[]
        self.key_points = []
        self.macro = []
        #bounds = {task: [] for task in tasks}
        for ind, task in enumerate(tasks):
            data = pickle.load(open(osp.join(folder_robot, 'metaworld_'+task+'.pkl'), 'rb'))
            #print(len(data))
            data = data[:20]
            #self.detect([data[i] for i in range(len(data))])
            #import pdb; pdb.set_trace()
            for i in range(len(data)):
                self.files_obs.append([data[i][j][0]for j in range(len(data[0]))])
                self.files_propri.append(np.array([data[i][j][1]for j in range(len(data[0]))]))
                self.actions.append(np.array([data[i][j][2]for j in range(len(data[0]))]))
            description = task_prompts[ind]
            self.task_desc.append(description)
        #import pdb;pdb.set_trace()
        #normed_actions = self.normalize()
        #import pdb;pdb.set_trace()
        # self.discretizer = action_tokenizer.QuantileDiscretizer(np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3], 48)
        # self.disc_actions = [np.concatenate([self.discretizer.discretize(self.actions[cut][:, :3]).reshape(-1, 3), np.array(self.disc_gripper(self.actions[cut][:, 3:]))], axis=-1) for cut in range(len(self.actions))]
        # self.discretizer = action_tokenizer.QuantileDiscretizer(normed_actions, 256)
        # normed_actions = normed_actions.reshape(-1, 150, 3)
        # self.disc_actions = [np.concatenate([self.discretizer.discretize(normed_actions[cut]), np.array(self.disc_gripper(self.actions[cut][:, 3:]))], axis=-1) for cut in range(len(self.actions))]
        #import pdb;pdb.set_trace()
        #self.disc_actions = self.actions
        print("OBS_NUMS:", len(self.files_obs))
        self.horizon = horizon
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model
        self.process_data()
    def normalize(self):
        self.mins = np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3].min(axis=0)
        self.maxs = np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3].max(axis=0)
        return (np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3]-self.mins) / (self.maxs-self.mins)
    # @property
    # def device(self):
    #     return self.vqvae.device
    def disc_gripper(self, x):
        gripper_list = [-1.0, 0.0, 0.10000000149011612, 0.5, 0.6000000238418579, 0.6499999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]
        return [[gripper_list.index(item)] for item in x]
    def process_data(self):
        
        self.robot_datas = [np.load(f'./data_meta/robot_latents_{idr}.npz')['robot'] for idr in range(len(self.files_obs))]
        
        self.indices = []
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas):
            #if i >= 20: continue
            if term is None:
                continue
            tokens = tokenize(self.task_desc[i//20]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
            
            max_start = self.robot_datas[i].shape[0]-1 #min(term.shape[0] - 1, term.shape[0] - self.horizon)
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        
                #self.finetune_ind.append((i, start))
        #self.indices = np.array(indices)
        #import pdb;pdb.set_trace()
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len
        #return len(self.finetune_ind)

    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
        paths = [(path_ind+20*k)%1000 for k in range(50)]
        #path_ind, start = self.finetune_ind[idx]
    #     #history  = np.zeros(self.wild_obs[0].shape)
        #traj_latents = np.zeros((1,1,1,24,24),dtype=np.float32)
        #propri = self.propri[path_ind][start]
        pred = start+20 if start+20 < len(self.files_obs[path_ind]) else len(self.files_obs[path_ind])-1
        
        # traj_latents = np.stack([self.robot_datas[path_ind][pred:pred+self.horizon] for path_ind in paths],axis=0, dtype=np.float32)
        # history = np.zeros((50,1,)+traj_latents.shape[2:],dtype=np.float32)
        # history[:, 0] = np.stack([self.robot_datas[path_ind][start] for path_ind in paths],axis=0, dtype=np.float32)
        
        
        task = np.array([self.robot_task_desc[path_ind] for path_ind in paths],dtype=np.float32)
        action = np.zeros((50, 4,4), dtype=np.float32)
        end = start+4 if start+4 <= self.actions[path_ind].shape[0] else self.actions[path_ind].shape[0]
        for i, path_ind in enumerate(paths):
            action[i, 0:end-start,:4] = self.actions[path_ind][start:end]
        imgs = np.array([rearrange(self.files_obs[path_ind][start], 'h w c->c h w') for path_ind in paths])
        # actions = np.zeros(4, 8)
        # if 
        #print(self.disc_actions[path_ind].shape, start)
        #print(task.dtype,traj_latents.dtype,action.dtype)
        
        #print(traj_latents.max(),traj_latents.min())
        batch = TaskBatch(action, imgs, task, 1)
        #batch = TaskBatch(traj_latents, history, task, 1)
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch

class DTDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm', 'png']
    #./lightning_logs/version_45/checkpoints/results/val/recon_loss=0.0003-v2.ckpt
    def __init__(self, data_folder, sequence_length, devices, horizon, tasks=None, pretrain=True, vqvae=METAWORLD_VAVAE, train=True, resolution=96):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.pretrain=pretrain
        self.device = devices
        self.load_vqvae(vqvae)
        self.actions = []
        #self.device = devices
        self.sequence_length = sequence_length
        self.resolution = resolution
        #self.classes = ['PickCube-v0']
        folder = osp.join(data_folder, 'ego4d')
        folder_robot = osp.join(data_folder, 'metaworld_image')
        # files = sum([glob.glob(osp.join(folder, f'*.{ext}'), recursive=True)
        #               for ext in self.exts], [])
        # print(files)
        #print(folder)
        exts = ['png']
        cams = ['front','left_shoulder','right_shoulder','wrist']
        self.task_desc = []
        self.files_obs = []
        self.files_propri =[]
        self.key_points = []
        self.macro = []
        #bounds = {task: [] for task in tasks}
        for ind, task in enumerate(tasks):
            data = pickle.load(open(osp.join(folder_robot, 'metaworld_'+task+'.pkl'), 'rb'))
            #print(len(data))
            data = data[:20]
            #self.detect([data[i] for i in range(len(data))])
            #import pdb; pdb.set_trace()
            for i in range(len(data)):
                self.files_obs.append([data[i][j][0]for j in range(len(data[0]))])
                self.files_propri.append(np.array([data[i][j][1]for j in range(len(data[0]))]))
                self.actions.append(np.array([data[i][j][2]for j in range(len(data[0]))]))
            description = task_prompts[ind]
            self.task_desc.append(description)
        #import pdb;pdb.set_trace()
        #normed_actions = self.normalize()
        #import pdb;pdb.set_trace()
        # self.discretizer = action_tokenizer.QuantileDiscretizer(np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3], 48)
        # self.disc_actions = [np.concatenate([self.discretizer.discretize(self.actions[cut][:, :3]).reshape(-1, 3), np.array(self.disc_gripper(self.actions[cut][:, 3:]))], axis=-1) for cut in range(len(self.actions))]
        # self.discretizer = action_tokenizer.QuantileDiscretizer(normed_actions, 256)
        # normed_actions = normed_actions.reshape(-1, 150, 3)
        # self.disc_actions = [np.concatenate([self.discretizer.discretize(normed_actions[cut]), np.array(self.disc_gripper(self.actions[cut][:, 3:]))], axis=-1) for cut in range(len(self.actions))]
        #import pdb;pdb.set_trace()
        #self.disc_actions = self.actions
        print("OBS_NUMS:", len(self.files_obs))
        self.horizon = horizon
        model, _ = load_clip('RN50', jit=False, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.clip_model.to(self.device)
        del model
        self.process_data()
    def normalize(self):
        self.mins = np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3].min(axis=0)
        self.maxs = np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3].max(axis=0)
        return (np.concatenate(self.actions, axis=0).reshape(-1, 4)[:, :3]-self.mins) / (self.maxs-self.mins)
    # @property
    # def device(self):
    #     return self.vqvae.device
    def disc_gripper(self, x):
        gripper_list = [-1.0, 0.0, 0.10000000149011612, 0.5, 0.6000000238418579, 0.6499999761581421, 0.699999988079071, 0.800000011920929, 0.8999999761581421, 1.0]
        return [[gripper_list.index(item)] for item in x]
    def process_data(self):
        
        self.robot_datas = [np.load(f'./data_meta/robot_latents_{idr}.npz')['robot'] for idr in range(len(self.files_obs))]
        
        self.indices = []
        self.robot_task_desc = []
        for i, term in enumerate(self.robot_datas):
            #if i >= 20: continue
            if term is None:
                continue
            tokens = tokenize(self.task_desc[i//20]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            lang_feats, lang_embs = self.clip_model.encode_text_with_embeddings(token_tensor)
            self.robot_task_desc.append(lang_feats[0].float().detach().cpu().numpy())
            
            max_start = self.robot_datas[i].shape[0]-1 #min(term.shape[0] - 1, term.shape[0] - self.horizon)
            for start in range(max_start):
                self.indices.append((i, start))
        print(len(self.indices))
        
                #self.finetune_ind.append((i, start))
        #self.indices = np.array(indices)
        #import pdb;pdb.set_trace()
    def load_vqvae(self, ckpt):
        from videogpt.download import load_vqvae
        if not os.path.exists(ckpt):
            self.vqvae = load_vqvae(ckpt)
        else:
            self.vqvae =  VQVAE.load_from_checkpoint(ckpt).to(self.device)
        #self.vqvae.cuda()
        self.vqvae.eval()
        #self.vqvae.to(self.device)
    @property
    def n_classes(self):
        return len(self.classes)
    def __len__(self):
        return len(self.indices) #self.robot_len - self.prd_len
        #return len(self.finetune_ind)

    def __getitem__(self, idx):
        path_ind, start = self.indices[idx]
        paths = [(path_ind+20*k)%1000 for k in range(50)]
        #path_ind, start = self.finetune_ind[idx]
    #     #history  = np.zeros(self.wild_obs[0].shape)
        #traj_latents = np.zeros((1,1,1,24,24),dtype=np.float32)
        #propri = self.propri[path_ind][start]
        pred = start+20 if start+20 < len(self.files_obs[path_ind]) else len(self.files_obs[path_ind])-1
        
        # traj_latents = np.stack([self.robot_datas[path_ind][pred:pred+self.horizon] for path_ind in paths],axis=0, dtype=np.float32)
        history = np.zeros((50,)+(1,24,24),dtype=np.float32)
        history[:] = np.stack([self.robot_datas[path_ind][start] for path_ind in paths],axis=0, dtype=np.float32)
        
        
        task = np.array([self.robot_task_desc[path_ind] for path_ind in paths],dtype=np.float32)
        action = np.zeros((50, 4,4), dtype=np.float32)
        end = start+4 if start+4 <= self.actions[path_ind].shape[0] else self.actions[path_ind].shape[0]
        for i, path_ind in enumerate(paths):
            action[i, 0:end-start,:4] = self.actions[path_ind][start:end]
        imgs = np.array([rearrange(self.files_obs[path_ind][start], 'h w c->c h w') for path_ind in paths])
        # actions = np.zeros(4, 8)
        # if 
        #print(self.disc_actions[path_ind].shape, start)
        #print(task.dtype,traj_latents.dtype,action.dtype)
        
        #print(traj_latents.max(),traj_latents.min())
        batch = TaskBatch(action, history, task, 1)
        #batch = TaskBatch(traj_latents, history, task, 1)
        #print(traj_latents.shape, history.shape,task.shape)
        
        return batch
def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


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
