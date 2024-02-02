"""Dataset for loading ManiKkill2 data."""

import os
import json
import h5py
import random
import datetime
from tqdm.notebook import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Sequence, Union
import clip
import gzip
import json
from pathlib import Path
from PIL import Image
def load_json(filename: Union[str, Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret

# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class PretDataset(Dataset):
    obs_modes = ['image']

    def __init__(self,
                 data_dir,
                 data_name,
                 tokenizer,
                 preprocess,
                 seq_len,
                 eval=False,
                 eval_data_split=0.2,
                 obs_mode='image',
                 action_mode='ee_delta_pose',
                 use_hand_observation=False):
        """Constructor.

        Args:
            data_dir: root directory of the data
            data_name: name of the data (maniskill2)
            tokenizer: tokenizer function
            preprocess: image preprcoess function
            seq_len: sequence length
            eval: evaluation/training
            obs_mode: image (will support rgbd and point cloud)
            action_mode: ee_delta_pose, joint_pos, joint_delta_pos
            use_hand_observation: whether to use observation image on the hand
        """
        assert obs_mode in self.obs_modes
        self.obs_mode = obs_mode
        self.data_dir = os.path.join(data_dir, data_name)
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.eval = eval
        self.use_hand_observation = use_hand_observation
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        
        if action_mode == 'ee_delta_pose':
            self.action_dim = 7
        elif action_mode == 'joint_delta_pos':
            self.action_dim = 8
        else:
            raise NotImplementedError()

        self.base_observations = []
        if use_hand_observation:
            self.hand_observations = []
        self.actions = []
        self.tokenized_texts = []
        self.env_stats = dict()
        self.seq_len = seq_len

        self.env_id_to_task_desc = {
            "AssemblingKits-v0": "Assemble the kit",
            "StackCube-v0": "Stack the red cube onto the green one",
            "PegInsertionSide-v0": "Insert the peg into the hole",
            "PlugCharger-v0": "Plug the charger into the receptable",
            "TurnFaucet-v0": "Turn on the faucet by rotating the handle"
        }

        # Find json file
        files = os.listdir(self.data_dir)
        json_path = None
        for file in files:
            if ('.json' in file) and (obs_mode in file) and (action_mode in file):
                json_path = os.path.join(self.data_dir, file)
                break
            else:
                continue
        if json_path == None:
            raise ValueError("Cannot find corresponding json file...")
        print(f"Loading json file: {json_path}")
        with open(json_path, 'r') as f:
            traj_dict = json.load(f)

        # Load trajectories
        num_trajs = 0
        h5_ids = traj_dict.keys()
        for h5_id in h5_ids:
            traj_path = os.path.join(self.data_dir, 'demos', traj_dict[h5_id])
            data = h5py.File(traj_path, "r")
            json_path = traj_path.replace(".h5", ".json")
            json_data = load_json(json_path)
            episodes = json_data["episodes"]
            env_id = json_data['env_info']['env_id']
            assert env_id in self.env_id_to_task_desc

            # Split train and eval data
            n_eps = len(episodes)
            if self.eval:
                start_ep = 0
                end_ep = np.floor(eval_data_split * n_eps).astype(np.int32)
            else:
                start_ep = np.ceil(eval_data_split * n_eps).astype(np.int32)
                if np.floor(eval_data_split * n_eps) == np.ceil(eval_data_split * n_eps):
                    start_ep += 1
                end_ep = n_eps

            # Load episodes
            print(f"start_ep: {start_ep}, end_ep: {end_ep}")
            for ep_id in range(start_ep, end_ep):
                ep = episodes[ep_id]
                trajectory = data[f"traj_{ep['episode_id']}"]
                trajectory = load_h5_data(trajectory)
                # we use :-1 here to ignore the last observation as that
                # is the terminal observation which has no actions
                ep_base_observations = trajectory["obs"]["image"]["base_camera"]["Color"][:-1]
                n_obs = ep_base_observations.shape[0]

                # # Debug
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(4)
                # ax[0].imshow(ep_base_observations[0, :, :, :3])
                # ax[1].imshow(ep_base_observations[10, :, :, :3])
                # ax[2].imshow(ep_base_observations[20, :, :, :3])
                # ax[3].imshow(ep_base_observations[30, :, :, :3])
                # plt.show()

                preprocessed_base_observations = None
                for obs_id in range(n_obs):
                    temp_base_observation = ep_base_observations[obs_id, :, :, :3]
                    temp_base_observation = np.uint8(
                        temp_base_observation * 255.0)
                    if preprocessed_base_observations is None:
                        preprocessed_base_observations = preprocess(
                            Image.fromarray(temp_base_observation)).unsqueeze(0)
                    else:
                        temp_base_observation = preprocess(
                            Image.fromarray(temp_base_observation)).unsqueeze(0)
                        preprocessed_base_observations = torch.cat(
                            [preprocessed_base_observations, temp_base_observation], dim=0)
                self.base_observations.append(preprocessed_base_observations)

                # # Debug
                # fig, ax = plt.subplots(4)
                # ax[0].imshow(np.moveaxis(preprocessed_base_observations[0].numpy(), 0, -1))
                # ax[1].imshow(np.moveaxis(preprocessed_base_observations[10].numpy(), 0, -1))
                # ax[2].imshow(np.moveaxis(preprocessed_base_observations[20].numpy(), 0, -1))
                # ax[3].imshow(np.moveaxis(preprocessed_base_observations[30].numpy(), 0, -1))
                # plt.show()
                # import ipdb; ipdb.set_trace()

                if self.use_hand_observation:
                    ep_hand_observations = trajectory["obs"]["image"]["hand_camera"]["Color"][:-1]
                    n_obs = ep_hand_observations.shape[0]
                    preprocessed_hand_observations = None
                    for obs_id in range(n_obs):
                        temp_hand_observation = ep_hand_observations[obs_id, :, :, :3]
                        temp_hand_observation = np.uint8(
                            temp_hand_observation * 255.0)
                        if preprocessed_hand_observations is None:
                            preprocessed_hand_observations = preprocess(
                                Image.fromarray(temp_hand_observation)).unsqueeze(0)
                        else:
                            temp_hand_observation = preprocess(
                                Image.fromarray(temp_hand_observation)).unsqueeze(0)
                            preprocessed_hand_observations = torch.cat(
                                [preprocessed_hand_observations, temp_hand_observation], dim=0)
                    self.hand_observations.append(
                        preprocessed_hand_observations)

                # Actions: discretize actions into 256 bins
                n_bin = 255
                if self.action_mode == 'ee_delta_pose':
                    # Normalized to (-1, 1) by ManiSkill2 already
                    low = -1
                    high = 1
                    acts = trajectory["actions"]
                elif self.action_mode == 'joint_delta_pos':
                    # Normalized to (-1, 1) by ManiSkill2 already
                    low = -1
                    high = 1
                    acts = trajectory["actions"]
                # elif self.action_mode == 'joint_pos':
                #     import ipdb; ipdb.set_trace()
                else:
                    raise NotImplementedError()
                bin_acts = np.round((acts - low) / ((high - low) / n_bin))
                self.actions.append(bin_acts)

                # Text
                ep_text = self.env_id_to_task_desc[env_id]
                ep_tokenized_text = tokenizer([ep_text])
                self.tokenized_texts.append(ep_tokenized_text)

                # Data stats
                if env_id not in self.env_stats:
                    self.env_stats[env_id] = 1
                else:
                    self.env_stats[env_id] += 1
                num_trajs += 1
                print(f"Finish loading {num_trajs} trajectories...")
        print(
            f"Finish loading {len(self)} trajectories including {len(self.env_stats)} envs")
        for env_id in self.env_stats:
            print(f"{env_id}: {self.env_stats[env_id]} trajectories")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        np.random.seed(idx)
        random.seed(idx)

        traj_idx = np.random.randint(len(self))
        base_observations = self.base_observations[traj_idx]
        if self.use_hand_observation:
            hand_observations = self.hand_observations[traj_idx]
        actions = self.actions[traj_idx]
        n_frames = actions.shape[0]
        tokenized_text_data = self.tokenized_texts[traj_idx]  # (1, clip_token_len)

        C, H, W = base_observations[0].shape
        start = np.random.randint(n_frames)
        end = start + self.seq_len
        timestep = np.zeros(self.seq_len) # (len)
        rgb = torch.zeros((self.seq_len, C, H, W)).float() # (len, C, H, W)
        action = np.zeros((self.seq_len, self.action_dim), dtype=np.float32) # (len, action_dim)
        attention_mask = np.ones(self.seq_len, dtype=np.float32) # (len)
        if end > n_frames:
            end = n_frames
            tlen = n_frames - start
            timestep[:tlen] = np.arange(start, end)
            rgb[:tlen] = base_observations[start:end]
            action[:tlen] = actions[start:end]
            attention_mask[tlen:] = 0.0
        else:
            timestep = np.arange(start, end) 
            rgb = base_observations[start:end] 
            action = actions[start:end] 
        timestep_data = torch.from_numpy(timestep).to(dtype=torch.float32).unsqueeze(0)  # (1, len)
        rgb_data = rgb  # (1, len, C, H, W)
        action_data = torch.from_numpy(action).unsqueeze(0)  # (1, len, action_dim)
        attention_mask_data = torch.from_numpy(attention_mask).unsqueeze(0) # (1, len)

        data = dict()
        data['rgb'] = rgb_data
        data['text'] = tokenized_text_data
        data['timestep'] = timestep_data
        data['action'] = action_data
        data['attention_mask'] = attention_mask_data

        return data


if __name__ == "__main__":
    DATA_DIR = './demos/PickCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5'
    DATA_NAME = 'maniskill2'
    obs_mode = 'image'
    device = "cuda" if torch.cuda.is_available() else "cpu"
   # model, preprocess = clip.load("ViT-B/32", device=device)
    dataset = PretDataset(
        DATA_DIR,
        DATA_NAME,
        clip.tokenize,
        preprocess,
        seq_len=100,
        action_mode='joint_delta_pos')
    data = dataset[0]