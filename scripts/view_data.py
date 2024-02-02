"""Replay the trajectory stored in HDF5.
The replayed trajectory can use different observation modes and control modes.
We support translating actions from certain controllers to a limited number of controllers.
The script is only tested for Panda, and may include some Panda-specific hardcode.

Need to export MS2_ASSET_DIR=/path/to/data/
"""

import argparse
import multiprocessing as mp
import os
from copy import deepcopy
from typing import Union

import gym
import h5py
import numpy as np
import sapien.core as sapien
from tqdm.auto import tqdm
from transforms3d.quaternions import quat2axangle
import gzip
import json
from pathlib import Path
from typing import Sequence, Union

import numpy as np
# import mani_skill2.envs
# from mani_skill2.agents.base_controller import CombinedController
# from mani_skill2.agents.controllers import *
# from mani_skill2.envs.sapien_env import BaseEnv
# from mani_skill2.trajectory.merge_trajectory import merge_h5
# from mani_skill2.utils.common import clip_and_scale_action, inv_scale_action
# from mani_skill2.utils.io_utils import load_json
# from mani_skill2.utils.sapien_utils import get_entity_by_name
# from mani_skill2.utils.wrappers import RecordEpisode



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


def view_data(traj_path):
    # Load HDF5 containing trajectories
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    #json_data = load_json(json_path)

    import ipdb; ipdb.set_trace()


def main():
    # args = parse_args()

    # if args.num_procs > 1:
    #     pool = mp.Pool(args.num_procs)
    #     proc_args = [(deepcopy(args), i, args.num_procs) for i in range(args.num_procs)]
    #     res = pool.starmap(_main, proc_args)
    #     pool.close()
    #     if args.save_traj:
    #         # A hack to find the path
    #         output_path = res[0][: -len("0.h5")] + "h5"
    #         merge_h5(output_path, res)
    #         for h5_path in res:
    #             tqdm.write(f"Remove {h5_path}")
    #             os.remove(h5_path)
    #             json_path = h5_path.replace(".h5", ".json")
    #             tqdm.write(f"Remove {json_path}")
    #             os.remove(json_path)
    # else:
    #     _main(args)

    traj_path = './demos/PickCube-v0/trajmslearn.rgbd.pd_ee_delta_pose.h5'
    view_data(traj_path)


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main()
