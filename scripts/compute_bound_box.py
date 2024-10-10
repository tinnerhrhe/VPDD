"""
This script computes the minimum and maximum gripper locations for
each task in the training set.
"""

import tap
from typing import List, Tuple, Optional
from pathlib import Path
import torch
import pprint
import json



if __name__ == "__main__":
    args = Arguments().parse_args()

    bounds = {task: [] for task in args.tasks}
    
    for task in args.tasks:
        instruction = load_instructions(
            args.instructions, tasks=[task], variations=args.variations
        )

        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]
        max_episode_length = get_max_episode_length([task], args.variations)

        dataset = RLBenchDataset(
            root=args.dataset,
            instructions=instruction,
            taskvar=taskvar,
            max_episode_length=max_episode_length,
            cache_size=args.cache_size,
            max_episodes_per_task=args.max_episodes_per_task,
            cameras=args.cameras,  # type: ignore
            return_low_lvl_trajectory=True,
            dense_interpolation=True,
            interpolation_length=50,
            training=False
        )

        print(
            f"Computing gripper location bounds for task {task} "
            f"from dataset of length {len(dataset)}"
        )

        for i in range(len(dataset)):
            ep = dataset[i]
            bounds[task].append(ep["action"][:, :3])
            bounds[task].append(ep["trajectory"][..., :3].reshape([-1, 3]))

    bounds = {
        task: [
            torch.cat(gripper_locs, dim=0).min(dim=0).values.tolist(),
            torch.cat(gripper_locs, dim=0).max(dim=0).values.tolist()
        ]
        for task, gripper_locs in bounds.items()
        if len(gripper_locs) > 0
    }

    pprint.pprint(bounds)
    json.dump(bounds, open(args.out_file, "w"), indent=4)