import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.VideoModel',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 1,
        'n_diffusion_steps': 100,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',
        ## distributed
        'num_node':1,

        ## dataset
        'loader': 'datasets.VideoDataset',
        'normalizer': 'GaussianNormalizer',
        'data_folder': './data',
        'sequence_length': 16,
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1000,
        'pretrain': True,
        'concat': False,
        'focal': False,
        'force': False,
        'act_classes': 256,
        'tasks': ['put_all_groceries_in_cupboard', 'set_the_table', 'hang_frame_on_hanger', 'setup_chess', 'turn_tap', 'take_plate_off_colored_dish_rack', 'take_toilet_roll_off_stand', 
        'close_jar', 'put_books_at_shelf_location', 'meat_on_grill', 'toilet_seat_down', 'light_bulb_in', 'take_cup_out_from_cabinet', 'wipe_desk', 'tv_on', 'slide_block_to_color_target', 
        'open_jar', 'sweep_to_dustpan_of_size', 'screw_nail', 'push_buttons', 'put_groceries_in_cupboard', 'empty_dishwasher', 'put_money_in_safe', 'put_tray_in_oven', 'straighten_rope',
        'solve_puzzle', 'slide_block_to_target', 'place_shape_in_shape_sorter', 'put_item_in_drawer', 'take_shoes_out_of_box', 'lamp_on', 'play_jenga', 'insert_usb_in_computer', 'water_plants',
        'insert_onto_square_peg', 'pour_from_cup_to_cup', 'hit_ball_with_queue', 'take_off_weighing_scales', 'scoop_with_spatula', 'move_hanger', 'unplug_charger', 'reach_and_drag', 'place_wine_at_rack_location', 
        'get_ice_from_fridge', 'stack_cups', 'place_cups', 'sweep_to_dustpan', 'meat_off_grill', 'change_clock', 'take_umbrella_out_of_umbrella_stand', 'slide_cabinet_open_and_place_cups', 'put_knife_in_knife_block', 'stack_blocks', 'hockey'],
        'single': False,
        'meta_tasks': ['basketball-v2', 'bin-picking-v2',  'button-press-topdown-v2',
 'button-press-v2', 'button-press-wall-v2', 'coffee-button-v2',
'coffee-pull-v2', 'coffee-push-v2', 'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2',
'door-open-v2', 'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2',
 'faucet-close-v2',  'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 'handle-pull-v2',
 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 'reach-v2', 'push-back-v2',
 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'plate-slide-back-v2',
 'plate-slide-back-side-v2',  'soccer-v2',
 'push-wall-v2',  'shelf-place-v2', 'sweep-into-v2', 'sweep-v2', 'window-open-v2', 'window-close-v2','assembly-v2',
 'button-press-topdown-wall-v2','hammer-v2','peg-unplug-side-v2',
                               'reach-wall-v2', 'stick-push-v2', 'stick-pull-v2', 'box-close-v2'],

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),
        'num_demos':20,

        ## training
        'n_steps_per_epoch': 50000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 50000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'plan': {
        'guide': 'sampling.nogradGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.997,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
        'meta_task':'basketball-v2'
    },
}


#------------------------ overrides ------------------------#


hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}
