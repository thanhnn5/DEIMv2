"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # Initialize wandb (only on main process)
    if WANDB_AVAILABLE and args.use_wandb and dist_utils.is_main_process():
        # Prepare wandb config from args
        wandb_config = {
            'config_file': args.config,
            'seed': args.seed,
            'use_amp': args.use_amp,
            'device': args.device,
            'output_dir': args.output_dir,
            'summary_dir': args.summary_dir,
        }

        # Add CLI updates to config
        if args.update:
            wandb_config['cli_updates'] = args.update

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            tags=args.wandb_tags.split(',') if args.wandb_tags else None,
            notes=args.wandb_notes,
            config=wandb_config,
            resume="allow" if args.resume else None,
            dir=args.output_dir if args.output_dir else None,
        )
        print(f"wandb initialized: {wandb.run.name} ({wandb.run.url})")

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    # Log full config to wandb
    if WANDB_AVAILABLE and args.use_wandb and dist_utils.is_main_process():
        wandb.config.update(cfg.yaml_cfg, allow_val_change=True)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    # Finish wandb run
    if WANDB_AVAILABLE and args.use_wandb and dist_utils.is_main_process():
        wandb.finish()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, default='')
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, default=0, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')

    # wandb arguments
    parser.add_argument('--use-wandb', action='store_true', help='enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='deimv2-training', help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity (username or team)')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='wandb run name (auto-generated if not specified)')
    parser.add_argument('--wandb-tags', type=str, default=None, help='comma-separated wandb tags')
    parser.add_argument('--wandb-notes', type=str, default=None, help='wandb run notes/description')

    args = parser.parse_args()

    main(args)
