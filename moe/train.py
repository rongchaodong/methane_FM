"""Command-line entry point for methane MoE training."""

import argparse
import json
from pathlib import Path

from encoding.training import load_config

from .training import run_training


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train or resume the Methane Continual MoE")
    parser.add_argument("--config", type=Path, required=True, help="MoE YAML or JSON configuration")
    parser.add_argument("--stage", choices=("pretrain", "finetune", "continual"), required=True)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--checkpoint", type=Path, help="Initialize a new stage from the previous stage")
    checkpoints.add_argument("--resume", type=Path, help="Resume a checkpoint from this stage")
    parser.add_argument("--output-dir", type=Path, help="Override checkpoint and metrics output directory")
    parser.add_argument("--prepare-data", action="store_true", help="Freeze data splits and print an audit without training")
    args = parser.parse_args(argv)
    if (args.stage != "pretrain" and not (args.prepare_data and args.stage == "finetune")
            and not (args.checkpoint or args.resume)):
        parser.error(f"{args.stage} requires --checkpoint or --resume")
    if args.stage == "pretrain" and args.checkpoint:
        parser.error("Use --resume to continue pretraining")
    result = run_training(load_config(args.config), args.stage, args.checkpoint, args.resume, args.output_dir,
                          prepare_data=args.prepare_data)
    if args.prepare_data:
        print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
