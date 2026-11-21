#!/usr/bin/env python3
"""Train Environmental JEPA from configured environmental data files."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoding.training import load_config, run_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/environmental_jepa.yaml")
    parser.add_argument("--output-dir", type=Path, help="Override the training output directory")
    parser.add_argument("--resume", type=Path, help="Resume from a training checkpoint")
    args = parser.parse_args(argv)
    result = run_training(load_config(args.config), output_dir=args.output_dir, resume=args.resume)
    print(f"Encoder saved: {result['encoder_path']}")
    return result


if __name__ == "__main__":
    main()
