from __future__ import annotations

import argparse
from pathlib import Path

from physqformer.config import CONFIG
from physqformer.experiment import run_ablation_suite, run_single_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhysQ-Former on CAPD-style pkl files.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing train_data_1.pkl ... train_data_13.pkl")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--split-mode", type=str, default=None, choices=["random", "by_file"], help="Rainy-sample split mode")
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or cuda:0")
    parser.add_argument("--ablation", action="store_true", help="Run the ablation suite")
    return parser.parse_args()


def main():
    args = parse_args()
    updates = {}
    if args.data_dir is not None:
        updates["data_dir"] = str(Path(args.data_dir))
    if args.epochs is not None:
        updates["epochs"] = args.epochs
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.split_mode is not None:
        updates["split_mode"] = args.split_mode
    if args.device is not None:
        updates["device"] = args.device

    CONFIG.update(updates)
    if args.ablation:
        run_ablation_suite()
    else:
        run_single_experiment({})


if __name__ == "__main__":
    main()
