"""Train the VAE for one experiment across its seeds and calibrate thresholds (validation only).

Usage (from the project root):
    python scripts/train_vae.py --experiment e1_primary            # all seeds in the config
    python scripts/train_vae.py --experiment e2_unsupervised --seeds 0
    python scripts/train_vae.py --experiment e1_primary --device auto   # use CUDA/MPS if available
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.prepare import load_experiment_inputs  # noqa: E402
from src.training.run import load_experiment_config, train_one  # noqa: E402
from src.utils.io import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="e1_primary")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--device")
    args = parser.parse_args()

    data_cfg = load_config(ROOT)
    exp_cfg = load_experiment_config(ROOT, args.experiment)
    data = load_experiment_inputs(ROOT, data_cfg, exp_cfg["experiment"])
    for seed in args.seeds if args.seeds is not None else exp_cfg["seeds"]:
        train_one(ROOT, data_cfg, exp_cfg, seed, data=data, device_name=args.device)


if __name__ == "__main__":
    main()
