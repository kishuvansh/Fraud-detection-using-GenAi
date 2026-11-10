"""Train the pre-registered variants (configs/model_selection.yaml) on normal training rows.

Usage:  python scripts/run_variants.py --variants latent4 latent16 [--seeds 0 1 2 3 4]
        python scripts/run_variants.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.variants import BASELINE, VariantData, all_overrides, load_protocol, train_variant, variant_config, variant_dir  # noqa: E402
from src.utils.io import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*")
    args = parser.parse_args()
    data_cfg, protocol = load_config(ROOT), load_protocol(ROOT)
    names = [n for n in all_overrides(protocol) if n != BASELINE] if args.all else args.variants
    seeds = args.seeds if args.seeds is not None else protocol["seeds"]
    cache: dict[str, VariantData] = {}
    for name in names:
        transform = variant_config(ROOT, protocol, name)["preprocessing"]["amount_transform"]
        if transform not in cache:
            cache[transform] = VariantData(ROOT, data_cfg, transform)
        for seed in seeds:
            if (variant_dir(ROOT, name, seed) / "checkpoint.pt").exists():
                print(f"[{name} seed {seed}] already trained; skipping")
                continue
            train_variant(ROOT, data_cfg, protocol, name, seed, cache[transform])


if __name__ == "__main__":
    main()
