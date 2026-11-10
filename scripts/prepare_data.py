"""Phase: leakage-safe split + preprocessing. Saves splits, fitted transformers and check results.

Usage (from the project root):  python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.prepare import prepare_data  # noqa: E402
from src.utils.io import load_config  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT)
    result = prepare_data(ROOT, cfg)
    print(result["partitions"].to_string(float_format=lambda v: f"{v:.4f}"))
    print()
    print(result["experiments"].drop(columns="used_for").to_string())
    print()
    for r in result["checks"]:
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['check']}: {r['detail']}")


if __name__ == "__main__":
    main()
