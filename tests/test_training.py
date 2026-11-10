"""Training reproducibility, threshold calibration and evaluation-order tests.

Run with `python tests/test_training.py` (pytest also works if installed).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.evaluate import evaluate_predictions  # noqa: E402
from src.detection.threshold import calibration_record, percentile_threshold  # noqa: E402
from src.evaluation.detection_metrics import sensitivity_table, threshold_metrics  # noqa: E402
from src.generation.sample import generate_model_space  # noqa: E402
from src.models.vae import VAE, VAEConfig  # noqa: E402
from src.training.train import TrainConfig, train_vae  # noqa: E402
from src.utils.io import load_config  # noqa: E402
from src.utils.runtime import set_seed  # noqa: E402

RNG = np.random.default_rng(0)
X_TRAIN = RNG.normal(size=(2000, 29)).astype(np.float32)
X_VAL = RNG.normal(size=(500, 29)).astype(np.float32)
SMALL = TrainConfig(batch_size=256, max_epochs=4, kl_warmup_epochs=2, early_stopping_patience=2)


def _train(seed: int):
    set_seed(seed, True, torch.device("cpu"))
    return train_vae(VAEConfig(), SMALL, X_TRAIN, X_VAL, seed, torch.device("cpu"), log=lambda _: None)


def test_training_is_bitwise_reproducible_on_cpu():
    a, b = _train(3), _train(3)
    cols = ["train_total", "train_recon", "train_kl", "val_total", "val_recon_mse_det"]
    assert np.array_equal(a["history"][cols].to_numpy(), b["history"][cols].to_numpy())
    for (k, va), (_, vb) in zip(a["model"].state_dict().items(), b["model"].state_dict().items()):
        assert torch.equal(va, vb), k


def test_different_seeds_give_different_models():
    a, b = _train(1), _train(2)
    assert not np.array_equal(a["history"]["train_total"].to_numpy(), b["history"]["train_total"].to_numpy())


def test_best_epoch_is_after_warmup_and_has_lowest_val_loss():
    r = _train(0)
    h = r["history"]
    eligible = h[h["epoch"] >= SMALL.kl_warmup_epochs]
    assert r["best_epoch"] == int(eligible.loc[eligible["val_total"].idxmin(), "epoch"])


def test_percentile_threshold_flags_expected_share():
    scores = np.arange(1, 10001, dtype=float)
    tau = percentile_threshold(scores, 99.0)
    assert abs(np.mean(scores > tau) - 0.01) < 1e-3
    rec = calibration_record(scores, 99.0, "e1", "recon_mse")
    assert rec["fraud_labels_used"] is False and rec["threshold"] == tau


def test_threshold_metrics_and_sensitivity_use_validation_thresholds():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.95])
    m = threshold_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 3) and abs(m["precision"] - 2 / 3) < 1e-12 and m["recall"] == 1.0
    val = np.linspace(0, 1, 101)
    table = sensitivity_table(y, s, val, [50, 90], primary=90)
    assert np.allclose(table["threshold"], [0.5, 0.9]) and list(table["role"]) == ["sensitivity", "primary"]


def test_labels_cannot_be_read_before_predictions_exist():
    data_cfg = load_config(ROOT)
    with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as tmp:
        fake_root = Path(tmp)
        exp_cfg = {"experiment": "e_missing", "evaluation": {}, "threshold": {"percentile": 99.0}}
        try:
            evaluate_predictions(fake_root, data_cfg, exp_cfg, seed=0)
        except RuntimeError as exc:
            assert "before test labels" in str(exc)
        else:
            raise AssertionError("evaluation ran without saved predictions")


def test_generation_is_reproducible_and_noise_variant_adds_spread():
    torch.manual_seed(0)
    model = VAE(VAEConfig())
    a = generate_model_space(model, 1000, seed=5)
    b = generate_model_space(model, 1000, seed=5)
    assert np.array_equal(a, b) and a.shape == (1000, 29)
    noisy = generate_model_space(model, 1000, seed=5, noise_std=np.full(29, 0.5))
    assert noisy.std(axis=0).mean() > a.std(axis=0).mean()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
