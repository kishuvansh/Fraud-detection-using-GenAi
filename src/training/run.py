"""Train one experiment/seed and save every artefact needed to reproduce and evaluate it.

Run folder: outputs/runs/<experiment>/seed_<k>/
    config.yaml        resolved data + experiment config
    checkpoint.pt      best model weights + model config
    history.csv        per-epoch losses
    train_meta.json    runtime, epochs, device, package versions, latent usage
    val_scores.npy     primary anomaly score on the calibration rows
    threshold.json     calibrated threshold and how it was obtained
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.prepare import load_experiment_inputs
from src.detection.scores import recon_mse
from src.detection.threshold import calibration_record
from src.models.vae import VAE, VAEConfig
from src.training.train import TrainConfig, kl_per_dimension, train_vae
from src.utils.io import load_json, save_json, sha256_file
from src.utils.runtime import environment_info, resolve_device, set_seed


def load_experiment_config(root: Path, name: str) -> dict:
    with open(root / "configs" / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_dir(root: Path, experiment: str, seed: int) -> Path:
    return root / "outputs" / "runs" / experiment / f"seed_{seed}"


def save_checkpoint(path: Path, model: VAE, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": model.cfg.to_dict(), **extra}, path)


def load_trained_model(run_path: Path, device: torch.device | str = "cpu") -> VAE:
    device = torch.device(device)
    ckpt = torch.load(run_path / "checkpoint.pt", map_location=device, weights_only=True)
    model = VAE(VAEConfig.from_dict(ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def train_one(root: Path, data_cfg: dict, exp_cfg: dict, seed: int, data: dict | None = None,
              device_name: str | None = None, log=print) -> dict:
    experiment = exp_cfg["experiment"]
    device = resolve_device(device_name or exp_cfg.get("device", "cpu"))
    set_seed(seed, exp_cfg.get("deterministic", True), device)
    data = data or load_experiment_inputs(root, data_cfg, experiment)
    model_cfg = VAEConfig.from_dict(exp_cfg["model"])
    train_cfg = TrainConfig.from_dict(exp_cfg["training"])
    out = run_dir(root, experiment, seed)
    out.mkdir(parents=True, exist_ok=True)

    log(f"[{experiment} seed {seed}] training on {len(data['X_train']):,} rows, validating on {len(data['X_val']):,} rows, device {device}")
    result = train_vae(model_cfg, train_cfg, data["X_train"], data["X_val"], seed, device, log=log)
    model = result["model"]

    # Threshold calibration: validation rows only, no labels (E1: normal val rows; E2: all val rows).
    val_scores = recon_mse(model, data["X_val"], device)
    threshold = calibration_record(val_scores, exp_cfg["threshold"]["percentile"], experiment, "recon_mse")
    kl_dims = kl_per_dimension(model, data["X_val"], device)

    with open(out / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump({"data": data_cfg, "experiment": exp_cfg, "seed": seed}, fh, sort_keys=False)
    save_checkpoint(out / "checkpoint.pt", model, {"experiment": experiment, "seed": seed, "best_epoch": result["best_epoch"],
                                                   "transformer": f"data/processed/transformer_{experiment}.json"})
    result["history"].to_csv(out / "history.csv", index=False)
    np.save(out / "val_scores.npy", val_scores)
    save_json(threshold, out / "threshold.json")
    best_row = result["history"].loc[result["history"]["epoch"] == result["best_epoch"]].iloc[0]
    meta = {
        "experiment": experiment, "seed": seed, "train_rows": len(data["X_train"]), "val_rows": len(data["X_val"]),
        "n_parameters": model.n_parameters(), "epochs_run": result["epochs_run"], "best_epoch": result["best_epoch"],
        "best_val_neg_elbo": result["best_val_total"], "best_val_recon_mse_det": float(best_row["val_recon_mse_det"]),
        "best_train_total": float(best_row["train_total"]),
        "runtime_seconds": result["runtime_seconds"], "seconds_per_epoch": result["runtime_seconds"] / result["epochs_run"],
        "kl_per_latent_dim": kl_dims.tolist(), "active_latent_dims": int((kl_dims > 0.01).sum()),
        "checkpoint_sha256": sha256_file(out / "checkpoint.pt"),
        "environment": environment_info(device),
    }
    save_json(meta, out / "train_meta.json")
    log(f"[{experiment} seed {seed}] best epoch {result['best_epoch']}/{result['epochs_run']}, "
        f"{result['runtime_seconds']:.0f}s, threshold {threshold['threshold']:.4f}, active latent dims {meta['active_latent_dims']}")
    return {"model": model, "history": result["history"], "threshold": threshold, "meta": meta, "run_dir": out}


def read_run(run_path: Path) -> dict:
    return {
        "threshold": load_json(run_path / "threshold.json"),
        "meta": load_json(run_path / "train_meta.json"),
        "val_scores": np.load(run_path / "val_scores.npy"),
    }
