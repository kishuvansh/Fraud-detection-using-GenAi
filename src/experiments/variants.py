"""Variant training and scoring for the pre-registered model selection and ablation study (configs/model_selection.yaml).

Every variant inherits the E1 configuration and changes one thing. Runs are written to
outputs/runs/variants/<name>/seed_<k>/; the frozen baseline (outputs/runs/e1) is only ever read.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.load import TARGET_COL, load_raw
from src.data.prepare import processed_dir
from src.data.preprocess import FeatureTransformer
from src.data.split import derived_partitions, load_splits
from src.detection.scores import neg_elbo, per_feature_squared_error, recon_mse
from src.models.vae import VAE, VAEConfig
from src.training.run import load_experiment_config, load_trained_model, save_checkpoint
from src.training.train import TrainConfig, kl_per_dimension, train_vae
from src.utils.io import save_json, sha256_file
from src.utils.runtime import environment_info, resolve_device, set_seed

BASELINE = "baseline"


def load_protocol(root: Path) -> dict:
    with open(root / "configs" / "model_selection.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def all_overrides(protocol: dict) -> dict[str, dict]:
    out = dict(protocol["selection"]["architectures"])
    out.update(protocol["ablation"]["configurations"])
    return out


def variant_config(root: Path, protocol: dict, name: str) -> dict:
    """E1 config with the variant's single override applied."""
    cfg = copy.deepcopy(load_experiment_config(root, protocol["base_config"]))
    cfg["preprocessing"] = {"amount_transform": "log1p"}
    for section, values in all_overrides(protocol).get(name, {}).items():
        cfg.setdefault(section, {}).update(values)
    cfg["variant"] = name
    return cfg


def variant_dir(root: Path, name: str, seed: int) -> Path:
    if name == BASELINE:
        return root / "outputs" / "runs" / "e1" / f"seed_{seed}"     # frozen, read-only
    return root / "outputs" / "runs" / "variants" / name / f"seed_{seed}"


class VariantData:
    """Features for one preprocessing choice. Labels are returned only for validation rows (selection)."""

    def __init__(self, root: Path, data_cfg: dict, amount_transform: str = "log1p"):
        df = load_raw(root, data_cfg)
        y = df[TARGET_COL].to_numpy()
        self.splits, _ = load_splits(processed_dir(root, data_cfg) / "splits.json")
        parts = derived_partitions(self.splits, y)
        if amount_transform == "log1p":
            self.transformer = FeatureTransformer.load(processed_dir(root, data_cfg) / "transformer_e1.json")
        else:
            self.transformer = FeatureTransformer(amount_transform=amount_transform).fit(df.iloc[parts["e1_train"]], "e1_train (normal training rows)")
        t = lambda idx: self.transformer.transform(df.iloc[idx], dtype=np.float32)
        self.X_train = t(parts["e1_train"])
        self.X_val_normal = t(parts["e1_val_calibration"])        # early stopping + threshold (label-free)
        self.X_val_all = t(self.splits["val"])                     # model selection
        self.y_val_all = y[self.splits["val"]]                     # validation labels (selection only)
        self.X_test = t(self.splits["test"])                       # scored only after the selection lock
        self.val_normal_mask = self.y_val_all == 0


def train_variant(root: Path, data_cfg: dict, protocol: dict, name: str, seed: int, data: VariantData, log=print) -> dict:
    if name == BASELINE:
        raise ValueError("The baseline is frozen and is never retrained")
    cfg = variant_config(root, protocol, name)
    device = resolve_device(cfg.get("device", "cpu"))
    set_seed(seed, cfg.get("deterministic", True), device)
    model_cfg = VAEConfig.from_dict(cfg["model"])
    train_cfg = TrainConfig.from_dict(cfg["training"])
    out = variant_dir(root, name, seed)
    out.mkdir(parents=True, exist_ok=True)
    log(f"[{name} seed {seed}] training on {len(data.X_train):,} normal rows")
    result = train_vae(model_cfg, train_cfg, data.X_train, data.X_val_normal, seed, device, log=lambda _: None)
    model = result["model"]
    save_checkpoint(out / "checkpoint.pt", model, {"variant": name, "seed": seed, "best_epoch": result["best_epoch"],
                                                   "amount_transform": cfg["preprocessing"]["amount_transform"],
                                                   "variational": train_cfg.variational})
    data.transformer.save(out / "transformer.json")
    result["history"].to_csv(out / "history.csv", index=False)
    with open(out / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump({"variant": name, "seed": seed, "config": cfg}, fh, sort_keys=False)
    kl = kl_per_dimension(model, data.X_val_normal, device)
    meta = {"variant": name, "seed": seed, "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
            "runtime_seconds": result["runtime_seconds"], "best_val_loss": result["best_val_total"],
            "active_latent_dims": int((kl > 0.01).sum()) if train_cfg.variational else None,
            "kl_per_latent_dim": kl.tolist() if train_cfg.variational else None,
            "checkpoint_sha256": sha256_file(out / "checkpoint.pt"), "environment": environment_info(device)}
    save_json(meta, out / "train_meta.json")
    log(f"[{name} seed {seed}] best epoch {result['best_epoch']}/{result['epochs_run']}, {result['runtime_seconds']:.0f}s")
    return meta


def load_variant_model(root: Path, name: str, seed: int) -> VAE:
    return load_trained_model(variant_dir(root, name, seed), "cpu")


def is_variational(root: Path, name: str, seed: int) -> bool:
    if name == BASELINE:
        return True
    ckpt = torch.load(variant_dir(root, name, seed) / "checkpoint.pt", map_location="cpu", weights_only=True)
    return bool(ckpt.get("variational", True))


def score_set(model: VAE, X: np.ndarray, feature_mse_train: np.ndarray, variational: bool, score_seed: int = 2024) -> dict[str, np.ndarray]:
    device = torch.device("cpu")
    sq = per_feature_squared_error(model, X, device)
    out = {"recon_mse": sq.mean(axis=1), "norm_recon_mse": (sq / feature_mse_train[None, :]).mean(axis=1)}
    if variational:
        out["neg_elbo"] = neg_elbo(model, X, device, n_samples=10, seed=score_seed)
    return out


def train_feature_mse(model: VAE, X_train: np.ndarray) -> np.ndarray:
    """Mean squared reconstruction error per feature on NORMAL TRAINING rows (normaliser for norm_recon_mse)."""
    return per_feature_squared_error(model, X_train, torch.device("cpu")).mean(axis=0)
