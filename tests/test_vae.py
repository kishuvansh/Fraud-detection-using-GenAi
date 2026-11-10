"""Forward-pass, gradient and numerical checks for the from-scratch VAE.

Run with `python tests/test_vae.py` (pytest also works if installed).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.vae import VAE, VAEConfig  # noqa: E402
from src.training.losses import beta_at_epoch, kl_divergence, reconstruction_sse, vae_loss  # noqa: E402
from src.utils.runtime import resolve_device  # noqa: E402

CFG = VAEConfig(input_dim=29, hidden_dims=[64, 32], latent_dim=8)


def _model(seed: int = 0) -> VAE:
    torch.manual_seed(seed)
    return VAE(CFG)


def test_forward_shapes_and_parameter_count():
    model = _model()
    out = model(torch.randn(16, 29))
    assert out.recon.shape == (16, 29) and out.mu.shape == (16, 8) and out.logvar.shape == (16, 8) and out.z.shape == (16, 8)
    # 29*64+64 + 64*32+32 + 2*(32*8+8) + 8*32+32 + 32*64+64 + 64*29+29
    assert model.n_parameters() == 8813


def test_architecture_is_configurable():
    model = VAE(VAEConfig(input_dim=29, hidden_dims=[128, 64, 16], latent_dim=3, activation="gelu"))
    out = model(torch.randn(4, 29))
    assert out.recon.shape == (4, 29) and out.mu.shape == (4, 3)


def test_bad_input_shape_and_values_are_rejected():
    model = _model()
    for bad in (torch.randn(4, 28), torch.randn(29), torch.full((2, 29), float("nan"))):
        try:
            model(bad)
        except ValueError:
            continue
        raise AssertionError(f"input {tuple(bad.shape)} was not rejected")


def test_deterministic_mode_uses_mu():
    model = _model()
    x = torch.randn(8, 29)
    out = model(x, sample=False)
    assert torch.equal(out.z, out.mu)
    assert torch.equal(model(x, sample=False).recon, out.recon)


def test_reparameterization_statistics_and_reproducibility():
    mu, logvar = torch.full((200_000, 1), 2.0), torch.full((200_000, 1), math.log(0.25))
    z = VAE.reparameterize(mu, logvar, torch.Generator().manual_seed(0))
    assert abs(z.mean().item() - 2.0) < 0.01 and abs(z.std().item() - 0.5) < 0.01
    z2 = VAE.reparameterize(mu, logvar, torch.Generator().manual_seed(0))
    assert torch.equal(z, z2)


def test_kl_closed_form_is_zero_at_prior_and_matches_monte_carlo():
    assert kl_divergence(torch.zeros(3, 8), torch.zeros(3, 8)).abs().max().item() < 1e-7
    g = torch.Generator().manual_seed(1)
    mu, logvar = torch.randn(1, 8, generator=g), torch.randn(1, 8, generator=g) * 0.5
    sigma = torch.exp(0.5 * logvar)
    z = mu + sigma * torch.randn(400_000, 8, generator=g)
    log_q = (-0.5 * ((z - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(1)
    log_p = (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi)).sum(1)
    mc = (log_q - log_p).mean().item()
    exact = kl_divergence(mu, logvar).item()
    assert abs(mc - exact) / exact < 0.01, (mc, exact)


def test_reconstruction_term_is_half_sse():
    x, r = torch.ones(2, 29), torch.zeros(2, 29)
    assert torch.allclose(reconstruction_sse(x, r), torch.full((2,), 14.5))


def test_gradients_reach_every_parameter_through_the_sampling_step():
    model = _model()
    loss = vae_loss(x := torch.randn(64, 29), model(x), beta=1.0)["total"]
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    # logvar head only receives gradient via the reparameterised sample and the KL term
    assert model.encoder.logvar_head.weight.grad.abs().sum() > 0


def test_beta_warmup_schedule():
    assert [round(beta_at_epoch(e, 10, 1.0), 2) for e in (1, 5, 10, 11, 50)] == [0.1, 0.5, 1.0, 1.0, 1.0]
    assert beta_at_epoch(1, 0, 0.5) == 0.5


def test_numerically_stable_on_extreme_inputs():
    model = _model()
    x = torch.randn(32, 29) * 1e3
    out = model(x)
    assert out.logvar.min() >= -10 and out.logvar.max() <= 10
    assert torch.isfinite(vae_loss(x, out, 1.0)["total"])


def test_can_overfit_a_small_batch():
    """Sanity check from the plan: the model can learn (near-)perfect reconstructions of 256 rows."""
    torch.manual_seed(0)
    model = VAE(CFG)
    x = torch.randn(256, 29)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for step in range(1500):
        out = model(x, sample=False)
        loss = reconstruction_sse(x, out.recon).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    mse = (x - model(x, sample=False).recon).pow(2).mean().item()
    assert mse < 0.15 * (first * 2 / 29), (first, mse)  # at least ~85% reduction in per-feature MSE


def test_prior_sampling_and_device_move():
    model = _model()
    samples = model.sample_prior(10, generator=torch.Generator().manual_seed(0))
    assert samples.shape == (10, 29) and torch.isfinite(samples).all()
    device = resolve_device("auto")
    moved = model.to(device)
    out = moved(torch.randn(4, 29, device=device))
    assert out.recon.device.type == device.type


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
