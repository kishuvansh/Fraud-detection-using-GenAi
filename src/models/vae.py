"""Variational Autoencoder for tabular data, written from scratch with torch.nn.

    x (29) -> Encoder MLP -> (mu, log sigma^2) -> z = mu + sigma * eps -> Decoder MLP -> x_hat (29)

* Encoder: q_phi(z|x) = N(mu(x), diag(sigma^2(x))). It outputs log-variance because that is
  unconstrained and numerically stable; it is clamped to a safe range.
* Reparameterization: z = mu + exp(0.5 * logvar) * eps with eps ~ N(0, I). The randomness lives
  in eps, so gradients flow through mu and logvar.
* Decoder: p_theta(x|z) = N(x_hat(z), I). Its output layer is linear because standardised
  features are unbounded.
* Prior: p(z) = N(0, I). New samples come from decoding z ~ p(z).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import NamedTuple

import torch
from torch import nn

ACTIVATIONS = {
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


@dataclass
class VAEConfig:
    input_dim: int = 29
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    latent_dim: int = 8
    activation: str = "leaky_relu"
    logvar_min: float = -10.0
    logvar_max: float = 10.0

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.latent_dim < 1 or not self.hidden_dims or min(self.hidden_dims) < 1:
            raise ValueError(f"Invalid dimensions in {self}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        if not self.logvar_min < self.logvar_max:
            raise ValueError("logvar_min must be below logvar_max")
        self.hidden_dims = [int(h) for h in self.hidden_dims]

    @classmethod
    def from_dict(cls, d: dict) -> "VAEConfig":
        d = dict(d)
        if "logvar_clamp" in d:
            d["logvar_min"], d["logvar_max"] = d.pop("logvar_clamp")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


def _mlp(dims: list[int], activation: str) -> nn.Sequential:
    """Linear + activation for every consecutive pair in dims."""
    layers: list[nn.Module] = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), ACTIVATIONS[activation]()]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = _mlp([cfg.input_dim, *cfg.hidden_dims], cfg.activation)
        self.mu_head = nn.Linear(cfg.hidden_dims[-1], cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.hidden_dims[-1], cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logvar = self.logvar_head(h).clamp(self.cfg.logvar_min, self.cfg.logvar_max)
        return self.mu_head(h), logvar


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        hidden = list(reversed(cfg.hidden_dims))
        self.trunk = _mlp([cfg.latent_dim, *hidden], cfg.activation)
        self.out = nn.Linear(hidden[-1], cfg.input_dim)  # linear output: features are unbounded

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(z))


class VAEOutput(NamedTuple):
    recon: torch.Tensor   # decoder mean x_hat, shape (n, input_dim)
    mu: torch.Tensor      # posterior mean, shape (n, latent_dim)
    logvar: torch.Tensor  # posterior log-variance, shape (n, latent_dim)
    z: torch.Tensor       # latent code that was decoded, shape (n, latent_dim)


class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[1] != self.cfg.input_dim:
            raise ValueError(f"Expected input of shape (n, {self.cfg.input_dim}), got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("Input contains NaN or Inf")

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_input(x)
        return self.encoder(x)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """z = mu + sigma * eps. A CPU generator makes the noise reproducible on any device."""
        if generator is None:
            eps = torch.randn_like(mu)
        else:
            eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype).to(mu.device)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.cfg.latent_dim:
            raise ValueError(f"Expected latent of shape (n, {self.cfg.latent_dim}), got {tuple(z.shape)}")
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample: bool = True, generator: torch.Generator | None = None) -> VAEOutput:
        """sample=True draws z (training / ELBO); sample=False uses z = mu (deterministic reconstruction)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator) if sample else mu
        return VAEOutput(self.decode(z), mu, logvar, z)

    @torch.no_grad()
    def sample_prior(self, n: int, temperature: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
        """Generate n new vectors (decoder means) from z ~ N(0, temperature^2 I)."""
        device = next(self.parameters()).device
        z = torch.randn((n, self.cfg.latent_dim), generator=generator).to(device) * temperature
        return self.decode(z)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_vae(cfg: VAEConfig | dict) -> VAE:
    return VAE(cfg if isinstance(cfg, VAEConfig) else VAEConfig.from_dict(cfg))
