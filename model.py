from __future__ import annotations
import argparse
import math
from typing import List
import torch
from torch.backends import nnpack
import torch.nn as nn
import numpy as np

def build_mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    activation: str = "gelu",
    dropout_p: float = 0.0,
    layernorm: bool = False,
) -> nn.Sequential:
    """Create an MLP with optional LayerNorm & Dropout only between hidden layers."""
    act_cls = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }[activation.lower()]

    layers: List[nn.Module] = []
    dims = [in_dim] + hidden_dims
    
    for i in range(len(hidden_dims)):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(act_cls())
        
        # Only add LayerNorm/Dropout between hidden layers (not after the last hidden layer)
        if i < len(hidden_dims) - 1:
            if layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
    
    # Final output layer (no activation, layernorm, or dropout)
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal time embeddings for diffusion/flow models."""
    # t: (B, 1)
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(10000.0), steps=half, device=device)
    )  # (half,)
    angles = t * freqs  # (B, half)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 2*half)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, torch.zeros((emb.size(0), dim - emb.shape[-1]), device=device)], dim=-1)
    return emb



class ResidualBlock1D(nn.Module):
    """1D residual block for sequence modeling."""
    def __init__(self, ch: int):
        super().__init__()
        # Choose a valid number of groups that divides ch (fallback to 1)
        groups = 8
        while groups > 1 and (ch % groups) != 0:
            groups //= 2
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, ch),
            nn.SiLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups, ch),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TinyUNet1D(nn.Module):
    """Tiny 1D U-Net for action sequence modeling."""
    def __init__(self, in_ch: int, base_ch: int, out_ch: int, depth: int = 3):
        super().__init__()
        self.in_proj = nn.Conv1d(in_ch, base_ch, kernel_size=1)
        self.downs = nn.ModuleList([ResidualBlock1D(base_ch) for _ in range(depth)])
        self.mid = ResidualBlock1D(base_ch)
        self.ups = nn.ModuleList([ResidualBlock1D(base_ch) for _ in range(depth)])
        self.out_proj = nn.Conv1d(base_ch, out_ch, kernel_size=1)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout1d(0.1)

    def forward(self, x):  # x: (B, C, L)
        x = self.in_proj(x)
        skips = []
        for blk in self.downs:
            x = blk(x)
            x = self.dropout(x)  # Add dropout
            skips.append(x)
        x = self.mid(x)
        for blk, skip in zip(self.ups, reversed(skips)):
            x = x + skip
            x = blk(x)
            x = self.dropout(x)  # Add dropout
        return self.out_proj(x)


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        obs_hidden: List[int],
        policy_hidden: List[int],
        dropout_p: float = 0.0,
        layernorm: bool = False,
        arch: str = "mlp",  # 'mlp', 'state_unet', 'rgb_unet'
        unet_base_ch: int = 64,
        unet_depth: int = 3,
        mlp_time_embed: bool = False,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.arch = arch
        self.mlp_time_embed = mlp_time_embed

        # Shared observation encoder (MLP for state, CNN for RGB)
        self.obs_encoder_mlp = build_mlp(
            obs_dim, obs_hidden, obs_hidden[-1],
            dropout_p=dropout_p, layernorm=layernorm
        )
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(64, obs_hidden[-1]), 
            nn.SiLU(), 
            nn.Linear(obs_hidden[-1], obs_hidden[-1])
        )

        if arch == "mlp":
            # Traditional MLP architecture
            # If using time embedding for MLP, fuse time context with obs and do not append scalar t
            if self.mlp_time_embed:
                in_dim = obs_hidden[-1] + action_dim * horizon  # (obs+time_ctx) + action
            else:
                in_dim = obs_hidden[-1] + action_dim * horizon + 1  # obs + action + scalar t
            self.mlp = build_mlp(
                in_dim, policy_hidden, action_dim * horizon,
                dropout_p=dropout_p, layernorm=layernorm
            )
        else:
            # UNet architectures (state_unet and rgb_unet)
            # Condition via sum of obs/time embeddings
            cond_ch = obs_hidden[-1]
            self.cond_to_ch = nn.Linear(obs_hidden[-1], obs_hidden[-1])
            self.unet = TinyUNet1D(
                in_ch=action_dim + cond_ch, 
                base_ch=unet_base_ch, 
                out_ch=action_dim, 
                depth=unet_depth
            )

        # Optional RGB encoder for rgb_unet (if inputs are images)
        if arch == "rgb_unet":
            self.rgb_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1), 
                nn.SiLU(), 
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), 
                nn.SiLU(), 
                nn.AdaptiveAvgPool2d(1),
            )
            self.rgb_proj = nn.Linear(64, obs_hidden[-1])

    def _encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations based on architecture type."""
        # If rgb_unet and obs is image tensor (B,C,H,W), use CNN; else MLP on flat vectors
        if self.arch == "rgb_unet" and obs.dim() == 4:
            h = self.rgb_encoder(obs)
            h = h.view(h.size(0), -1)  # Flatten
            h = self.rgb_proj(h)
        else:
            h = self.obs_encoder_mlp(obs)
        return h

    def forward(self, obs: torch.Tensor, action_seq_flat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass supporting all three architectures."""
        if self.arch == "mlp":
            # MLP path: either concatenate scalar t or fuse sinusoidal time embedding with obs
            h_obs = self._encode_obs(obs)
            if self.mlp_time_embed:
                t_emb = sinusoidal_time_embedding(t, 64)
                t_ctx = self.time_mlp(t_emb)
                ctx = h_obs + t_ctx
                h = torch.cat([ctx, action_seq_flat], dim=-1)
            else:
                h = torch.cat([h_obs, action_seq_flat, t], dim=-1)
            return self.mlp(h)

        # UNet variants (state_unet and rgb_unet)
        B = action_seq_flat.size(0)
        H = self.horizon
        A = self.action_dim
        
        # Reshape action sequence for 1D convolution: (B, A, H)
        x = action_seq_flat.view(B, A, H)
        
        # Encode observations and time
        h_obs = self._encode_obs(obs)  # (B, C)
        t_emb = sinusoidal_time_embedding(t, 64)  # (B, 64)
        t_ctx = self.time_mlp(t_emb)  # (B, C)
        
        # Combine observation and time context
        ctx = h_obs + t_ctx  # (B, C)
        ctx = self.cond_to_ch(ctx)  # (B, C)
        
        # Expand context to match action sequence length
        ctx_map = ctx.unsqueeze(-1).expand(-1, ctx.size(1), H)  # (B, C, H)
        
        # Concatenate action and context for UNet input
        x_in = torch.cat([x, ctx_map], dim=1)  # (B, A+C, H)
        
        # Pass through UNet
        out = self.unet(x_in)  # (B, A, H)
        
        # Flatten back to (B, A*H)
        return out.view(B, A * H)
    
    def flow_matching_loss(
        self,
        obs_batch,
        action_batch,
        beta_alpha1=None,
        beta_alpha2=None,
        discrete_t_choices: torch.Tensor | None = None,
        t_dist: str | None = None,
    ):
        """Compute flow matching loss.

        Time sampling priority:
        1) discrete_t_choices (uniform over provided values)
        2) Beta(alpha1, alpha2) if provided
        3) Uniform(0,1)
        """
        batch_size = obs_batch.shape[0]
        device = obs_batch.device
        
        # Sample time steps - priority: discrete -> explicit t_dist -> beta/uniform fallback
        if discrete_t_choices is not None and discrete_t_choices.numel() > 0:
            # discrete_t_choices expected on the same device
            idx = torch.randint(low=0, high=discrete_t_choices.numel(), size=(batch_size,), device=device)
            t = discrete_t_choices[idx].view(batch_size, 1)
        else:
            # Normalize t_dist string
            td = (t_dist.lower() if isinstance(t_dist, str) else None)
            if td in ("poly_x2p1", "poly_x2_plus_1", "x2p1", "x2_plus_1"):
                # Custom distribution with pdf f(t) = (12/13) * (t^2 - t + 5/4), t in [0,1]
                # Mixture representation with non-negative weights summing to 1:
                #   w0 * Uniform(0,1)         with w0 = 3/13
                # + w1 * Beta(1,2) (pdf 2(1-t)) with w1 = 6/13
                # + w2 * Beta(3,1) (pdf 3 t^2)  with w2 = 4/13
                r = torch.rand(batch_size, 1, device=device)
                # Component samples
                t_uniform = torch.rand(batch_size, 1, device=device)
                # Beta(1,2) inverse CDF: x = 1 - sqrt(1-u)
                t_beta12 = 1.0 - torch.rand(batch_size, 1, device=device).sqrt()
                # Beta(3,1) inverse CDF: x = u^(1/3)
                t_beta31 = torch.rand(batch_size, 1, device=device).pow(1.0 / 3.0)
                # Select components according to weights 3/13, 6/13, 4/13
                t = torch.where(
                    r < (3.0/13.0),
                    t_uniform,
                    torch.where(r < (9.0/13.0), t_beta12, t_beta31)
                )
            elif td in ("left_low", "leftlow"):
                # Mixture: 1/4 Uniform + 3/4 Beta(3,1) (more mass near t=1)
                r = torch.rand(batch_size, 1, device=device)
                t_uniform = torch.rand(batch_size, 1, device=device)
                t_beta31 = torch.rand(batch_size, 1, device=device).pow(1.0 / 3.0)
                t = torch.where(r < 0.25, t_uniform, t_beta31)
            elif td in ("ploy0.75", "ploy075", "ploy0p75", "poly0.75"):
                # Custom polynomial pdf: f(t) = (t - 3/4)^2 + 41/48 on [0,1] (normalized)
                # Sample via rejection sampling with Uniform(0,1) proposal and M = 17/12 envelope
                M = 17.0 / 12.0
                t = torch.empty(batch_size, 1, device=device)
                filled = 0
                while filled < batch_size:
                    n = batch_size - filled
                    cand = torch.rand(n, 1, device=device)
                    f = (cand - 0.75) * (cand - 0.75) + (41.0 / 48.0)
                    accept = (torch.rand(n, 1, device=device) * M) < f
                    if accept.any():
                        idx = accept.squeeze(1)
                        m = int(idx.sum().item())
                        t[filled:filled + m, 0] = cand[idx, 0]
                        filled += m
            elif td in ("mix_uniform_beta", "mix_ubeta", "mix_u_beta"):
                # 50% Uniform(0,1) + 50% Beta(alpha1, alpha2)
                # Fall back to uniform if beta params are not provided
                if beta_alpha1 is None or beta_alpha2 is None:
                    t = torch.rand(batch_size, 1, device=device)
                else:
                    sel_uniform = (torch.rand(batch_size, 1, device=device) < 0.5)
                    t_uniform = torch.rand(batch_size, 1, device=device)
                    t_beta = torch.distributions.Beta(beta_alpha1, beta_alpha2).sample((batch_size, 1)).to(device)
                    t = torch.where(sel_uniform, t_uniform, t_beta)
            elif td == "uniform":
                t = torch.rand(batch_size, 1, device=device)
            elif (beta_alpha1 is not None and beta_alpha2 is not None) and (td in (None, "beta")):
                # Beta sampling via reparameterization
                u = torch.distributions.Beta(beta_alpha1, beta_alpha2).sample((batch_size, 1)).to(device)
                t = u
            else:
                # Default uniform sampling
                t = torch.rand(batch_size, 1, device=device)
        
        # Sample random Gaussian noise as the starting point
        x0 = torch.randn_like(action_batch)
        
        # Linear interpolation between noise and target
        x_t = t * action_batch + (1 - t) * x0
        
        # Velocity field (derivative of the interpolation)
        v_true = action_batch - x0
        
        # Predict velocity field
        v_pred = self.forward(obs_batch, x_t, t)
        
        # Mean squared error loss
        loss = torch.nn.functional.mse_loss(v_pred, v_true)
        return loss

    def diffusion_eps_loss(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
        t: torch.Tensor | None = None,
        *,
        T: int | None = None,
        noise: torch.Tensor | None = None,
        schedule: str = "linear",
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Diffusion loss (epsilon-prediction) compatible with baseline/diffusion_policy.

        Trains the same policy network to predict the additive noise epsilon used in
        q(x_t | x_0, t). By default uses the linear schedule alpha(tau) = 1 - tau,
        where tau is the normalized time in (0,1].

        Args:
            obs_batch: (B, obs_dim)
            action_batch: (B, A*H)
            t: (B,1) time steps. If float in [0,1], interpreted as tau. If integer in [1..T],
               provide T and it will be normalized to tau = t/T. If None, sampled U(0,1].
            T: Optional diffusion steps if t provided as integer steps.
            noise: Optional epsilon tensor; if None, sampled ~ N(0, I).
            schedule: Only 'linear' is supported (alpha=1-tau).
            reduction: 'mean' or 'none'.

        Returns:
            loss tensor (scalar if reduction='mean', else per-batch vector)
        """
        device = action_batch.device
        B = action_batch.shape[0]

        # Normalize time to tau in (0,1]
        if t is None:
            tau = torch.rand(B, 1, device=device)
            tau = tau.clamp_min(1e-6)
        else:
            if t.dtype.is_floating_point:
                tau = t.to(device)
            else:
                if T is None:
                    raise ValueError("Integer t provided but T is None; set T for normalization")
                tau = t.to(device).float() / float(max(1, int(T)))
            tau = tau.clamp_min(1e-6)

        # Linear schedule alpha = 1 - tau
        if schedule.lower() != "linear":
            raise ValueError("Only 'linear' schedule is supported in diffusion_eps_loss")
        alpha = (1.0 - tau).clamp(min=1e-6, max=1.0)
        sqrt_alpha = torch.sqrt(alpha)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha)

        # Sample/add noise epsilon
        if noise is None:
            noise = torch.randn_like(action_batch)

        # q-sample
        x_t = sqrt_alpha * action_batch + sqrt_one_minus_alpha * noise

        # Predict epsilon using the model at time tau
        eps_hat = self.forward(obs_batch, x_t, tau)

        # MSE loss
        loss = torch.nn.functional.mse_loss(eps_hat, noise, reduction=reduction)
        return loss

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", choices=["pendulum", "walker2d"], default="pendulum")
    parser.add_argument("--model", choices=["vae", "mlp"], default="vae")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layernorm", action="store_true")
    return parser.parse_args()

