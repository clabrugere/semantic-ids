from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from semantic_ids.kmeans import kmeans, nearest_code


def quantize_residuals(
    z: Tensor,
    codebooks: Tensor,
    num_levels: int,
    normalize: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the residual chain. Returns ``(entries [num_levels, B, dim], residuals [num_levels, B, dim], codes [B, num_levels])``.

    ``residuals[k]`` is the residual entering step k (so ``residuals[0] is z``) and ``entries[k]`` is the code picked for it.

    With ``normalize``, a code is chosen by direction alone (the codebook is unit-normalized before the nearest-centroid
    search), but the entry returned is still the raw row. Per-code norms therefore stop contributing to attribution and only
    affect reconstruction, so one centroid cannot capture assignments by magnitude.
    """

    bs, dim = z.size(0), z.size(1)
    entries = torch.empty((num_levels, bs, dim), device=z.device)
    residuals = torch.empty((num_levels, bs, dim), device=z.device)
    codes = torch.empty((bs, num_levels), dtype=torch.long, device=z.device)

    residual = z

    for k in range(num_levels):
        raw_codebook = codebooks[k]  # [V, dim]
        search_codebook = F.normalize(raw_codebook, dim=-1) if normalize else raw_codebook
        idx = nearest_code(residual, search_codebook)  # [B] nearest by direction alone when normalize
        entry = raw_codebook[idx]  # [B, dim] the entry keeps its magnitude regardless of normalize

        residuals[k] = residual
        entries[k] = entry
        codes[:, k] = idx

        residual = residual - entry

    return entries, residuals, codes


class ResidualQuantizer(nn.Module):
    """K-1 stacked codebooks; step k quantizes the residual left by step k-1."""

    # Registered-buffer type declaration (nn.Module.__getattr__ returns Tensor | Module).
    usage_ema: Tensor

    def __init__(
        self,
        num_levels: int,
        num_codes: int,
        dim: int,
        commitment: float = 0.25,
        normalize_codebook: bool = True,
        usage_decay: float = 0.99,
    ):
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")

        super().__init__()
        self.num_levels = num_levels
        self.num_codes = num_codes
        self.commitment = commitment
        self.normalize_codebook = normalize_codebook
        self.usage_decay = usage_decay
        # Running per-code share of assignments, so expire_dead_codes_ can tell a code that never gets
        # assigned from one that merely got no assignment in this batch. A share rather than a count, so it
        # does not scale with the batch size. Initialized at the uniform share, so nothing looks dead before
        # training runs.
        self.register_buffer("usage_ema", torch.full((num_levels, num_codes), 1.0 / num_codes))
        self.codebooks = nn.Parameter(torch.randn(num_levels, num_codes, dim) * 0.1)  # flattened [K, V, dim] codebooks

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (quantized_straight_through [B, dim], codes [B, num_levels], vq_loss)."""
        entries, residuals, codes = quantize_residuals(z, self.codebooks, self.num_levels, self.normalize_codebook)

        # F.mse_loss means over K*B*dim, so scale by K to keep the per-step sum of per-step means:
        # without it the codebook's effective learning rate and the commitment weight both drop by K.
        residual_loss = F.mse_loss(entries, residuals.detach())  # move codebooks towards the residuals
        commitment_loss = F.mse_loss(residuals, entries.detach())  # move residuals towards the codebooks
        vq_loss = self.num_levels * (residual_loss + self.commitment * commitment_loss)

        # straight-through: gradients flow to the encoder as if quantization were the identity.
        quantized_st = z + (entries.sum(0) - z).detach()

        return quantized_st, codes, vq_loss

    @torch.no_grad()
    def update_(self, z: Tensor) -> None:
        """Fold one batch of latents' assignments into ``usage_ema``. Call once per real training step,
        after ``backward()``/``optimizer.step()``.
        """
        _, _, codes = quantize_residuals(z, self.codebooks, self.num_levels, self.normalize_codebook)
        self.update_usage_(codes)

    @torch.no_grad()
    def update_usage_(self, codes: Tensor) -> None:
        """Fold this batch's per-code assignment share into ``usage_ema``."""
        offsets = torch.arange(self.num_levels, device=codes.device) * self.num_codes
        flat_codes = (codes + offsets).reshape(-1)
        flat_counts = torch.bincount(flat_codes, minlength=self.num_levels * self.num_codes)
        counts = flat_counts.view(self.num_levels, self.num_codes)

        self.usage_ema.mul_(self.usage_decay).add_((1 - self.usage_decay) * counts / codes.size(0))

    @torch.no_grad()
    def expire_dead_codes_(self, z: Tensor, generator: torch.Generator, threshold: float) -> Tensor:
        """Reseed codes whose usage share fell below ``threshold`` times uniform onto observed residuals.

        ``threshold`` is a fraction of the uniform share ``1/num_codes``, so it means the same thing at
        any ``num_codes`` or batch size: 0.1 expires codes assigned less than a tenth as often as uniform.

        Returns the ``BoolTensor[num_levels, num_codes]`` mask of what was reseeded. Deliberately not
        called from ``forward``: it writes to ``codebooks``, which must not happen mid-graph.
        """
        _, residuals, _ = quantize_residuals(z, self.codebooks, self.num_levels, self.normalize_codebook)
        dead = self.usage_ema < threshold / self.num_codes  # [num_levels, num_codes]

        for k in range(self.num_levels):
            num_dead = int(dead[k].sum())
            if num_dead == 0:
                continue
            # With replacement: a heavily collapsed level can need more seeds than the batch has rows.
            picks = torch.randint(residuals.size(1), (num_dead,), generator=generator, device=z.device)
            self.codebooks[k][dead[k]] = residuals[k][picks]
            self.usage_ema[k][dead[k]] = 1.0 / self.num_codes  # a fresh code is not yet evidence of death

        return dead

    @torch.no_grad()
    def kmeans_init_(self, z: Tensor, generator: torch.Generator, max_samples: int):
        """Seed each stage's codebook with k-means centroids of that stage's residuals."""
        if z.size(0) > max_samples:
            samples = torch.randperm(z.size(0), generator=generator)[:max_samples].to(z.device)
            z = z[samples]

        residual = z

        for k in range(self.num_levels):
            centroids = kmeans(residual, self.num_codes, generator)
            self.codebooks[k].copy_(centroids)
            idx = nearest_code(residual, centroids)
            residual = residual - centroids[idx]


class EmaResidualQuantizer(ResidualQuantizer):
    """Codebooks as ``c_i = m_i / N_i``, EMAs of each code's assigned-residual sum and assignment count,
    instead of parameters descending ``residual_loss``. :meth:`update_codebooks_` maintains them, so the
    codebooks are frozen and leave the optimizer, and a code that gets assigned to no row in a batch does not move.
    """

    # Registered-buffer type declarations (nn.Module.__getattr__ returns Tensor | Module).
    cluster_size: Tensor  # [num_levels, num_codes] N_i
    cluster_sum: Tensor  # [num_levels, num_codes, dim] m_i

    def __init__(self, *args: Any, codebook_decay: float = 0.99, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.codebook_decay = codebook_decay
        self.codebooks.requires_grad_(False)
        # Seeded so m_i/N_i reproduces the initial codebook, at the weight of one batch's evidence.
        self.register_buffer("cluster_size", torch.ones(self.num_levels, self.num_codes))
        self.register_buffer("cluster_sum", self.codebooks.detach().clone())

    @torch.no_grad()
    def update_(self, z: Tensor) -> None:
        """Fold one batch of latents into ``usage_ema`` and the EMA codebooks. Call once per real
        training step, after ``backward()``/``optimizer.step()``.
        """
        _, residuals, codes = quantize_residuals(z, self.codebooks, self.num_levels, self.normalize_codebook)
        self.update_usage_(codes)
        self.update_codebooks_(residuals, codes)

    @torch.no_grad()
    def update_codebooks_(self, residuals: Tensor, codes: Tensor) -> None:
        """Fold one batch of latents into the EMAs, then recompute the codebooks from them."""
        decay = self.codebook_decay
        assigned = torch.zeros_like(self.cluster_size, dtype=torch.bool)

        for k in range(self.num_levels):
            # One-hot matmul: all V cluster sums in one kernel per level, rather than V index_add calls.
            assignment = F.one_hot(codes[:, k], self.num_codes).to(residuals.dtype)  # [B, V]
            counts = assignment.sum(0)
            assigned[k] = counts > 0
            self.cluster_size[k].mul_(decay).add_((1 - decay) * counts)
            self.cluster_sum[k].mul_(decay).add_((1 - decay) * (assignment.t() @ residuals[k]))

        # Only rows that got assigned are rewritten
        floor = torch.finfo(self.cluster_size.dtype).tiny
        centroids = self.cluster_sum / self.cluster_size.clamp_min(floor).unsqueeze(-1)
        self.codebooks.copy_(torch.where(assigned.unsqueeze(-1), centroids, self.codebooks))

    @torch.no_grad()
    def resync_accumulators_(self, rows: Tensor | None = None) -> None:
        """Re-point ``N_i, m_i`` at the current codebook rows, so ``m_i / N_i`` reproduces them.

        Must follow every direct write to ``codebooks``, or the next :meth:`update_codebooks_` reverts it
        from the replaced centroid's statistics. ``rows`` is the mask of what was written, all if ``None``.
        """
        mask = torch.ones_like(self.cluster_size, dtype=torch.bool) if rows is None else rows
        self.cluster_size[mask] = 1.0
        self.cluster_sum[mask] = self.codebooks.detach()[mask]

    @torch.no_grad()
    def expire_dead_codes_(self, z: Tensor, generator: torch.Generator, threshold: float) -> Tensor:
        """Reseed as the base class does, then re-point the EMAs at the rows it rewrote."""
        dead = super().expire_dead_codes_(z, generator, threshold)
        self.resync_accumulators_(dead)

        return dead

    @torch.no_grad()
    def kmeans_init_(self, z: Tensor, generator: torch.Generator, max_samples):
        """Seed as the base class does, then re-point the EMAs: every level was rewritten."""
        super().kmeans_init_(z, generator, max_samples)
        self.resync_accumulators_()
