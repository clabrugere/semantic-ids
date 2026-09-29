from collections.abc import Sequence
from enum import StrEnum

import torch
from torch import Tensor, nn

from semantic_ids.quantization import EmaResidualQuantizer, ResidualQuantizer


class MLP(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Sequence[int]):
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        super().__init__(*layers)


class CodebookUpdate(StrEnum):
    GRADIENT = "gradient"  # parameters descending the VQ loss inside the same optimizer as the MLPs
    EMA = "ema"  # a running mean of each code's assigned residuals, maintained outside the optimizer


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_levels: int,
        num_codes: int,
        encoder_hidden_dims: Sequence[int],
        decoder_hidden_dims: Sequence[int],
        commitment: float = 0.25,
        normalize_codebook: bool = True,
        codebook_update: CodebookUpdate = CodebookUpdate.GRADIENT,
    ):
        super().__init__()
        quantizer = EmaResidualQuantizer if codebook_update is CodebookUpdate.EMA else ResidualQuantizer
        self.encoder = MLP(input_dim, latent_dim, encoder_hidden_dims)
        self.quantizer = quantizer(num_levels, num_codes, latent_dim, commitment, normalize_codebook)
        self.decoder = MLP(latent_dim, input_dim, decoder_hidden_dims)

    @property
    def num_levels(self) -> int:
        return self.quantizer.num_levels

    @property
    def num_codes(self) -> int:
        return self.quantizer.num_codes

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(reconstructed, codes, vq_loss)``."""
        z = self.encoder(x)
        quantized, codes, vq_loss = self.quantizer(z)
        reconstructed = self.decoder(quantized)

        return reconstructed, codes, vq_loss

    @torch.no_grad()
    def init_codebooks_(
        self,
        x: Tensor,
        generator: torch.Generator,
        batch_size: int | None = None,
        kmeans_max_samples: int = 50_000,
    ):
        """Seed the codebooks from the encoder's latents over the whole catalog, before training."""
        batch_size = batch_size or x.size(0)
        latents = torch.cat([self.encoder(chunk) for chunk in x.split(batch_size)])  # [N, latent_dim]
        self.quantizer.kmeans_init_(latents, generator, kmeans_max_samples)

    @torch.no_grad()
    def encode_with_residual_norm(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Codes plus each row's leftover residual norm ``‖z - quantized‖`` after all ``K-1`` levels.

        Every item sharing a full prefix reconstructs to the same ``quantized`` point, so this norm is
        the distance from its group's centroid.
        """
        z = self.encoder(x)
        quantized, chunk_codes, _ = self.quantizer(z)
        residual_norm = (z - quantized).norm(dim=-1)

        return chunk_codes, residual_norm

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode inputs: embeddings -> codes [B, num_levels]."""
        codes, _ = self.encode_with_residual_norm(x)
        return codes
