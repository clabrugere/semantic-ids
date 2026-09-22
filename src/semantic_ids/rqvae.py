from collections.abc import Sequence
from enum import StrEnum

import torch
from torch import Tensor, nn

from semantic_ids.quantization import EmaResidualQuantizer, ResidualQuantizer
from semantic_ids.semantic_ids import SemanticIds


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
        encoder_hidden: Sequence[int],
        decoder_hidden: Sequence[int],
        commitment: float = 0.25,
        normalize_codebook: bool = True,
        codebook_update: CodebookUpdate = CodebookUpdate.GRADIENT,
    ):
        super().__init__()
        quantizer = EmaResidualQuantizer if codebook_update is CodebookUpdate.EMA else ResidualQuantizer
        self.encoder = MLP(input_dim, latent_dim, encoder_hidden)
        self.quantizer = quantizer(num_levels, num_codes, latent_dim, commitment, normalize_codebook)
        self.decoder = MLP(latent_dim, input_dim, decoder_hidden)

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
    def encode_with_residual_norm(self, x: Tensor, batch_size: int | None = None) -> tuple[Tensor, Tensor]:
        """Codes plus each row's leftover residual norm ``‖z - quantized‖`` after all ``K-1`` levels.

        Everyone sharing a full prefix reconstructs to the same ``quantized`` point, so this norm is
        each item's distance from its group's shared centroid.
        """
        batch_size = batch_size or x.size(0)
        codes_chunks, residual_norm_chunks = [], []
        for chunk in x.split(batch_size):
            z = self.encoder(chunk)
            quantized, chunk_codes, _ = self.quantizer(z)
            codes_chunks.append(chunk_codes)
            residual_norm_chunks.append((z - quantized).norm(dim=-1))

        return torch.cat(codes_chunks), torch.cat(residual_norm_chunks)

    @torch.no_grad()
    def encode(self, x: Tensor, batch_size: int | None = None) -> Tensor:
        """Encode inputs: embeddings -> codes [B, num_levels]."""
        codes, _ = self.encode_with_residual_norm(x, batch_size)
        return codes

    @torch.no_grad()
    def export_semantic_ids(self, data: Tensor, batch_size: int | None = None) -> SemanticIds:
        """Encode every row of ``data``, disambiguate collisions, and package as a :class:`~semantic_ids.semantic_ids.SemanticIds`.

        Within a collision group, the disambiguation ordinal is ranked by ascending distance to the group's centroid.
        """
        self.eval()
        prefixes, residual_norm = self.encode_with_residual_norm(data, batch_size)  # [M, K-1], [M]
        return SemanticIds.from_codes(prefixes, self.quantizer.num_codes, sort_key=residual_norm)
