import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import pyarrow.parquet as pq
import torch
from config import Config
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from semantic_ids.rqvae import RQVAE
from semantic_ids.semantic_ids import SemanticIds

logger = logging.getLogger(__name__)


class EvalResults(NamedTuple):
    reconstruction_loss: float
    used_codes: Tensor
    effective_codes: Tensor
    dead_code_fraction: Tensor


def load_embeddings(path: str | Path) -> Tensor:
    table = pq.read_table(path, columns=["embedding"])
    column = table.column("embedding").combine_chunks()
    flat = column.flatten().to_numpy(zero_copy_only=True)

    return torch.from_numpy(flat).view(table.num_rows, -1)


def code_usage(codes: Tensor, num_codes: int) -> tuple[Tensor, Tensor, Tensor]:
    m, k = codes.size()
    # One bincount for all levels: offset level k's codes into their own [k*V, (k+1)*V) range
    flat = (codes + torch.arange(k, device=codes.device) * num_codes).reshape(-1)  # [M*K]
    counts = torch.bincount(flat, minlength=k * num_codes).view(k, num_codes)
    p = counts / m

    used_codes = (counts > 0).sum(1)
    effective_codes = 1.0 / p.pow(2).sum(1)
    dead_code_fraction = 1.0 - used_codes / num_codes

    return used_codes, effective_codes, dead_code_fraction


@torch.no_grad()
def evaluate(model: RQVAE, val_dataloader: DataLoader, device: torch.device) -> EvalResults:
    model.eval()
    reconstruction_loss = torch.tensor(0.0, device=device)
    all_codes = []
    for val_batch in val_dataloader:
        val_batch = val_batch[0].to(device)
        reconstructed, codes, _ = model(val_batch)
        reconstruction_loss += F.mse_loss(reconstructed, val_batch)
        all_codes.append(codes)

    reconstruction_loss /= len(val_dataloader)
    used_codes, effective_codes, dead_code_fraction = code_usage(torch.cat(all_codes, dim=0), model.quantizer.num_codes)

    model.train()

    return EvalResults(
        reconstruction_loss=float(reconstruction_loss),
        used_codes=used_codes.cpu(),
        effective_codes=effective_codes.cpu(),
        dead_code_fraction=dead_code_fraction.cpu(),
    )


def train(
    model: RQVAE,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    max_steps: int,
    lr: float,
    expiry_threshold: float,
    expire_every: int,
    device: torch.device,
    eval_every: int,
    log_every: int = 1000,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator(device=device)
    batch_iterator = iter(train_dataloader)

    for step in range(1, max_steps + 1):
        try:
            batch = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(train_dataloader)
            batch = next(batch_iterator)

        batch = batch[0].to(device)
        reconstructed, _, vq_loss = model(batch)
        reconstruction_loss = F.mse_loss(reconstructed, batch)
        loss = reconstruction_loss + vq_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update code usage and in the case of EMA, the codebooks
        model.quantizer.update_(model.encoder(batch))

        if expiry_threshold > 0 and step % expire_every == 0:
            model.quantizer.expire_dead_codes_(model.encoder(batch), generator, expiry_threshold)

        if step == 1 or step % log_every == 0:
            logger.info(
                "train | step=%d/%d | loss=%.6f | recon=%.6f | vq=%.6f",
                step,
                max_steps,
                loss.item(),
                reconstruction_loss.item(),
                vq_loss.item(),
            )

        if step % eval_every == 0:
            eval_results = evaluate(model, val_dataloader, device)
            code_summary = " | ".join(
                f"L{level}: {used}/{model.quantizer.num_codes} used, {effective:.1f} effective, {dead:.1%} dead"
                for level, (used, effective, dead) in enumerate(
                    zip(
                        eval_results.used_codes.tolist(),
                        eval_results.effective_codes.tolist(),
                        eval_results.dead_code_fraction.tolist(),
                        strict=True,
                    )
                )
            )
            logger.info(
                "valid | step=%d/%d | recon=%.6f | %s",
                step,
                max_steps,
                eval_results.reconstruction_loss,
                code_summary,
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)

    Config.add_arguments(parser)
    args = parser.parse_args()
    config = Config.from_args(args)

    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("setup | device=%s", device)

    data = load_embeddings(config.data)
    inds = torch.randperm(data.size(0), generator=generator)
    train_size = int((1 - config.val_fraction) * data.size(0))
    train_inds = inds[:train_size]
    val_inds = inds[train_size:]
    train_dataloader = DataLoader(TensorDataset(data[train_inds]), batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(TensorDataset(data[val_inds]), batch_size=config.batch_size)

    logger.info(
        "data | items=%s (train=%s val=%s) | dimensions=%d",
        f"{data.size(0):,}",
        f"{train_inds.size(0):,}",
        f"{val_inds.size(0):,}",
        data.size(1),
    )
    logger.info(
        "model | levels=%d | codes/level=%d | max_steps=%s",
        config.num_levels,
        config.num_codes,
        f"{config.max_steps:,}",
    )

    model = RQVAE(
        input_dim=data.size(1),
        latent_dim=config.latent_dim,
        num_levels=config.num_levels,
        num_codes=config.num_codes,
        encoder_hidden_dims=config.encoder_hidden_dims,
        decoder_hidden_dims=config.decoder_hidden_dims,
        commitment=config.commitment,
        normalize_codebook=config.normalize_codebook,
        codebook_update=config.codebook_update,
    )
    model.init_codebooks_(data, generator, batch_size=8192)
    model.to(device)

    logger.info("train | starting")
    train(
        model,
        train_dataloader,
        val_dataloader,
        config.max_steps,
        config.lr,
        config.expiry_threshold,
        config.expire_every,
        device,
        config.eval_every,
    )
    logger.info("train | complete")

    logger.info("export | generating code sequences")
    dataloader = DataLoader(TensorDataset(data), batch_size=config.batch_size)
    codes_sequences, residual_norms = [], []

    for batch in dataloader:
        batch = batch[0].to(device)
        prefix, residual_norm = model.encode_with_residual_norm(batch)
        codes_sequences.append(prefix.cpu())
        residual_norms.append(residual_norm.cpu())

    codes_sequences = torch.cat(codes_sequences, dim=0)
    residual_norms = torch.cat(residual_norms, dim=0)
    logger.info("export | code sequences generated | shape=(%s, %s)", *codes_sequences.size())

    semantic_ids = SemanticIds.from_codes(codes_sequences, model.num_codes, sort_key=residual_norms)
    logger.info("export | semantic IDs generated | count=%s", len(semantic_ids))

    torch.save(semantic_ids, config.out)
    logger.info("export | semantic IDs saved to %s", config.out)


if __name__ == "__main__":
    main()
