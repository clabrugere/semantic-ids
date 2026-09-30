from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from dataclasses import dataclass

from semantic_ids.rqvae import CodebookUpdate


@dataclass(frozen=True)
class Config:
    data: str
    seed: int
    out: str
    max_steps: int
    batch_size: int
    lr: float
    val_fraction: float
    eval_every: int
    latent_dim: int
    num_levels: int
    num_codes: int
    encoder_hidden_dims: tuple[int, ...]
    decoder_hidden_dims: tuple[int, ...]
    commitment: float
    normalize_codebook: bool
    codebook_update: CodebookUpdate
    expiry_threshold: float
    revive_every: int = 200

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:

        parser.add_argument("--data", type=str, required=True, help="content embeddings Parquet export")
        parser.add_argument("--out", type=str, required=True, help="Path to save the generated semantic IDs")
        parser.add_argument("--seed", type=int, default=1234)

        parser.add_argument("--max-steps", type=int, default=10_000)
        parser.add_argument("--batch-size", type=int, default=1024)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--val-fraction", type=float, default=0.2)
        parser.add_argument("--eval-every", type=int, default=2000)

        parser.add_argument("--latent-dim", type=int, default=32)
        parser.add_argument("--num-levels", type=int, default=4, help="Number of quantization levels")
        parser.add_argument("--num-codes", type=int, default=512, help="Number of codes per level")
        parser.add_argument("--encoder-hidden-dims", type=int, nargs="+", default=(256, 128))
        parser.add_argument("--decoder-hidden-dims", type=int, nargs="+", default=(128, 256))
        parser.add_argument("--commitment", type=float, default=0.25)
        parser.add_argument("--normalize-codebook", action=BooleanOptionalAction, default=True)
        parser.add_argument(
            "--codebook-update", type=CodebookUpdate, choices=list(CodebookUpdate), default=CodebookUpdate.GRADIENT
        )
        parser.add_argument("--expiry-threshold", type=float, default=0.1)
        parser.add_argument("--revive-every", type=int, default=200)

    @classmethod
    def from_args(cls, args: Namespace) -> Config:
        return cls(
            data=args.data,
            seed=args.seed,
            out=args.out,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            val_fraction=args.val_fraction,
            eval_every=args.eval_every,
            latent_dim=args.latent_dim,
            num_levels=args.num_levels,
            num_codes=args.num_codes,
            encoder_hidden_dims=tuple(args.encoder_hidden_dims),
            decoder_hidden_dims=tuple(args.decoder_hidden_dims),
            commitment=args.commitment,
            normalize_codebook=args.normalize_codebook,
            codebook_update=args.codebook_update,
            expiry_threshold=args.expiry_threshold,
            revive_every=args.revive_every,
        )
