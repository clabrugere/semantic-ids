import pytest
import torch
import torch.nn.functional as F

from semantic_ids.rqvae import RQVAE, CodebookUpdate


def test_rqvae_forward_shapes():
    model = RQVAE(
        input_dim=8,
        latent_dim=4,
        num_levels=3,
        num_codes=6,
        encoder_hidden_dims=[4],
        decoder_hidden_dims=[4],
    )
    x = torch.randn(5, 8)
    reconstructed, codes, vq_loss = model(x)

    assert reconstructed.size() == x.size()
    assert codes.size() == (5, 3)
    assert vq_loss.dim() == 0
    assert bool((codes >= 0).all()) and bool((codes < 6).all())


def test_rqvae_encode_matches_forward_codes():
    model = RQVAE(
        input_dim=8,
        latent_dim=4,
        num_levels=3,
        num_codes=6,
        encoder_hidden_dims=[4],
        decoder_hidden_dims=[4],
    )
    x = torch.randn(5, 8)

    _, codes, _ = model(x)
    encoded = model.encode(x)
    assert torch.equal(encoded, codes)


def test_rqvae_encode_disables_grad_tracking():
    model = RQVAE(
        input_dim=8,
        latent_dim=4,
        num_levels=2,
        num_codes=4,
        encoder_hidden_dims=[4],
        decoder_hidden_dims=[4],
    )
    x = torch.randn(3, 8, requires_grad=True)
    assert not model.encode(x).requires_grad


def test_encode_with_residual_norm_matches_manual_computation():
    model = RQVAE(
        input_dim=8,
        latent_dim=4,
        num_levels=3,
        num_codes=6,
        encoder_hidden_dims=[4],
        decoder_hidden_dims=[4],
    )
    x = torch.randn(5, 8)

    codes, residual_norm = model.encode_with_residual_norm(x)
    z = model.encoder(x)
    quantized, expected_codes, _ = model.quantizer(z)

    assert torch.equal(codes, expected_codes)
    assert torch.allclose(residual_norm, (z - quantized).norm(dim=-1))


def test_init_codebooks_seeds_from_encoder_latents():
    model = RQVAE(
        input_dim=8,
        latent_dim=4,
        num_levels=2,
        num_codes=5,
        encoder_hidden_dims=[4],
        decoder_hidden_dims=[4],
    )
    initial_codebooks = model.quantizer.codebooks.clone()
    x = torch.randn(50, 8)

    model.init_codebooks_(x, torch.Generator().manual_seed(0))

    assert not torch.equal(model.quantizer.codebooks, initial_codebooks)


def test_forward_gradients_reach_encoder_and_decoder_parameters():
    """Only the quantizer's straight-through path is proven elsewhere (test_quantization.py); nothing
    confirms the gradient actually completes the round trip through the encoder and decoder MLPs too."""
    model = RQVAE(
        input_dim=8,
        latent_dim=4,
        num_levels=2,
        num_codes=5,
        encoder_hidden_dims=[6],
        decoder_hidden_dims=[6],
    )
    x = torch.randn(5, 8)

    reconstructed, _, vq_loss = model(x)
    (F.mse_loss(reconstructed, x) + vq_loss).backward()

    for module in (model.encoder, model.decoder):
        for param in module.parameters():
            assert param.grad is not None
            assert bool((param.grad != 0).any())


@pytest.mark.parametrize("codebook_update", [CodebookUpdate.GRADIENT, CodebookUpdate.EMA])
def test_encode_never_mutates_quantizer_state_even_in_training_mode(codebook_update):
    """encode() is @torch.no_grad() and never calls self.eval() -- that used to matter, because the
    quantizer's forward() auto-updated usage_ema (and, under EMA codebooks, the codebooks themselves)
    whenever self.training was True. forward() is pure now, so a no_grad peek can't mutate state
    regardless of train/eval mode; only quantizer.update_() may."""
    model = RQVAE(
        input_dim=8,
        latent_dim=4,
        num_levels=2,
        num_codes=5,
        encoder_hidden_dims=[4],
        decoder_hidden_dims=[4],
        codebook_update=codebook_update,
    )
    x = torch.randn(5, 8)

    model.train()
    usage_before = model.quantizer.usage_ema.clone()
    codebooks_before = model.quantizer.codebooks.clone()

    model.encode(x)

    assert torch.equal(model.quantizer.usage_ema, usage_before)
    assert torch.equal(model.quantizer.codebooks, codebooks_before)
