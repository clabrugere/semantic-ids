import pytest
import torch
import torch.nn.functional as F

from semantic_ids.kmeans import nearest_code
from semantic_ids.quantization import EmaResidualQuantizer, ResidualQuantizer, quantize_residuals

# quantize_residuals


@pytest.mark.parametrize("normalize", [True, False])
def test_quantize_residuals_recursion_and_shapes(normalize):
    """The chain's defining property: residuals[k+1] == residuals[k] - entries[k], residuals[0] is latent,
    and the entries sum to the quantized vector."""
    g = torch.Generator().manual_seed(0)
    num_levels, num_codes, dim = 3, 7, 4
    codebooks = torch.randn(num_levels, num_codes, dim, generator=g)
    latent = torch.randn(6, dim, generator=g)

    entries, residuals, codes = quantize_residuals(latent, codebooks, num_levels, normalize)

    assert entries.size() == (num_levels, 6, dim)
    assert residuals.size() == (num_levels, 6, dim)
    assert codes.size() == (6, num_levels)
    assert torch.equal(residuals[0], latent)
    for k in range(num_levels - 1):
        assert torch.equal(residuals[k + 1], residuals[k] - entries[k])


@pytest.mark.parametrize("normalize", [True, False])
def test_quantize_residuals_entries_are_the_selected_codebook_rows(normalize):
    g = torch.Generator().manual_seed(1)
    num_levels, num_codes, dim = 2, 5, 3
    codebooks = torch.randn(num_levels, num_codes, dim, generator=g)
    latent = torch.randn(8, dim, generator=g)

    entries, residuals, codes = quantize_residuals(latent, codebooks, num_levels, normalize)

    for k in range(num_levels):
        assert torch.equal(entries[k], codebooks[k][codes[:, k]])
        search_codebook = F.normalize(codebooks[k], dim=-1) if normalize else codebooks[k]
        assert torch.equal(codes[:, k], nearest_code(residuals[k], search_codebook))


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("num_levels", [1, 2, 3, 4])
def test_vq_loss_is_the_sum_of_per_step_means_not_their_mean(normalize, num_levels):
    """The stacked reduction's trap: F.mse_loss over a [K, B, dim] tensor means over K*B*dim, which
    is 1/K of the per-step sum the loop accumulated. Unscaled, the codebook's effective learning
    rate and the commitment weight both silently drop K-fold."""
    commitment = 0.25
    rq = ResidualQuantizer(
        num_levels=num_levels, num_codes=6, dim=4, commitment=commitment, normalize_codebook=normalize
    )
    latent = torch.randn(12, 4, generator=torch.Generator().manual_seed(3))

    _, _, vq_loss = rq(latent)

    entries, residuals, _ = quantize_residuals(latent, rq.codebooks, num_levels, normalize)
    expected = sum(
        F.mse_loss(entries[k], residuals[k]) + commitment * F.mse_loss(residuals[k], entries[k])
        for k in range(num_levels)
    )
    assert vq_loss.item() == pytest.approx(expected.item(), rel=1e-6)


# ResidualQuantizer


@pytest.mark.parametrize("normalize", [True, False])
def test_forward_shapes_and_code_range(normalize):
    rq = ResidualQuantizer(num_levels=3, num_codes=5, dim=4, normalize_codebook=normalize)
    latent = torch.randn(6, 4)
    quantized_st, codes, vq_loss = rq(latent)

    assert quantized_st.size() == latent.size()
    assert codes.size() == (6, 3)
    assert vq_loss.dim() == 0
    assert bool((codes >= 0).all()) and bool((codes < 5).all())


@pytest.mark.parametrize("normalize", [True, False])
def test_straight_through_gradient_is_identity(normalize):
    rq = ResidualQuantizer(num_levels=2, num_codes=5, dim=3, normalize_codebook=normalize)
    latent = torch.randn(4, 3, requires_grad=True)
    quantized_st, _, _ = rq(latent)

    quantized_st.sum().backward()
    assert torch.equal(latent.grad, torch.ones_like(latent))


@pytest.mark.parametrize("normalize", [True, False])
def test_vq_loss_gradient_flows_to_codebooks(normalize):
    rq = ResidualQuantizer(num_levels=2, num_codes=5, dim=3, normalize_codebook=normalize)
    latent = torch.randn(4, 3)
    _, _, vq_loss = rq(latent)

    vq_loss.backward()
    assert rq.codebooks.grad is not None
    assert bool((rq.codebooks.grad != 0).any())


@pytest.mark.parametrize("normalize", [True, False])
def test_kmeans_init_seeds_codebook_to_match_data_clusters(normalize):
    gen = torch.Generator().manual_seed(0)
    rq = ResidualQuantizer(num_levels=1, num_codes=2, dim=2, normalize_codebook=normalize)
    cluster_a = torch.tensor([5.0, 5.0]) + 0.01 * torch.randn(20, 2)
    cluster_b = torch.tensor([-5.0, -5.0]) + 0.01 * torch.randn(20, 2)
    latent = torch.cat([cluster_a, cluster_b])

    rq.kmeans_init_(latent, gen, max_samples=50_000)
    quantized_st, _, _ = rq(latent)
    assert torch.allclose(quantized_st, latent, atol=0.1)


# EmaResidualQuantizer


def seeded_ema_quantizer(codebook_rows, codebook_decay=0.99):
    """num_levels=1 EmaResidualQuantizer seeded to these exact rows, with accumulators resynced to
    match. This is the setup every EMA-mechanics test below starts from. (Resyncing is a no-op when
    codebook_decay=0, since the next update discards prior accumulator state anyway; harmless to
    always do it.)
    """
    rq = EmaResidualQuantizer(
        num_levels=1, num_codes=len(codebook_rows), dim=len(codebook_rows[0]), codebook_decay=codebook_decay
    )
    rq.codebooks.copy_(torch.tensor([codebook_rows]))
    rq.resync_accumulators_()
    return rq


def test_ema_update_lands_on_the_exact_cluster_mean():
    """At decay 0 the EMA keeps only this batch, so c_i must equal the mean of the residuals assigned
    to it."""
    rq = seeded_ema_quantizer([[-1.0, 0.0], [1.0, 0.0]], codebook_decay=0.0)
    latent = torch.tensor([[-2.0, 1.0], [-2.0, 3.0], [4.0, -1.0]])  # first two -> code 0, third -> code 1

    rq.update_(latent)

    assert torch.allclose(rq.codebooks[0][0], torch.tensor([-2.0, 2.0]))  # mean of the first two
    assert torch.allclose(rq.codebooks[0][1], torch.tensor([4.0, -1.0]))  # the third alone


def test_ema_update_leaves_an_unassigned_code_exactly_where_it_was():
    """N_i and m_i decay at the same rate, so a code that wins nothing does not move at all."""
    rq = seeded_ema_quantizer([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    unassigned = rq.codebooks[0][2].clone()
    latent = torch.tensor([[0.1, 0.0], [9.0, 0.5]])  # nothing is nearest to code 2

    for _ in range(20):
        rq.update_(latent)

    assert torch.equal(rq.codebooks[0][2], unassigned)


def test_ema_does_not_drift_a_never_winning_code_as_its_accumulators_underflow():
    """Over a run's worth of quiet steps N_i and m_i decay into the denormals, where m_i/N_i loses
    precision and can read 0. Parking the code on the origin, which a normalized argmin can never pick
    again. Skipping the write for unassigned rows is what holds the centroid exactly still."""
    rq = seeded_ema_quantizer([[0.0, 5.0], [10.0, 0.0]])
    quiet = rq.codebooks[0][0].clone()
    latent = torch.tensor([[10.0, 0.0]])  # only code 1 ever wins

    for _ in range(12_000):
        rq.update_(latent)

    assert float(rq.cluster_size[0][0]) < 1e-30, "the test no longer reaches the underflow regime"
    assert torch.equal(rq.codebooks[0][0], quiet)


def test_expiry_reseeds_survive_the_next_ema_update():
    """The trap the subclass exists to close: without resyncing, m_i/N_i still describes the centroid
    expiry just replaced, so the next update silently reverts the reseed."""
    rq = seeded_ema_quantizer([[0.0, 0.0], [50.0, 50.0]])
    rq.usage_ema[0][1] = 0.0  # code 1 has stopped winning, so expiry will reseed it
    latent = torch.tensor([[1.0, 1.0], [1.2, 0.8]])

    dead = rq.revive_dead_codes_(latent, torch.Generator().manual_seed(0), threshold=0.1)
    reseeded = rq.codebooks[0][1].clone()
    rq.update_(latent)

    assert bool(dead[0][1]) and not bool(dead[0][0])
    assert not torch.allclose(reseeded, torch.tensor([50.0, 50.0])), "expiry did not move the code"
    assert torch.allclose(rq.cluster_sum[0][1] / rq.cluster_size[0][1], rq.codebooks[0][1])


def test_update_also_folds_assignment_counts_into_usage_ema():
    """update_() must do everything forward()'s removed training-mode branch used to: usage_ema too,
    not just the codebook EMA that update_codebooks_ covers alone."""
    rq = seeded_ema_quantizer([[-1.0, 0.0], [1.0, 0.0]])
    latent = torch.tensor([[-2.0, 1.0], [-2.0, 3.0], [4.0, -1.0]])  # code 0 wins twice, code 1 once

    rq.update_(latent)

    expected = 0.99 * torch.full((2,), 0.5) + 0.01 * torch.tensor([2 / 3, 1 / 3])
    assert torch.allclose(rq.usage_ema[0], expected)


def test_kmeans_init_leaves_the_accumulators_describing_the_codebooks():
    rq = EmaResidualQuantizer(num_levels=2, num_codes=2, dim=2)
    latent = torch.cat([torch.tensor([5.0, 5.0]) + 0.01 * torch.randn(20, 2), -5.0 + 0.01 * torch.randn(20, 2)])

    rq.kmeans_init_(latent, torch.Generator().manual_seed(0), max_samples=50_000)

    assert torch.allclose(rq.cluster_sum / rq.cluster_size.unsqueeze(-1), rq.codebooks)


def test_ema_codebooks_are_frozen_and_drop_the_codebook_term_from_the_loss():
    """The codebooks leave the optimizer, so ``residual_loss`` has no gradient path and ``vq_loss``
    trains the encoder by commitment alone. The dead term still shows up in the logged scalar. Left in
    training mode (the default) on purpose: forward() no longer EMA-updates codebooks as a side effect,
    so this holds without an eval() workaround."""
    commitment, num_levels = 0.25, 2
    rq = EmaResidualQuantizer(num_levels=num_levels, num_codes=5, dim=3, commitment=commitment)
    latent = torch.randn(8, 3, requires_grad=True)

    _, _, vq_loss = rq(latent)
    vq_loss.backward()

    assert rq.codebooks.grad is None and not rq.codebooks.requires_grad
    entries, residuals, _ = quantize_residuals(latent, rq.codebooks, num_levels, rq.normalize_codebook)
    expected = sum(commitment * F.mse_loss(residuals[k], entries[k]) for k in range(num_levels))
    assert latent.grad is not None
    assert vq_loss.item() == pytest.approx(expected.item() + num_levels * F.mse_loss(entries, residuals).item())


# forward() purity


@pytest.mark.parametrize("training", [True, False])
def test_forward_never_mutates_usage_ema(training):
    """update_() is the only thing allowed to touch usage_ema now -- a no_grad peek (encode, a logging
    pass, a checkpoint recompute) must never mutate it as a side effect of calling forward()."""
    rq = ResidualQuantizer(num_levels=2, num_codes=5, dim=3)
    rq.train(training)
    before = rq.usage_ema.clone()

    rq(torch.randn(6, 3))

    assert torch.equal(rq.usage_ema, before)


@pytest.mark.parametrize("training", [True, False])
def test_ema_forward_never_mutates_state(training):
    """Same guarantee for the EMA subclass, extended to the buffers update_codebooks_ writes: codebooks,
    cluster_size, and cluster_sum must all be untouched by forward() alone, in either train or eval mode."""
    rq = EmaResidualQuantizer(num_levels=2, num_codes=5, dim=3)
    rq.train(training)
    usage_before = rq.usage_ema.clone()
    codebooks_before = rq.codebooks.clone()
    cluster_size_before = rq.cluster_size.clone()
    cluster_sum_before = rq.cluster_sum.clone()

    rq(torch.randn(6, 3))

    assert torch.equal(rq.usage_ema, usage_before)
    assert torch.equal(rq.codebooks, codebooks_before)
    assert torch.equal(rq.cluster_size, cluster_size_before)
    assert torch.equal(rq.cluster_sum, cluster_sum_before)
