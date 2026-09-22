import torch

from semantic_ids.kmeans import kmeans, nearest_code


def test_nearest_code_worked_example():
    codebook = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    x = torch.tensor([[1.0, 1.0], [9.0, 1.0], [1.0, 9.0]])  # each closest to a distinct centroid
    assert torch.equal(nearest_code(x, codebook), torch.tensor([0, 1, 2]))


def test_nearest_code_matches_brute_force_cdist():
    torch.manual_seed(0)
    x = torch.randn(20, 5)
    codebook = torch.randn(7, 5)
    expected = torch.cdist(x, codebook).argmin(dim=-1)
    assert torch.equal(nearest_code(x, codebook), expected)


def test_kmeans_revives_empty_clusters_without_nan():
    """2 tight clusters but 5 requested centroids: several centroids start with zero assignments,
    exercising the dead-centroid revival path."""
    gen = torch.Generator().manual_seed(1)
    x = torch.cat([torch.zeros(6, 2), torch.full((6, 2), 10.0)])
    centroids = kmeans(x, num_clusters=5, generator=gen)
    assert centroids.size() == (5, 2)
    assert torch.isfinite(centroids).all()


def test_kmeans_deterministic_given_seeded_generator():
    x = torch.randn(30, 3)
    a = kmeans(x, num_clusters=4, generator=torch.Generator().manual_seed(42))
    b = kmeans(x, num_clusters=4, generator=torch.Generator().manual_seed(42))
    assert torch.equal(a, b)
