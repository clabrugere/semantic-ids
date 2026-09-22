import torch
from torch import Generator, Tensor


def nearest_code(x: Tensor, codebook: Tensor) -> Tensor:
    """Index of the nearest centroid for each row of ``x`` ``[N, dim]`` -> ``[N]``.

    Uses ``argmin ||x - c||^2 = argmin ||c||^2 - 2 x·c`` (the ``||x||^2`` term is constant across
    centroids).
    """
    scores = codebook.pow(2).sum(1) - 2.0 * (x @ codebook.t())  # [N, V]
    return scores.argmin(dim=-1)  # [N]


def kmeans_once(x: Tensor, num_clusters: int, generator: Generator, iters: int) -> Tensor:
    """Single random-init run. Returns centroids ``[num_clusters, dim]``."""
    num_points, dim = x.size()
    if num_clusters > num_points:
        raise ValueError(f"num_clusters={num_clusters} exceeds num_points={num_points}")

    picks = torch.randperm(num_points, generator=generator)[:num_clusters].to(x.device)
    centroids = x[picks].clone()

    for _ in range(iters):
        assign = nearest_code(x, centroids)  # [N]
        counts = torch.bincount(assign, minlength=num_clusters)  # [num_clusters]
        sums = torch.zeros(num_clusters, dim, device=x.device).index_add_(0, assign, x)  # [num_clusters, dim]
        centroids = sums / counts.clamp(min=1).unsqueeze(1)  # [num_clusters, dim]

        empty = counts == 0  # revive dead centroids on random points
        if bool(empty.any()):
            revive = torch.randint(0, num_points, (int(empty.sum()),), generator=generator).to(x.device)
            centroids[empty] = x[revive]

    return centroids  # [num_clusters, dim]


def kmeans(x: Tensor, num_clusters: int, generator: Generator, iters: int = 10, num_restarts: int = 5) -> Tensor:
    """Multiple random-init runs. Returns best centroids ``[num_clusters, dim]``.

    Used to seed codebooks: a random small-value init leaves most entries unused, since the
    encoder's outputs cluster far from them and only a few centroids ever win the nearest-centroid
    argmin, collapsing the catalog onto a handful of prefixes.
    """
    if num_restarts < 1:
        raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")

    best_centroids = kmeans_once(x, num_clusters, generator, iters)
    best_loss = (x - best_centroids[nearest_code(x, best_centroids)]).pow(2).sum()

    for _ in range(num_restarts - 1):
        centroids = kmeans_once(x, num_clusters, generator, iters)
        loss = (x - centroids[nearest_code(x, centroids)]).pow(2).sum()

        if loss < best_loss:
            best_centroids, best_loss = centroids, loss

    return best_centroids  # [num_clusters, dim]
