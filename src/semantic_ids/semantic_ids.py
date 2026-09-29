from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class SemanticIds:
    """``M`` items, each with a unique ``K``-code semantic ID over a codebook of ``V`` codes.

    ``eq=False``: the default dataclass ``__eq__`` would compare tensor fields with ``==``, which
    raises on any tensor with more than one element. Compare with :func:`torch.equal` instead.

    Args:
        semantic_ids: ``LongTensor[M, K]``. Rows must be unique.
        item_ids: ``LongTensor[M]``, aligned with ``semantic_ids`` rows.
        num_codes: ``V``; codes are in ``[0, V)``.
    """

    semantic_ids: Tensor
    item_ids: Tensor
    num_codes: int

    def __post_init__(self) -> None:
        if self.semantic_ids.dim() != 2:
            raise ValueError(f"semantic_ids must be 2-D [M, K], got shape {self.semantic_ids.size()}")
        if self.item_ids.dim() != 1:
            raise ValueError(f"item_ids must be 1-D [M], got shape {self.item_ids.size()}")
        if self.semantic_ids.size(0) != self.item_ids.size(0):
            raise ValueError(
                f"semantic_ids and item_ids first dimension must match: {self.semantic_ids.size(0)} vs {self.item_ids.size(0)}"
            )
        if self.num_codes < 1:
            raise ValueError(f"num_codes must be >= 1, got {self.num_codes}")
        if self.semantic_ids.size(0) < 1:
            raise ValueError("semantic_ids must hold at least one item")
        if self.semantic_ids.size(1) < 1:
            raise ValueError("semantic_ids must have at least one level")

    def __len__(self) -> int:
        return self.semantic_ids.size(0)

    @property
    def num_items(self) -> int:
        return self.semantic_ids.size(0)

    @property
    def num_levels(self) -> int:
        return self.semantic_ids.size(1)

    @property
    def num_disambiguation_codes(self) -> int:
        """Size of the largest collision group ``c_K``."""
        if self.num_levels == 1:
            return self.num_items  # no prefix columns at all: every item shares the one group

        _, counts = torch.unique(self.semantic_ids[:, :-1], dim=0, return_counts=True)
        return int(counts.max())

    def save(self, path: str | Path) -> None:
        """Write ``{semantic_ids, item_ids, num_codes}`` as a plain dict."""
        torch.save({"semantic_ids": self.semantic_ids, "item_ids": self.item_ids, "num_codes": self.num_codes}, path)

    @classmethod
    def load(cls, path: str | Path) -> SemanticIds:
        """Read a catalog written by :meth:`save`."""
        raw = torch.load(path)
        return cls(semantic_ids=raw["semantic_ids"], item_ids=raw["item_ids"], num_codes=raw["num_codes"])

    @classmethod
    def from_codes(
        cls,
        prefixes: Tensor,
        num_codes: int,
        item_ids: Tensor | None = None,
        sort_key: Tensor | None = None,
    ) -> SemanticIds:
        """Append the ordinal disambiguation code to ``prefixes`` ``[M, K-1]`` and build :class:`SemanticIds`."""
        disambiguated, _ = assign_disambiguation(prefixes, sort_key)
        semantic_ids = torch.cat([prefixes, disambiguated.unsqueeze(-1)], dim=1)  # [M, K]
        if item_ids is None:
            item_ids = torch.arange(semantic_ids.size(0), device=semantic_ids.device)

        return cls(semantic_ids=semantic_ids, item_ids=item_ids, num_codes=num_codes)


def assign_disambiguation(prefixes: Tensor, sort_key: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Assign an ordinal 0,1,2,... within each group of identical prefix rows.

    Returns ``(disambiguation_codes [M], per-group counts)``. Vectorized: group rows by unique prefix,
    then compute each row's rank within its group.

    Args:
        sort_key: ``FloatTensor[M]`` or ``None``. Within a group, rows are ranked by ascending
            ``sort_key`` (ordinal 0 = smallest). ``None`` (default) ranks by input row order instead.
    """
    num_items = prefixes.size(0)
    _, inverse, counts = torch.unique(prefixes, dim=0, return_inverse=True, return_counts=True)

    if sort_key is None:
        order = torch.argsort(inverse, stable=True)  # rows grouped by ascending group id
    else:
        # Sort by the tie-break key first, then stably re-sort by group id
        key_order = torch.argsort(sort_key, stable=True)
        order = key_order[torch.argsort(inverse[key_order], stable=True)]

    group_start = torch.zeros_like(counts)
    group_start[1:] = torch.cumsum(counts, dim=0)[:-1]
    within_group = torch.arange(num_items, device=prefixes.device) - group_start[inverse[order]]
    disambiguated = torch.empty(num_items, dtype=torch.long, device=prefixes.device)
    disambiguated[order] = within_group

    return disambiguated, counts
