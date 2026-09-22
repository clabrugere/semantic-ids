import pytest
import torch

from semantic_ids.semantic_ids import SemanticIds, assign_disambiguation, remap_disambiguation_table


@pytest.fixture
def tmp_pt(tmp_path):
    return tmp_path / "catalog.pt"


def test_save_writes_a_plain_dict_with_exactly_three_keys(tmp_pt):
    """This is the guard for the module's stated invariant: it must fail if `save` is ever changed
    to `torch.save(self)`, since torch's weights-only default rejects an arbitrary pickled object."""
    catalog = SemanticIds(semantic_ids=torch.tensor([[0, 0], [1, 0]]), item_ids=torch.tensor([0, 1]), num_codes=4)
    catalog.save(tmp_pt)

    raw = torch.load(tmp_pt)
    assert isinstance(raw, dict)
    assert sorted(raw) == ["item_ids", "num_codes", "semantic_ids"]


def test_save_then_load_round_trips(tmp_pt):
    catalog = SemanticIds(
        semantic_ids=torch.tensor([[0, 0], [0, 1], [2, 3]]), item_ids=torch.tensor([5, 6, 7]), num_codes=8
    )
    catalog.save(tmp_pt)
    loaded = SemanticIds.load(tmp_pt)

    assert torch.equal(loaded.semantic_ids, catalog.semantic_ids)
    assert torch.equal(loaded.item_ids, catalog.item_ids)
    assert loaded.num_codes == catalog.num_codes


def test_rejects_non_2d_semantic_ids():
    with pytest.raises(ValueError):
        SemanticIds(semantic_ids=torch.tensor([0, 1, 2]), item_ids=torch.tensor([0, 1, 2]), num_codes=4)


def test_rejects_mismatched_item_id_count():
    with pytest.raises(ValueError):
        SemanticIds(semantic_ids=torch.tensor([[0, 0], [0, 1]]), item_ids=torch.tensor([0]), num_codes=4)


# assign_disambiguation


def test_assign_disambiguation_worked_example():
    prefixes = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 0]])  # two rows share [0,0]... three do
    codes, counts = assign_disambiguation(prefixes)
    assert codes.tolist() == [0, 1, 0, 2]  # ranked in input order within the [0,0] group
    assert sorted(counts.tolist()) == [1, 3]


def test_assign_disambiguation_orders_by_sort_key_ascending():
    """With a sort_key, ordinal 0 goes to the smallest key in each group, independent of row order."""
    prefixes = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 0]])  # rows 0,1,3 collide; row 2 is alone
    sort_key = torch.tensor([5.0, 1.0, 0.0, 3.0])  # within the colliding group: row1 < row3 < row0

    codes, counts = assign_disambiguation(prefixes, sort_key)

    assert codes.tolist() == [2, 0, 0, 1]  # row1 -> 0, row3 -> 1, row0 -> 2; row2's own group -> 0
    assert sorted(counts.tolist()) == [1, 3]


def test_assign_disambiguation_sort_key_ties_fall_back_to_input_order():
    prefixes = torch.tensor([[0, 0], [0, 0], [0, 0]])
    sort_key = torch.tensor([1.0, 1.0, 0.0])  # rows 0 and 1 tie; row 2 is smallest

    codes, _ = assign_disambiguation(prefixes, sort_key)

    assert codes.tolist() == [1, 2, 0]  # row2 first (smallest), then the tie broken by input order


def test_assign_disambiguation_is_not_per_row_permutation_invariant_but_the_id_set_is():
    """Codes are handed out within a group in *input* row order, so permuting the input changes
    which row gets which code. The resulting set of (prefix, code) pairs, and the per-group counts,
    are the same either way. This is the property that matters: item_ids stay aligned with whichever
    row produced which code."""
    torch.manual_seed(0)
    prefixes = torch.randint(0, 3, (12, 2))
    perm = torch.randperm(12)

    codes_a, counts_a = assign_disambiguation(prefixes)
    codes_b, counts_b = assign_disambiguation(prefixes[perm])

    assert not torch.equal(codes_b, codes_a[perm])  # NOT per-row invariant

    pairs_a = {tuple(row) for row in torch.cat([prefixes, codes_a[:, None]], dim=1).tolist()}
    pairs_b = {tuple(row) for row in torch.cat([prefixes[perm], codes_b[:, None]], dim=1).tolist()}
    assert pairs_a == pairs_b  # but the resulting ID set is invariant
    assert torch.equal(counts_a, counts_b)  # and so are the group counts


# build semantic ids from codes


def test_build_semantic_ids_produces_unique_rows():
    torch.manual_seed(0)
    prefixes = torch.randint(0, 3, (20, 2))
    catalog = SemanticIds.from_codes(prefixes, num_codes=8)
    assert torch.unique(catalog.semantic_ids, dim=0).size(0) == catalog.num_items


def test_build_semantic_ids_defaults_item_ids_to_arange():
    prefixes = torch.tensor([[0, 0], [1, 1]])
    catalog = SemanticIds.from_codes(prefixes, num_codes=4)
    assert torch.equal(catalog.item_ids, torch.arange(2))


def test_build_semantic_ids_honours_explicit_item_ids():
    prefixes = torch.tensor([[0, 0], [1, 1]])
    item_ids = torch.tensor([100, 101])
    catalog = SemanticIds.from_codes(prefixes, num_codes=4, item_ids=item_ids)
    assert torch.equal(catalog.item_ids, item_ids)


def test_collision_groups_may_exceed_num_codes():
    """How many items collide on one prefix is a property of the catalog, not the codebook, so c_K's
    vocabulary is floored at num_codes but not capped there. A group larger than num_codes widens it
    further rather than raising."""
    prefixes = torch.zeros(5, 2, dtype=torch.long)  # 5 identical prefixes against num_codes=4
    catalog = SemanticIds.from_codes(prefixes, num_codes=4)

    assert catalog.semantic_ids[:, -1].tolist() == [0, 1, 2, 3, 4]
    assert catalog.num_disambiguation_codes == 5  # > num_codes, and that is not an error
    assert catalog.num_codes == 4  # the scored vocabulary is untouched


def test_num_disambiguation_codes_is_one_when_nothing_collides():
    """Every group here has exactly one member, so the largest group's size is 1. This property
    answers that honestly, with no floor. SemanticTrie's own same-named property floors this at
    num_codes for a trie-internal encoding reason, but that floor does not belong on the catalog,
    which only describes the data."""
    catalog = SemanticIds.from_codes(torch.tensor([[0, 0], [0, 1], [1, 0]]), num_codes=4)
    assert catalog.num_disambiguation_codes == 1


def test_empty_catalog_is_rejected():
    """num_disambiguation_codes has no answer on M=0, and the trie it built could serve no beam."""
    with pytest.raises(ValueError, match="at least one item"):
        SemanticIds(
            semantic_ids=torch.zeros(0, 4, dtype=torch.long),
            item_ids=torch.zeros(0, dtype=torch.long),
            num_codes=4,
        )


def test_zero_levels_is_rejected_before_it_can_reach_an_unanswerable_property():
    """num_disambiguation_codes reads semantic_ids[:, -1], which has no answer on K=0 either."""
    with pytest.raises(ValueError):
        catalog = SemanticIds(
            semantic_ids=torch.zeros(3, 0, dtype=torch.long), item_ids=torch.zeros(3, dtype=torch.long), num_codes=4
        )
        assert catalog.num_disambiguation_codes >= 0


#  device placement


def test_every_output_follows_the_input_device_not_the_default_device():
    """assign_disambiguation used to allocate its arange/empty on the *default* device, so a non-CPU
    prefixes tensor raised on the subtraction; item_ids defaulted to a separate arange that
    __post_init__ never device-checked against semantic_ids.

    "meta" as the default device reproduces that RuntimeError with no GPU, so this runs on CPU CI
    rather than skipping the way a cuda-parametrized fixture would.
    """
    prefixes = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 0]])
    with torch.device("meta"):
        codes, _ = assign_disambiguation(prefixes)
        catalog = SemanticIds.from_codes(prefixes, num_codes=4)

    assert codes.device == prefixes.device
    assert codes.tolist() == [0, 1, 0, 2]
    assert catalog.semantic_ids.device == prefixes.device
    assert catalog.item_ids.device == catalog.semantic_ids.device


#  remap_disambiguation_table


def test_remap_copies_the_shared_prefix_exactly():
    old = torch.randn(3, 8)
    new = torch.randn(3, 8)
    merged = remap_disambiguation_table(old, new)
    assert torch.equal(merged, old)


def test_remap_fills_a_grown_tail_with_the_mean_of_the_copied_prefix():
    old = torch.randn(3, 8)
    new = torch.randn(6, 8)
    merged = remap_disambiguation_table(old, new)
    assert torch.equal(merged[:3], old)
    assert torch.allclose(merged[3:], old.mean(dim=0).expand(3, 8))


def test_remap_truncates_without_error_when_the_table_shrinks():
    old = torch.randn(6, 8)
    new = torch.randn(3, 8)
    merged = remap_disambiguation_table(old, new)
    assert torch.equal(merged, old[:3])


def test_remap_does_not_mutate_either_argument():
    old, new = torch.randn(3, 8), torch.randn(6, 8)
    old_ref, new_ref = old.clone(), new.clone()
    merged = remap_disambiguation_table(old, new)
    assert torch.equal(old, old_ref)
    assert torch.equal(new, new_ref)
    assert merged.data_ptr() != old.data_ptr()
    assert merged.data_ptr() != new.data_ptr()


def test_remap_rejects_a_dimension_mismatch():
    with pytest.raises(ValueError):
        remap_disambiguation_table(torch.randn(3, 8), torch.randn(6, 16))
