from types import SimpleNamespace

import torch

from methods.liveedit_med.trace_parity import (
    REQUIRED_TRACE_BOUNDARIES,
    compare_discrete,
    compare_tensor,
    compare_tensor_mapping,
    install_source_era_llava_merge,
    state_dict_sha256,
    summarize_trace,
)


def test_trace_summary_requires_every_named_boundary():
    row = compare_tensor("hidden", torch.ones(2), torch.ones(2), atol=0, rtol=0)
    assert summarize_trace([row], required_names=["hidden"])["all_passed"]
    assert not summarize_trace([row], required_names=["hidden", "logits"])["all_passed"]


def test_stage_a_protocol_has_exactly_27_unique_boundaries():
    assert len(REQUIRED_TRACE_BOUNDARIES) == 27
    assert len(set(REQUIRED_TRACE_BOUNDARIES)) == 27


def test_tensor_mapping_requires_identical_keys_and_values():
    passed = compare_tensor_mapping(
        "processor_outputs", {"ids": torch.tensor([1])}, {"ids": torch.tensor([1])}, atol=0, rtol=0
    )
    assert passed["passed"]
    failed = compare_tensor_mapping(
        "processor_outputs", {"ids": torch.tensor([1])}, {"mask": torch.tensor([1])}, atol=0, rtol=0
    )
    assert not failed["passed"]


def test_discrete_and_state_hash_are_exact_and_order_stable():
    assert compare_discrete("ids", torch.tensor([1, 2]), [1, 2])["passed"]
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    second = {"a": first["a"].clone(), "b": first["b"].clone()}
    assert state_dict_sha256(first) == state_dict_sha256(second)


def test_source_era_llava_merge_compatibility_boundary():
    model = SimpleNamespace(config=SimpleNamespace(image_token_index=99, pad_token_id=0, ignore_index=-100))
    assert install_source_era_llava_merge(model)
    image = torch.tensor([[[10.0, 11.0], [12.0, 13.0]]])
    embeds = torch.tensor([[[1.0, 1.0], [9.0, 9.0], [2.0, 2.0]]])
    ids = torch.tensor([[1, 99, 2]])
    mask = torch.ones_like(ids)
    merged, merged_mask, labels, positions = model._merge_input_ids_with_image_features(
        image, embeds, ids, mask, None
    )
    assert labels is None
    assert merged.tolist() == [[[1.0, 1.0], [10.0, 11.0], [12.0, 13.0], [2.0, 2.0]]]
    assert merged_mask.tolist() == [[1, 1, 1, 1]]
    assert positions.tolist() == [[0, 1, 2, 3]]
    assert not install_source_era_llava_merge(model)
