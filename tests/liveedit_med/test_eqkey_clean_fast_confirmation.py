from __future__ import annotations

from methods.liveedit_med.eqkey_clean_fast import (allocate_purged, build_families, canonical_hash,
    exact_mcnemar, wilson_interval)
from scripts.liveedit_med.cache_eqkey_clean_router_r1 import clean_train
from scripts.liveedit_med.select_eqkey_clean_router_r1 import relative_failures
from scripts.liveedit_med.train_eqkey_clean_router_r1 import role_view, stable_repository


def row(record_id, split, eqkey, role="native", target="A", ordinal=0):
    return {
        "record_id": str(record_id), "source_split": split, "source_ordinal": ordinal,
        "selection_hash": canonical_hash([record_id]), "eqkey": eqkey, "role": role,
        "target": target, "image_path": "/x", "raw_image_sha256": "i" + eqkey,
        "question_sha256": "q" + eqkey, "cache_file_path": "/cache",
        "cache_file_sha256": "c" + str(record_id),
    }


def test_transitive_components_and_seen_propagation():
    rows = [row(1, "train", "a"), row(2, "validation", "a", "visual"),
            row(2, "validation", "b"), row(3, "heldout", "b", "visual")]
    families = build_families(rows, blind_record_ids=set(), blind_eqkeys=set(), record953_eqkeys=set())
    assert len(families) == 1
    assert families[0]["member_edit_ids"] == ["1", "2", "3"]
    assert families[0]["generator_exposure"] == "GENERATOR_SEEN_FAMILY"


def test_target_conflict_and_blind_isolation():
    rows = [row(1, "validation", "a", target="Alpha"), row(2, "heldout", "a", target="Beta")]
    family = build_families(rows, blind_record_ids={"2"}, blind_eqkeys=set(), record953_eqkeys=set())[0]
    assert family["target_conflict"] is True
    assert family["touches_sealed_blind"] is True


def test_within_family_eqkey_is_canonicalized_once():
    rows = [row(1, "validation", "a", "visual"), row(2, "validation", "a", "native")]
    family = build_families(rows, blind_record_ids=set(), blind_eqkeys=set(), record953_eqkeys=set())[0]
    assert len(family["canonical_views"]) == 1
    assert family["canonical_views"][0]["role"] == "native"
    assert len(family["canonical_views"][0]["provenance"]) == 2


def test_full_lane_is_deterministic_and_family_disjoint():
    rows = []
    for index in range(64):
        for role in ("native", "textual", "visual", "paired"):
            rows.append(row(index, "validation" if index < 32 else "heldout",
                            f"{index}-{role}", role, ordinal=index))
    families = build_families(rows, blind_record_ids=set(), blind_eqkeys=set(), record953_eqkeys=set())
    first, second = allocate_purged(families), allocate_purged(list(reversed(families)))
    assert first["label"] == "EQKEY_PURGED_FULL_FAST_LANE"
    assert [x["family_id"] for x in first["validation"]] == [x["family_id"] for x in second["validation"]]
    assert set(x["family_id"] for x in first["validation"]).isdisjoint(
        x["family_id"] for x in first["heldout"])


def test_seen_families_never_enter_purged_eval():
    rows = []
    for index in range(64):
        split = "train" if index == 0 else "validation"
        for role in ("native", "textual", "visual", "paired"):
            rows.append(row(index, split, f"{index}-{role}", role, ordinal=index))
    families = build_families(rows, blind_record_ids=set(), blind_eqkeys=set(), record953_eqkeys=set())
    allocation = allocate_purged(families)
    selected = allocation["validation"] + allocation["heldout"]
    assert all(not family["touches_original_train"] for family in selected)


def test_exact_mcnemar_and_wilson():
    result = exact_mcnemar([False] * 8, [True] * 8)
    assert result["s0_wrong_forced_right"] == 8
    assert result["s0_right_forced_wrong"] == 0
    assert result["p_value_two_sided_exact"] == 2 / 256
    low, high = wilson_interval(8, 8)
    assert 0.67 < low < high == 1.0


def test_eqkey_clean_router_r1_family_contracts():
    families = []
    for index in range(40):
        families.append({"family_id": str(index), "allocation_hash": canonical_hash([index]),
            "touches_original_train": True, "target_conflict": False,
            "touches_record953": False, "touches_sealed_blind": False})
    selected = clean_train(families)
    ids = [item["family_id"] for item in selected]
    repository = stable_repository(ids[0], 32, ids)
    assert repository[0] == ids[0]
    assert len(repository) == len(set(repository)) == 32
    family = {"family_id": "x", "canonical_views": [
        {"role": "native", "selection_hash": "b", "eqkey": "2"},
        {"role": "native", "selection_hash": "a", "eqkey": "1"}]}
    assert role_view(family, "native")["eqkey"] == "1"


def test_selector_accepts_singleton_batched_final_weights_without_semantic_drift():
    def result(weights):
        route = {"candidate_ids": ["distractor", "target"], "final_weights": weights}
        return {"rows": [{"family_id": "target", "repositories": {"32": {"routed": {
            "native": {"route": route},
        }}}}]}

    assert relative_failures(result([0.25, 0.75])) == 0
    assert relative_failures(result([[0.25, 0.75]])) == 0
    assert relative_failures(result([[0.75, 0.25]])) == 1
