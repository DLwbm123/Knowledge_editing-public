"""Deterministic EqKey-family contracts for the LiveEdit-Med fast pilot.

This module is deliberately model-free.  It operates only on frozen cache
metadata and therefore cannot open the blind set with an edited checkpoint or
mutate any cached tensor.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL = "LIVEEDIT_MED_EQKEY_CLEAN_FAST_CONFIRMATION_V1"
POSITIVE_ROLES = ("native", "textual", "visual", "paired")
ROLE_PRIORITY = {name: index for index, name in enumerate(POSITIVE_ROLES)}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_target(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        value = str(value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def _canonical_view(occurrences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (ROLE_PRIORITY[str(row["role"])], str(row["selection_hash"]),
                str(row["record_id"]), str(row["source_split"]), int(row["source_ordinal"]))

    selected = min(occurrences, key=key)
    result = dict(selected)
    result["provenance"] = [
        {name: row[name] for name in ("source_split", "source_ordinal", "record_id", "role",
                                      "selection_hash", "cache_file_path", "cache_file_sha256")}
        for row in sorted(occurrences, key=key)
    ]
    return result


def build_families(ledger: Sequence[Mapping[str, Any]], *, blind_record_ids: set[str],
                   blind_eqkeys: set[str], record953_eqkeys: set[str]) -> list[dict[str, Any]]:
    edit_ids = sorted({str(row["record_id"]) for row in ledger})
    union = UnionFind(edit_ids)
    by_eqkey: dict[str, list[str]] = defaultdict(list)
    for row in ledger:
        by_eqkey[str(row["eqkey"])].append(str(row["record_id"]))
    for record_ids in by_eqkey.values():
        first = record_ids[0]
        for record_id in record_ids[1:]:
            union.union(first, record_id)

    rows_by_record: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ledger:
        rows_by_record[str(row["record_id"])].append(row)
    component_members: dict[str, set[str]] = defaultdict(set)
    for record_id in edit_ids:
        component_members[union.find(record_id)].add(record_id)

    families = []
    for members in component_members.values():
        rows = [row for record_id in members for row in rows_by_record[record_id]]
        eqkeys = sorted({str(row["eqkey"]) for row in rows})
        family_id = canonical_hash({"member_edit_ids": sorted(members), "positive_eqkeys": eqkeys})
        by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_key[str(row["eqkey"])].append(row)
        canonical_views = [_canonical_view(by_key[key]) for key in sorted(by_key)]
        record_order = sorted(
            members,
            key=lambda record_id: (min(str(row["selection_hash"])
                                       for row in rows_by_record[record_id]), record_id),
        )
        canonical_record_id = record_order[0]
        canonical_native = min(
            (row for row in rows_by_record[canonical_record_id] if row["role"] == "native"),
            key=lambda row: (str(row["selection_hash"]), int(row["source_ordinal"])),
        )
        targets = sorted({normalize_target(row["target"]) for row in rows})
        splits = sorted({str(row["source_split"]) for row in rows})
        role_counts = Counter(str(row["role"]) for row in canonical_views)
        touches_record953 = "953" in members or bool(set(eqkeys) & record953_eqkeys)
        touches_blind = bool(members & blind_record_ids or set(eqkeys) & blind_eqkeys)
        family = {
            "protocol": PROTOCOL,
            "family_id": family_id,
            "member_edit_ids": sorted(members),
            "member_original_splits": splits,
            "positive_eqkeys": eqkeys,
            "roles_per_eqkey": {
                key: sorted({str(row["role"]) for row in by_key[key]}) for key in sorted(by_key)
            },
            "normalized_target_set": targets,
            "source_rows": sorted({
                (str(row["source_split"]), int(row["source_ordinal"]), str(row["record_id"]))
                for row in rows
            }),
            "image_hashes": sorted({str(row["raw_image_sha256"]) for row in rows}),
            "question_hashes": sorted({str(row["question_sha256"]) for row in rows}),
            "component_size": len(members),
            "touches_original_train": "train" in splits,
            "touches_original_validation": "validation" in splits,
            "touches_original_heldout": "heldout" in splits,
            "touches_record953": touches_record953,
            "touches_sealed_blind": touches_blind,
            "target_conflict": len(targets) != 1,
            "generator_exposure": "GENERATOR_SEEN_FAMILY" if "train" in splits else "GENERATOR_UNSEEN_FAMILY",
            "canonical_record_id": canonical_record_id,
            "canonical_native_eqkey": str(canonical_native["eqkey"]),
            "canonical_target": str(canonical_native["target"]),
            "canonical_target_normalized": normalize_target(canonical_native["target"]),
            "canonical_views": canonical_views,
            "canonical_role_counts": {role: int(role_counts[role]) for role in POSITIVE_ROLES},
        }
        family["allocation_hash"] = hashlib.sha256(
            (family_id + family["canonical_native_eqkey"] + family["canonical_target_normalized"]).encode()
        ).hexdigest()
        families.append(family)
    return sorted(families, key=lambda row: row["family_id"])


def split_role_counts(families: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for family in families:
        counts.update({role: int(value) for role, value in family["canonical_role_counts"].items()})
    return {role: int(counts[role]) for role in POSITIVE_ROLES}


def _meets(families: Sequence[Mapping[str, Any]], family_count: int,
           minimums: Mapping[str, int]) -> bool:
    counts = split_role_counts(families)
    return len(families) == family_count and all(counts[role] >= minimum for role, minimum in minimums.items())


def allocate_purged(families: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = sorted((family for family in families
                       if not family["touches_original_train"]
                       and not family["target_conflict"]
                       and not family["touches_record953"]
                       and not family["touches_sealed_blind"]),
                      key=lambda row: (row["allocation_hash"], row["family_id"]))
    full_val, full_held = eligible[:32], eligible[32:64]
    full_min = {"native": 32, "textual": 16, "visual": 16, "paired": 16}
    low_val, low_held = eligible[:16], eligible[16:32]
    low_min = {"native": 16, "textual": 8, "visual": 8, "paired": 8}
    if (_meets(full_val, 32, full_min) and _meets(full_held, 32, full_min)):
        label, validation, heldout = "EQKEY_PURGED_FULL_FAST_LANE", full_val, full_held
    elif (_meets(low_val, 16, low_min) and _meets(low_held, 16, low_min)):
        label, validation, heldout = "EQKEY_PURGED_LOW_SAMPLE_FAST_LANE", low_val, low_held
    else:
        label, validation, heldout = "EQKEY_PURGED_SPLIT_INSUFFICIENT_FOR_FAST_CONFIRMATION", [], []
    return {
        "protocol": PROTOCOL,
        "label": label,
        "eligible_family_count": len(eligible),
        "eligible_family_ids": [row["family_id"] for row in eligible],
        "validation": list(validation),
        "heldout": list(heldout),
        "validation_role_counts": split_role_counts(validation),
        "heldout_role_counts": split_role_counts(heldout),
    }


def future_family_split(families: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = sorted((family for family in families if not family["target_conflict"]
                       and not family["touches_record953"] and not family["touches_sealed_blind"]),
                      key=lambda row: (row["allocation_hash"], row["family_id"]))
    total = len(eligible)
    train_end = int(total * .8)
    validation_end = train_end + int(total * .1)
    return {
        "protocol": PROTOCOL,
        "status": "PROPOSAL_ONLY__DO_NOT_TRAIN_IN_FAST_CONFIRMATION",
        "allocation": {
            "train": [row["family_id"] for row in eligible[:train_end]],
            "validation": [row["family_id"] for row in eligible[train_end:validation_end]],
            "heldout": [row["family_id"] for row in eligible[validation_end:]],
        },
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def exact_mcnemar(s0: Sequence[bool], forced: Sequence[bool]) -> dict[str, Any]:
    if len(s0) != len(forced) or not s0:
        raise ValueError("paired non-empty outcomes required")
    improved = sum(not left and right for left, right in zip(s0, forced))
    regressed = sum(left and not right for left, right in zip(s0, forced))
    discordant = improved + regressed
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(0, min(improved, regressed) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** discordant))
    return {"s0_wrong_forced_right": improved, "s0_right_forced_wrong": regressed,
            "discordant": discordant, "p_value_two_sided_exact": p_value}
