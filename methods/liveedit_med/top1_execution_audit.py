"""Model-free R2A validation, selection, and reproducibility contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


STEPS = (80, 160, 240, 320, 400, 480, 560, 640)
ROLES = ("native", "textual", "visual", "paired")
EFFECTIVENESS_FLOORS = {"native": 18, "textual": 13, "visual": 14, "paired": 14}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def validation_eligible(row: Mapping[str, Any]) -> bool:
    routed = row["repository_sizes"]["32"]["routed"]
    safety = row["safety"]
    return (
        all(int(routed[role]) >= floor for role, floor in EFFECTIVENESS_FLOORS.items())
        and int(row["repository_sizes"]["32"]["locality_exact_preservation"]) == 64
        and int(safety["hard_negative_exact_s0"]) >= 152
        and int(row["repository_sizes"]["32"]["target_contaminations"]) == 0
        and int(row["repository_sizes"]["32"]["clinical_canonical_failures"]) == 0
        and int(safety["target_contaminations"]) == 0
        and int(safety["clinical_canonical_failures"]) == 0
        and int(row.get("generation_parity_failures", 0)) == 0
    )


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    routed = row["repository_sizes"]["32"]["routed"]
    safety = row["safety"]
    return (
        int(safety["hard_negative_exact_s0"]),
        int(routed["visual"]) + int(routed["paired"]),
        int(routed["native"]) + int(routed["textual"]),
        int(row["repository_sizes"]["32"]["target_expert_top1_count"]),
        -float(safety["mean_selected_final_weight"]),
        -float(safety["mean_selected_residual_norm"]),
        -int(row["step"]),
    )


def select_checkpoint(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if tuple(sorted(int(row["step"]) for row in rows)) != STEPS:
        raise RuntimeError("R2A_TOP1_INCOMPLETE_VALIDATION_SET")
    eligible = [row for row in rows if validation_eligible(row)]
    return max(eligible, key=selection_key) if eligible else None


def capture_ratio(r1: int, oracle_o2: int, r2a: int) -> dict[str, Any]:
    oracle_recovery = int(oracle_o2) - int(r1)
    r2a_recovery = int(r2a) - int(r1)
    return {
        "r1_success": int(r1),
        "oracle_o2_success": int(oracle_o2),
        "r2a_success": int(r2a),
        "oracle_o2_recovery": oracle_recovery,
        "r2a_recovery": r2a_recovery,
        "oracle_capture_ratio": None if oracle_recovery == 0 else r2a_recovery / oracle_recovery,
        "zero_oracle_recovery": oracle_recovery == 0,
    }

