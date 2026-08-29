#!/usr/bin/env python3
"""Freeze Router-R2 provenance, anchors, policy, and calibration split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.router_r2 import PROTOCOL, ROLES
from scripts.liveedit_med.eqkey_fixed_expert_utils import sha256_file


EXPECTED = {
    "canonical_bank_hash": "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db",
    "frozen_module_hash": "d1d5ce232ad2aeb7a29c4c5586a6af5dcdee064321cfc94c3393d576b1bc2249",
    "base_model_hash": "d8b7032a563e32f22fd51eb65d92bbb0177c913d19c5c1e6ce6ad73d0e5ca75d",
}
CATEGORIES = ("same_image_different_question", "same_question_different_image",
              "visual_nearest", "text_nearest", "joint_near_miss")
SPLIT_SALT = "LIVEEDIT_MED_ROUTER_R2_CALIBRATION_SPLIT_V1_SEED_42"


def write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def stable_key(family_id: str) -> tuple[str, str]:
    return hashlib.sha256(f"{SPLIT_SALT}:{family_id}".encode()).hexdigest(), family_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--oracle-run", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--r1-closure-source-commit", required=True)
    parser.add_argument("--r2-source-branch", required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError(args.out_root)
    args.out_root.mkdir(parents=True)

    anchor_path = args.oracle_run / "anchor_and_immutability_audit.json"
    anchor = json.loads(anchor_path.read_text())
    actual = {key: anchor[key] for key in EXPECTED}
    if actual != EXPECTED or len(anchor["checkpoints"]) != 8 \
            or not all(row["tensor_hashes_verified"] for row in anchor["checkpoints"]):
        raise RuntimeError("ROUTER_R2_STOP__FROZEN_HASH_MISMATCH")
    step640 = args.formal_run / "training/checkpoint_0640"
    checkpoint_manifest = json.loads((step640 / "manifest.json").read_text())
    for key, value in EXPECTED.items():
        if checkpoint_manifest.get(key) != value:
            raise RuntimeError(f"ROUTER_R2_STOP__FROZEN_HASH_MISMATCH:{key}")

    selection_path = args.formal_run / "evaluation/selection/checkpoint_selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("selected_step") is not None \
            or selection.get("label") != "ROUTER_R1_NO_ELIGIBLE_CLEAN_VALIDATION_CHECKPOINT":
        raise RuntimeError("ROUTER_R2_STOP__R1_SELECTION_STATE_MISMATCH")
    policy_path = ROOT / "scripts/liveedit_med/select_eqkey_clean_router_r1.py"
    policy_source = policy_path.read_text()
    required_policy_fragments = (
        'math.ceil(.90 * forced["native"])',
        'math.ceil(.75 * forced["textual"])',
        'math.ceil(.75 * forced["visual"])',
        'math.ceil(.75 * forced["paired"])',
        'metrics["target_contamination"] == 0',
        'metrics["clinical_canonical_failures"] == 0',
    )
    if not all(fragment in policy_source for fragment in required_policy_fragments):
        raise RuntimeError("ROUTER_R2_STOP__FROZEN_SAFETY_POLICY_NOT_FOUND")

    train_path = args.parent_run / "router_r1/clean_train_family_manifest.json"
    hard_path = args.parent_run / "router_r1/hard_negative_cache_manifest.json"
    train = json.loads(train_path.read_text())
    hard = json.loads(hard_path.read_text())
    families = train["families"]
    family_ids = [row["family_id"] for row in families]
    hard_by_id = {row["family_id"]: row for row in hard["records"]}
    if len(family_ids) != 467 or len(set(family_ids)) != 467 or set(hard_by_id) != set(family_ids):
        raise RuntimeError("ROUTER_R2_CALIBRATION_SOURCE_SET_MISMATCH")
    # ``touches_original_validation/heldout`` describe historical source-graph
    # provenance, not membership in either sealed purged evaluation manifest.
    # Calibration eligibility is the generator-seen train-family contract.
    forbidden = [row["family_id"] for row in families if (
        row.get("generator_exposure") != "GENERATOR_SEEN_FAMILY"
        or row.get("touches_record953") or row.get("touches_sealed_blind")
        or not row.get("touches_original_train") or row.get("target_conflict"))]
    if forbidden:
        raise RuntimeError("ROUTER_R2_CALIBRATION_FORBIDDEN_FAMILY")
    for row in families:
        if not set(ROLES).issubset({view["role"] for view in row["canonical_views"]}):
            raise RuntimeError(f"ROUTER_R2_CALIBRATION_ROLE_MISSING:{row['family_id']}")
        if set(hard_by_id[row["family_id"]]["negative_eqkeys"]) != set(CATEGORIES):
            raise RuntimeError(f"ROUTER_R2_CALIBRATION_NEGATIVE_SUBTYPE_MISSING:{row['family_id']}")

    ordered = sorted(family_ids, key=stable_key)
    fit_count = int(math.floor(len(ordered) * 0.8 + 0.5))
    fit_ids, lock_ids = ordered[:fit_count], ordered[fit_count:]
    if set(fit_ids) & set(lock_ids) or set(fit_ids) | set(lock_ids) != set(family_ids):
        raise RuntimeError("ROUTER_R2_CALIBRATION_SPLIT_INVALID")
    family_by_id = {row["family_id"]: row for row in families}

    def split_rows(ids):
        return [{"family_id": family_id,
                 "canonical_record_id": family_by_id[family_id]["canonical_record_id"],
                 "allocation_hash": family_by_id[family_id]["allocation_hash"],
                 "stable_split_hash": stable_key(family_id)[0]}
                for family_id in ids]

    split = {
        "protocol": PROTOCOL, "split_protocol": SPLIT_SALT,
        "created_before_r2_score_inspection": True,
        "source_family_count": len(family_ids),
        "calibration_fit": split_rows(fit_ids),
        "calibration_lock": split_rows(lock_ids),
        "calibration_fit_family_count": len(fit_ids),
        "calibration_lock_family_count": len(lock_ids),
        "positive_role_counts": {
            "calibration_fit": {role: len(fit_ids) for role in ROLES},
            "calibration_lock": {role: len(lock_ids) for role in ROLES},
        },
        "negative_subtype_counts": {
            "calibration_fit": {category: len(fit_ids) for category in CATEGORIES},
            "calibration_lock": {category: len(lock_ids) for category in CATEGORIES},
        },
        "record953_excluded": True, "sealed_blind_excluded": True,
        "validation_excluded": True, "heldout_excluded": True,
    }
    split_path = args.out_root / "CALIBRATION_SPLIT_MANIFEST.json"
    write_json(split_path, split)

    input_files = [anchor_path, selection_path,
                   args.oracle_run / "r1_safety_closure/ROUTER_R1_FINAL_SELECTION.json",
                   args.oracle_run / "r1_safety_closure/safety_aggregate.json",
                   args.oracle_run / "oracle_diagnosis_summary.json",
                   args.oracle_run / "run_manifest_final.json",
                   train_path, hard_path, step640 / "manifest.json", step640 / "model.safetensors"]
    hash_audit = {
        "protocol": PROTOCOL, **actual, "expected_hashes": EXPECTED,
        "anchor_match": actual == EXPECTED,
        "checkpoint_tensor_hashes_verified": True,
        "step640_checkpoint_manifest_sha256": sha256_file(step640 / "manifest.json"),
        "step640_checkpoint_model_sha256": sha256_file(step640 / "model.safetensors"),
        "input_file_hashes": {str(path): sha256_file(path) for path in input_files},
        "hard_stop_on_mismatch": True,
    }
    write_json(args.out_root / "FROZEN_INPUT_HASH_AUDIT.json", hash_audit)
    write_json(args.out_root / "SOURCE_PROVENANCE.json", {
        "protocol": PROTOCOL,
        "r1_closure_source_commit_sha": args.r1_closure_source_commit,
        "r2_source_branch": args.r2_source_branch,
        "r2_source_commit_sha": "PENDING_R2_SOURCE_ONLY_COMMIT",
        "push_performed": False,
        "source_only": True,
        "forbidden_artifact_staged": False,
    })
    write_json(args.out_root / "DATA_ACCESS_AUDIT.json", {
        "protocol": PROTOCOL,
        "calibration_source": str(train_path),
        "hard_negative_source": str(hard_path),
        "family_graph_flags_used_for_overlap_audit": True,
        "calibration_fit_lock_overlap": 0,
        "historical_source_graph_counts": {
            "touches_original_validation": sum(bool(row.get("touches_original_validation"))
                                                for row in families),
            "touches_original_heldout": sum(bool(row.get("touches_original_heldout"))
                                             for row in families),
            "generator_seen_family": sum(
                row.get("generator_exposure") == "GENERATOR_SEEN_FAMILY"
                for row in families),
        },
        "purged_validation_manifest_loaded": False,
        "purged_heldout_manifest_loaded": False,
        "validation_loaded": False, "heldout_loaded": False,
        "record953_loaded": False, "sealed_blind_loaded": False,
        "stage2_loaded": False,
        "all_calibration_families_generator_seen_train": True,
    })
    write_json(args.out_root / "ROUTER_R2_PREREGISTRATION.json", {
        "protocol": PROTOCOL,
        "primary_candidate": "R2_E0_SELECTIVE_TOP1_DUAL_MARGIN",
        "diagnostic_candidate": "R2_E0_TOP1_ORIGINAL_COEFFICIENT_NO_REJECTION",
        "source_router": "Router-R1 step 640 only",
        "main_execution_mode": "TOP1_ORIGINAL_COEFFICIENT",
        "post_pruning_renormalization": False,
        "hard_gate_changed": False,
        "threshold_fit_split": "calibration_fit",
        "threshold_lock_split": "calibration_lock",
        "effectiveness_policy": {"native": "ceil(0.90*forced_on)",
                                 "textual": "ceil(0.75*forced_on)",
                                 "visual": "ceil(0.75*forced_on)",
                                 "paired": "ceil(0.75*forced_on)"},
        "safety_policy": {"clinical_canonical_failures": 0, "target_contamination": 0},
        "policy_source_path": str(policy_path), "policy_source_sha256": sha256_file(policy_path),
        "validation_may_be_loaded_only_after_candidate_freeze": True,
        "heldout_permitted": False, "record953_permitted": False,
        "sealed_blind_permitted": False, "stage2_permitted": False,
    })
    print(json.dumps({"status": "ROUTER_R2_PREREGISTRATION_COMPLETE",
                      "out_root": str(args.out_root), "fit": len(fit_ids),
                      "lock": len(lock_ids), "r1_source_commit": args.r1_closure_source_commit},
                     sort_keys=True))


if __name__ == "__main__":
    main()
