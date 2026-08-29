#!/usr/bin/env python3
"""Merge frozen E0 calibration shards, fit once, and lock once."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.router_r2 import PROTOCOL, ROLES, apply_thresholds, select_dual_thresholds
from scripts.liveedit_med.eqkey_fixed_expert_utils import sha256_file


CATEGORIES = ("same_image_different_question", "same_question_different_image",
              "visual_nearest", "text_nearest", "joint_near_miss")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_rows(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_rows, rows = [], []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                family = json.loads(line); family_rows.append(family); rows.extend(family["rows"])
    return family_rows, rows


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p25", "median", "p75", "max", "mean")}
    ordered = sorted(float(value) for value in values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
    return {"min": ordered[0], "p25": pick(.25), "median": statistics.median(ordered),
            "p75": pick(.75), "max": ordered[-1], "mean": statistics.fmean(ordered)}


def diagnostics(rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> dict[str, Any]:
    positives = [row for row in rows if row["kind"] == "positive"]
    negatives = [row for row in rows if row["kind"] == "negative"]
    tau_abs = float(metrics["thresholds"]["tau_abs"])
    tau_gap = float(metrics["thresholds"]["tau_gap"])
    accepted = lambda row: (int(row["candidate_count"]) > 0 and
                            float(row["s1"]) >= tau_abs and float(row["gap"]) >= tau_gap)
    by_role = {}
    for role in ROLES:
        group = [row for row in positives if row["role"] == role]
        hard = sum(bool(row["target_hard_recalled"]) for row in group)
        top1 = sum(bool(row["target_selected_top1"]) for row in group)
        top1_given_hard = sum(bool(row["target_selected_top1"]) for row in group
                              if row["target_hard_recalled"])
        by_role[role] = {
            "count": len(group), "hard_gate_target_recall": hard,
            "selected_top1_target": top1,
            "selected_top1_target_given_hard": top1_given_hard,
            "hard_gate_denominator": hard,
            "accepted": sum(accepted(row) for row in group),
            "forced_success": sum(bool(row["forced_success"]) for row in group),
            "top1_no_rejection_success": sum(bool(row["top1_success"]) for row in group),
            "routed_after_rejection": metrics["routed"][role],
            "s1": quantiles([row["s1"] for row in group if row["s1"] is not None]),
            "gap": quantiles([row["gap"] for row in group if row["gap"] is not None]),
            "candidate_count": dict(Counter(int(row["candidate_count"]) for row in group)),
        }
    by_category = {}
    for category in CATEGORIES:
        group = [row for row in negatives if row["category"] == category]
        accepted_group = [row for row in group if accepted(row)]
        by_category[category] = {
            "count": len(group), "no_edit": len(group) - len(accepted_group),
            "false_activation": len(accepted_group),
            "exact_s0": sum(bool(row["top1_exact_s0"]) if accepted(row) else True for row in group),
            "clinical_canonical_failures": sum(
                accepted(row) and not bool(row["top1_clinical_passed"]) for row in group),
            "target_contamination": sum(
                int(row["top1_target_contamination_count"]) if accepted(row) else 0 for row in group),
            "s1": quantiles([row["s1"] for row in group if row["s1"] is not None]),
            "gap": quantiles([row["gap"] for row in group if row["gap"] is not None]),
            "candidate_count": dict(Counter(int(row["candidate_count"]) for row in group)),
        }
    return {"by_role": by_role, "by_negative_subtype": by_category,
            "positive_s1": quantiles([row["s1"] for row in positives if row["s1"] is not None]),
            "negative_s1": quantiles([row["s1"] for row in negatives if row["s1"] is not None]),
            "positive_gap": quantiles([row["gap"] for row in positives if row["gap"] is not None]),
            "negative_gap": quantiles([row["gap"] for row in negatives if row["gap"] is not None])}


def verify_membership(manifest: Mapping[str, Any], partition: str,
                      families: Sequence[Mapping[str, Any]]) -> None:
    expected = {row["family_id"] for row in manifest[partition]}
    actual = [row["family_id"] for row in families]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise RuntimeError(f"ROUTER_R2_{partition.upper()}_MEMBERSHIP_MISMATCH")


def fit(args) -> None:
    if args.selection_out.exists() or args.diagnostic_out.exists():
        raise FileExistsError(args.selection_out)
    manifest = json.loads(args.calibration_manifest.read_text())
    families, rows = load_rows(args.shard)
    verify_membership(manifest, "calibration_fit", families)
    no_rejection = apply_thresholds(rows, -1e30, -1e30)
    selection = select_dual_thresholds(rows)
    selected = selection["selected"]
    diagnostic = {
        "protocol": PROTOCOL, "candidate": "R2_E0_TOP1_ORIGINAL_COEFFICIENT_NO_REJECTION",
        "source_checkpoint": "Router-R1 step 640", "diagnostic_only": True,
        "metrics": no_rejection, "diagnostics": diagnostics(rows, no_rejection),
        "family_count": len(families), "positive_count": sum(row["kind"] == "positive" for row in rows),
        "negative_count": sum(row["kind"] == "negative" for row in rows),
        "validation_loaded": False, "heldout_loaded": False, "record953_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False,
        "worker_shards": {str(path): sha256_file(path) for path in args.shard},
    }
    write_json(args.diagnostic_out, diagnostic)
    freeze = {
        "protocol": PROTOCOL, "candidate": "R2_E0_SELECTIVE_TOP1_DUAL_MARGIN",
        "source_checkpoint": "Router-R1 step 640",
        "source_checkpoint_manifest_sha256": sha256_file(args.checkpoint_manifest),
        "thresholds": selected["thresholds"], "fit_metrics": selected,
        "fit_diagnostics": diagnostics(rows, selected),
        "candidate_grid": selection["candidate_grid"],
        "selection_objective_and_tie_break": selection["selection_order"],
        "threshold_fit_manifest_sha256": sha256_file(args.calibration_manifest),
        "calibration_lock_manifest_sha256": sha256_file(args.calibration_manifest),
        "thresholds_frozen_before_calibration_lock_read": True,
        "calibration_lock_loaded": False, "validation_loaded": False,
        "heldout_loaded": False, "record953_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False,
        "worker_shards": {str(path): sha256_file(path) for path in args.shard},
    }
    write_json(args.selection_out, freeze)
    print(json.dumps({"status": "ROUTER_R2_E0_THRESHOLDS_FROZEN",
                      "thresholds": selected["thresholds"],
                      "fit_joint_eligible": selected["joint_eligible"],
                      "no_rejection": no_rejection}, sort_keys=True))


def lock(args) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    freeze = json.loads(args.threshold_freeze.read_text())
    if not freeze.get("thresholds_frozen_before_calibration_lock_read"):
        raise RuntimeError("ROUTER_R2_LOCK_READ_BEFORE_THRESHOLD_FREEZE")
    manifest = json.loads(args.calibration_manifest.read_text())
    families, rows = load_rows(args.shard)
    verify_membership(manifest, "calibration_lock", families)
    thresholds = freeze["thresholds"]
    metrics = apply_thresholds(rows, thresholds["tau_abs"], thresholds["tau_gap"])
    if metrics["joint_eligible"]:
        label = "ROUTER_R2_E0_CALIBRATION_PASS"
    elif not metrics["effectiveness_eligible"] and not metrics["safety_eligible"]:
        label = "ROUTER_R2_E0_CALIBRATION_FAIL__JOINT"
    elif not metrics["effectiveness_eligible"]:
        label = "ROUTER_R2_E0_CALIBRATION_FAIL__EFFECTIVENESS"
    else:
        label = "ROUTER_R2_E0_CALIBRATION_FAIL__SAFETY"
    result = {**freeze, "decision_label": label,
        "calibration_lock_loaded": True, "lock_family_count": len(families),
        "lock_metrics": metrics, "lock_diagnostics": diagnostics(rows, metrics),
        "lock_worker_shards": {str(path): sha256_file(path) for path in args.shard},
        "candidate_frozen_for_validation": bool(metrics["joint_eligible"]),
        "validation_loaded": False, "heldout_loaded": False, "record953_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False}
    write_json(args.out, result)
    print(json.dumps({"status": label, "thresholds": thresholds,
                      "lock_metrics": metrics}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    current = sub.add_parser("fit")
    current.add_argument("--calibration-manifest", type=Path, required=True)
    current.add_argument("--checkpoint-manifest", type=Path, required=True)
    current.add_argument("--shard", type=Path, action="append", required=True)
    current.add_argument("--diagnostic-out", type=Path, required=True)
    current.add_argument("--selection-out", type=Path, required=True)
    current = sub.add_parser("lock")
    current.add_argument("--calibration-manifest", type=Path, required=True)
    current.add_argument("--threshold-freeze", type=Path, required=True)
    current.add_argument("--shard", type=Path, action="append", required=True)
    current.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); {"fit": fit, "lock": lock}[args.mode](args)


if __name__ == "__main__":
    main()
