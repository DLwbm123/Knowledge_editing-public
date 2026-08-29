#!/usr/bin/env python3
"""Fit and lock T1 thresholds, then select at most one checkpoint by calibration."""
from __future__ import annotations

import argparse
import json
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
from scripts.liveedit_med.select_router_r2_e0 import CATEGORIES, load_rows, verify_membership


STEPS = (80, 160, 240, 320, 400, 480, 560, 640)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def proxy_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source in rows:
        row = dict(source)
        fixed_eligible = bool(row["hard_gate_admitted"] and row["expert_is_overall_top"])
        row["candidate_count"] = int(fixed_eligible)
        row["s1"] = float(row["null_margin"])
        row["s2"] = None
        row["gap"] = float(row["expert_gap"])
        result.append(row)
    return result


def remap_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    raw = metrics["thresholds"]
    result["thresholds"] = {"tau_null": float(raw["tau_abs"]),
                            "tau_gap": float(raw["tau_gap"])}
    return result


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p25", "median", "p75", "max", "mean")}
    values = sorted(float(value) for value in values)
    pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]
    return {"min": values[0], "p25": pick(.25), "median": statistics.median(values),
            "p75": pick(.75), "max": values[-1], "mean": statistics.fmean(values)}


def accepted(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> bool:
    return (bool(row["hard_gate_admitted"]) and bool(row["expert_is_overall_top"])
            and float(row["null_margin"]) >= float(thresholds["tau_null"])
            and float(row["expert_gap"]) >= float(thresholds["tau_gap"]))


def diagnostics(rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = metrics["thresholds"]
    positives = [row for row in rows if row["kind"] == "positive"]
    negatives = [row for row in rows if row["kind"] == "negative"]
    by_role = {}
    for role in ROLES:
        group = [row for row in positives if row["role"] == role]
        hard = sum(bool(row["target_hard_recalled"]) for row in group)
        by_role[role] = {
            "count": len(group), "hard_gate_target_recall": hard,
            "selected_top1_target": sum(bool(row["target_selected_top1"]) for row in group),
            "selected_top1_target_given_hard": sum(
                bool(row["target_selected_top1"]) for row in group if row["target_hard_recalled"]),
            "hard_gate_denominator": hard, "accepted": sum(accepted(row, thresholds) for row in group),
            "forced_success": sum(bool(row["forced_success"]) for row in group),
            "top1_no_threshold_success": sum(bool(row["top1_success"]) for row in group),
            "routed_after_rejection": metrics["routed"][role],
            "positive_false_rejection": sum(not accepted(row, thresholds) for row in group),
            "null_margin": quantiles([row["null_margin"] for row in group]),
            "expert_gap": quantiles([row["expert_gap"] for row in group]),
            "candidate_count": dict(Counter(int(row["candidate_count"]) for row in group)),
        }
    by_category = {}
    for category in CATEGORIES:
        group = [row for row in negatives if row["category"] == category]
        active = [row for row in group if accepted(row, thresholds)]
        by_category[category] = {
            "count": len(group), "no_edit": len(group) - len(active),
            "false_activation": len(active),
            "exact_s0": sum(bool(row["top1_exact_s0"]) if accepted(row, thresholds) else True
                            for row in group),
            "clinical_canonical_failures": sum(
                accepted(row, thresholds) and not bool(row["top1_clinical_passed"]) for row in group),
            "target_contamination": sum(
                int(row["top1_target_contamination_count"]) if accepted(row, thresholds) else 0
                for row in group),
            "null_margin": quantiles([row["null_margin"] for row in group]),
            "expert_gap": quantiles([row["expert_gap"] for row in group]),
            "candidate_count": dict(Counter(int(row["candidate_count"]) for row in group)),
        }
    return {
        "by_role": by_role, "by_negative_subtype": by_category,
        "positive_null_margin": quantiles([row["null_margin"] for row in positives]),
        "negative_null_margin": quantiles([row["null_margin"] for row in negatives]),
        "positive_expert_gap": quantiles([row["expert_gap"] for row in positives]),
        "negative_expert_gap": quantiles([row["expert_gap"] for row in negatives]),
        "positive_hard_gate_target_recall": sum(bool(row["target_hard_recalled"]) for row in positives),
        "positive_target_top1": sum(bool(row["target_selected_top1"]) for row in positives),
        "negative_no_edit_accuracy": metrics["no_edit_count"],
        "base_no_edit_exact_parity": len(negatives),
    }


def checkpoint(args) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads(args.calibration_manifest.read_text())
    fit_families, fit_rows = load_rows(args.fit_shard)
    lock_families, lock_rows = load_rows(args.lock_shard)
    verify_membership(manifest, "calibration_fit", fit_families)
    verify_membership(manifest, "calibration_lock", lock_families)
    fit_proxy = proxy_rows(fit_rows)
    selection = select_dual_thresholds(fit_proxy)
    fit_metrics = remap_metrics(selection["selected"])
    thresholds = fit_metrics["thresholds"]
    lock_raw = apply_thresholds(proxy_rows(lock_rows), thresholds["tau_null"], thresholds["tau_gap"])
    lock_metrics = remap_metrics(lock_raw)
    checkpoint_manifest = json.loads(args.checkpoint_manifest.read_text())
    step = int(checkpoint_manifest["step"])
    if step not in STEPS:
        raise RuntimeError("ROUTER_R2_T1_CHECKPOINT_STEP")
    result = {
        "protocol": PROTOCOL, "candidate": "R2_T1_EXPLICIT_NOEDIT_KPLUS1", "step": step,
        "checkpoint_manifest_sha256": sha256_file(args.checkpoint_manifest),
        "thresholds_fitted_on": "calibration_fit", "thresholds_locked_before_lock_metrics": True,
        "thresholds": thresholds, "fit_metrics": fit_metrics,
        "fit_diagnostics": diagnostics(fit_rows, fit_metrics),
        "lock_metrics": lock_metrics, "lock_diagnostics": diagnostics(lock_rows, lock_metrics),
        "candidate_grid": selection["candidate_grid"],
        "selection_objective_and_tie_break": selection["selection_order"],
        "fit_shards": {str(path): sha256_file(path) for path in args.fit_shard},
        "lock_shards": {str(path): sha256_file(path) for path in args.lock_shard},
        "eligible": bool(lock_metrics["joint_eligible"]),
        "evaluation_data_loaded": False, "record953_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False,
    }
    write_json(args.out, result)
    print(json.dumps({"status": "ROUTER_R2_T1_CHECKPOINT_CALIBRATED", "step": step,
                      "eligible": result["eligible"], "lock": lock_metrics}, sort_keys=True))


def mechanism(row: Mapping[str, Any]) -> str:
    diag = row["lock_diagnostics"]
    roles = diag["by_role"].values()
    hard = sum(value["hard_gate_target_recall"] for value in roles)
    total = sum(value["count"] for value in roles)
    top1_given_hard = sum(value["selected_top1_target_given_hard"] for value in roles)
    selected = sum(value["selected_top1_target"] for value in roles)
    forced = sum(value["forced_success"] for value in roles)
    top1_success = sum(value["top1_no_threshold_success"] for value in roles)
    if hard and top1_given_hard / hard < .75:
        return "R2_PRIMARY_TOP1_IDENTITY_FAILURE"
    if total and hard / total < .75:
        return "R2_PRIMARY_HARD_RECALL_FAILURE"
    if selected and forced and top1_success / forced < .75:
        return "R2_PRIMARY_SINGLE_RESIDUAL_EXECUTION_FAILURE"
    return "R2_PRIMARY_REJECTION_SEPARATION_FAILURE"


def finalize(args) -> None:
    if args.out.exists() or args.report.exists():
        raise FileExistsError(args.out)
    rows = [json.loads(path.read_text()) for path in args.result]
    if sorted(row["step"] for row in rows) != list(STEPS):
        raise RuntimeError("ROUTER_R2_T1_INCOMPLETE_CALIBRATION_SET")
    eligible = [row for row in rows if row["eligible"]]
    def key(row):
        metrics = row["lock_metrics"]
        return (float(metrics["role_macro_success"]), int(metrics["routed_total"]),
                -int(metrics["false_activation"]), -int(metrics["clinical_canonical_failures"]),
                -int(metrics["target_contamination"]), -int(row["step"]))
    selected = max(eligible, key=key) if eligible else None
    best_diagnostic = max(rows, key=key)
    label = ("ROUTER_R2_T1_CALIBRATION_PASS" if selected is not None
             else "ROUTER_R2_T1_NO_ELIGIBLE_CALIBRATION_CHECKPOINT")
    payload = {
        "protocol": PROTOCOL, "candidate": "R2_T1_EXPLICIT_NOEDIT_KPLUS1",
        "decision_label": label, "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint_manifest_sha256": (None if selected is None else
                                                selected["checkpoint_manifest_sha256"]),
        "selected_thresholds": None if selected is None else selected["thresholds"],
        "selected_lock_metrics": None if selected is None else selected["lock_metrics"],
        "primary_mechanism_if_failed": None if selected is not None else mechanism(best_diagnostic),
        "best_diagnostic_step": best_diagnostic["step"],
        "checkpoint_results": rows,
        "result_hashes": {str(path): sha256_file(path) for path in args.result},
        "candidate_frozen_for_clean_evaluation": selected is not None,
        "evaluation_data_loaded": False, "record953_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False,
    }
    write_json(args.out, payload)
    lines = ["# LiveEdit-Med Router-R2 T1 calibration", "", f"Decision: `{label}`", "",
             f"- Selected step: **{payload['selected_step']}**",
             f"- Frozen for clean evaluation: **{payload['candidate_frozen_for_clean_evaluation']}**",
             f"- Primary mechanism if failed: `{payload['primary_mechanism_if_failed']}`", "",
             "No evaluation, heldout, record 953, sealed blind, or Stage-2 data was loaded."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({"status": label, "selected_step": payload["selected_step"],
                      "primary_mechanism": payload["primary_mechanism_if_failed"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    current = sub.add_parser("checkpoint")
    current.add_argument("--calibration-manifest", type=Path, required=True)
    current.add_argument("--checkpoint-manifest", type=Path, required=True)
    current.add_argument("--fit-shard", type=Path, action="append", required=True)
    current.add_argument("--lock-shard", type=Path, action="append", required=True)
    current.add_argument("--out", type=Path, required=True)
    current = sub.add_parser("finalize")
    current.add_argument("--result", type=Path, action="append", required=True)
    current.add_argument("--out", type=Path, required=True)
    current.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(); {"checkpoint": checkpoint, "finalize": finalize}[args.mode](args)


if __name__ == "__main__":
    main()
