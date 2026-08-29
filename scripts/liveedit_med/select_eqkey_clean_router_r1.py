#!/usr/bin/env python3
"""Select an EqKey-clean Router-R1 checkpoint on purged validation only."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.router_r1 import CHECKPOINT_STEPS, selection_key
from scripts.liveedit_med.eqkey_fixed_expert_utils import sha256_file


PROTOCOL = "LIVEEDIT_MED_EQKEY_CLEAN_ROUTER_R1_V1"
ROLES = ("native", "textual", "visual", "paired")


def flattened_route_weights(route: Mapping[str, Any]) -> list[Any]:
    """Return per-candidate weights, accepting the audited singleton batch axis."""
    ids = route.get("candidate_ids", [])
    weights = route.get("final_weights", [])
    if len(weights) == 1 and isinstance(weights[0], list):
        weights = weights[0]
    if len(weights) != len(ids):
        raise RuntimeError(
            f"ROUTER_R1_ROUTE_WEIGHT_SHAPE_MISMATCH:ids={len(ids)}:weights={len(weights)}"
        )
    return weights


def relative_failures(value: Mapping[str, Any]) -> int:
    failures = 0
    for row in value["rows"]:
        target = row["family_id"]
        for generation in row["repositories"]["32"]["routed"].values():
            route = generation["route"]
            ids = route.get("candidate_ids", [])
            weights = flattened_route_weights(route)
            if target not in ids:
                failures += 1
                continue
            target_weight = float(weights[ids.index(target)])
            if any(float(weight) >= target_weight for candidate, weight in zip(ids, weights)
                   if candidate != target):
                failures += 1
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--safety", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if any((args.out_dir / name).exists() for name in ("validation_results.jsonl", "checkpoint_selection.json")):
        raise FileExistsError(args.out_dir)
    by_step = {int(value["step"]): (path, value) for path in args.result
               for value in [json.loads(path.read_text())]}
    safety = {int(value["selected_step"]): (path, value) for path in args.safety
              for value in [json.loads(path.read_text())]}
    if tuple(sorted(by_step)) != CHECKPOINT_STEPS or tuple(sorted(safety)) != CHECKPOINT_STEPS:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_VALIDATION_CHECKPOINT_SET")
    forced_sets = {json.dumps(value[1]["forced_on"], sort_keys=True) for value in by_step.values()}
    family_hashes = {json.dumps([row["family_id"] for row in value[1]["rows"]]) for value in by_step.values()}
    if len(forced_sets) != 1 or len(family_hashes) != 1:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_VALIDATION_DRIFT")
    forced = next(iter(by_step.values()))[1]["forced_on"]
    rows = []
    for step in CHECKPOINT_STEPS:
        result_path, result = by_step[step]; safety_path, safe = safety[step]
        repo = result["repository_sizes"]["32"]
        metrics = {"protocol": PROTOCOL, "step": step,
            "result_path": str(result_path.resolve()), "result_sha256": sha256_file(result_path),
            "safety_path": str(safety_path.resolve()), "safety_sha256": sha256_file(safety_path),
            "routed_native": int(repo["routed"]["native"]),
            "routed_textual": int(repo["routed"]["textual"]),
            "routed_visual": int(repo["routed"]["visual"]),
            "routed_paired": int(repo["routed"]["paired"]),
            "negative_locality_exact_s0": int(safe["hard_negative_exact_s0"]),
            "negative_locality_total": int(safe["hard_negative_count"]),
            "mean_candidate_count": float(repo["mean_candidate_count"]),
            "text_relative_competition_failures": relative_failures(result),
            "negative_locality_kl": float(safe["mean_negative_locality_kl"]),
            "target_contamination": int(repo["target_contaminations"] + safe["target_contaminations"]),
            "clinical_canonical_failures": int(repo["clinical_canonical_failures"]
                                                + safe["clinical_canonical_failures"]),
            "fixed_locality_exact": int(repo["locality_exact_preservation"]),
            "fixed_locality_total": int(repo["locality_total"]), "forced_on": forced}
        metrics["eligible"] = bool(
            metrics["routed_native"] >= math.ceil(.90 * forced["native"])
            and metrics["routed_textual"] >= math.ceil(.75 * forced["textual"])
            and metrics["routed_visual"] >= math.ceil(.75 * forced["visual"])
            and metrics["routed_paired"] >= math.ceil(.75 * forced["paired"])
            and metrics["target_contamination"] == 0
            and metrics["clinical_canonical_failures"] == 0)
        rows.append(metrics)
    eligible = [row for row in rows if row["eligible"]]
    selected = min(eligible, key=selection_key) if eligible else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "validation_results.jsonl").open("x") as handle:
        for row in rows: handle.write(json.dumps(row, sort_keys=True) + "\n")
    output = {"protocol": PROTOCOL,
        "label": "ROUTER_R1_NO_ELIGIBLE_CLEAN_VALIDATION_CHECKPOINT" if selected is None
                 else "EQKEY_CLEAN_ROUTER_R1_CHECKPOINT_SELECTED__NO_TEST_LEAKAGE",
        "selected_step": None if selected is None else selected["step"],
        "selected_metrics": selected, "forced_on_upper_bound": forced,
        "eligible_steps": [row["step"] for row in eligible], "heldout_used": False,
        "record953_used": False, "sealed_blind_loaded": False}
    (args.out_dir / "checkpoint_selection.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["label"], "selected_step": output["selected_step"]}, sort_keys=True))


if __name__ == "__main__":
    main()
