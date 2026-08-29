#!/usr/bin/env python3
"""Freeze the no-eligible Router-R1 outcome after all validation safety jobs."""
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

from methods.liveedit_med.router_r1_oracle import PROTOCOL, ROLES


STEPS = (80, 160, 240, 320, 400, 480, 560, 640)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def flat_weights(route):
    values = route.get("final_weights", [])
    return values[0] if values and isinstance(values[0], list) else values


def safety_tuple(value):
    negatives = [item for row in value["rows"] for item in row["negatives"]]
    candidates = [len(item["routed"]["route"].get("candidate_ids", [])) for item in negatives]
    weights = [float(weight) for item in negatives for weight in flat_weights(item["routed"]["route"])]
    fused = [float(item["routed"]["residual_norms"]["fused_residual_norm"]) for item in negatives]
    stop_mismatch = sum(item["s0"]["stop_reason"] != item["routed"]["stop_reason"] for item in negatives)
    normalized = sum(item["s0"]["raw_output"].strip().casefold()
                     == item["routed"]["raw_output"].strip().casefold() for item in negatives)
    by_category = {}
    for category in sorted({item["category"] for item in negatives}):
        rows = [item for item in negatives if item["category"] == category]
        by_category[category] = {"count": len(rows), "exact_s0": sum(item["exact_s0"] for item in rows),
                                 "false_activation": sum(bool(item["routed"]["route"].get("candidate_ids")) for item in rows)}
    return {
        "hard_negative_count": len(negatives),
        "exact_s0": sum(item["exact_s0"] for item in negatives),
        "normalized_s0": normalized,
        "clinical_canonical_failures": sum(not item["clinical"]["passed"] for item in negatives),
        "target_contamination": sum(len(item["target_contaminations"]) for item in negatives),
        "stop_reason_mismatch": stop_mismatch,
        "mean_nll_drift": sum(float(item["nll_drift"]) for item in negatives) / len(negatives),
        "max_abs_nll_drift": max(abs(float(item["nll_drift"])) for item in negatives),
        "mean_candidate_count": sum(candidates) / len(candidates),
        "false_activation_count": sum(value > 0 for value in candidates),
        "maximum_negative_final_weight": max(weights, default=0.0),
        "mean_sum_final_weights": sum(sum(flat_weights(item["routed"]["route"])) for item in negatives) / len(negatives),
        "maximum_fused_residual_norm": max(fused, default=0.0),
        "mean_fused_residual_norm": sum(fused) / len(fused),
        "category_breakdown": by_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    target = args.out_root / "r1_safety_closure"
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)

    rows, safety_inputs = [], None
    for step in STEPS:
        main_path = args.formal_run / "evaluation/validation_main" / f"checkpoint_{step:04d}.json"
        safety_path = args.formal_run / "evaluation/validation_safety" / f"checkpoint_{step:04d}.json"
        if not main_path.is_file() or not safety_path.is_file():
            raise RuntimeError(f"ROUTER_R1_SAFETY_INCOMPLETE:{step}")
        main_value, safety_value = json.loads(main_path.read_text()), json.loads(safety_path.read_text())
        identities = [(row["family_id"], item["category"], item["eqkey"])
                      for row in safety_value["rows"] for item in row["negatives"]]
        if safety_inputs is None:
            safety_inputs = identities
        if identities != safety_inputs or len(identities) != 160:
            raise RuntimeError(f"ROUTER_R1_SAFETY_INPUT_DRIFT:{step}")
        if (safety_value.get("record953_loaded") is not False
                or safety_value.get("sealed_blind_loaded") is not False
                or not all(safety_value["eqkey_role_audit"].values())):
            raise RuntimeError(f"ROUTER_R1_SAFETY_BOUNDARY_FAILURE:{step}")
        copied = target / f"safety_checkpoint_{step:04d}.json"
        for family_row in safety_value["rows"]:
            for item in family_row["negatives"]:
                item["checkpoint_step"] = step
                item["family_id"] = family_row["family_id"]
                item["negative_category"] = item["category"]
                item["EqKey"] = item["eqkey"]
        write_new(copied, safety_value)
        repo = main_value["repository_sizes"]["32"]
        forced = main_value["forced_on"]
        floors = {"native": math.ceil(.90 * forced["native"]),
                  **{role: math.ceil(.75 * forced[role]) for role in ("textual", "visual", "paired")}}
        effectiveness = {role: int(repo["routed"][role]) for role in ROLES}
        eligible = (all(effectiveness[role] >= floors[role] for role in ROLES)
                    and repo["target_contaminations"] == 0
                    and repo["clinical_canonical_failures"] == 0)
        rows.append({
            "step": step, "effectiveness": effectiveness, "forced_on": forced,
            "effectiveness_floors": floors, "effectiveness_eligible": eligible,
            "fixed_locality_exact": repo["locality_exact_preservation"],
            "fixed_locality_total": repo["locality_total"],
            "routing_false_positives": repo["routing_false_positives"],
            "target_contamination": repo["target_contaminations"],
            "clinical_canonical_failures": repo["clinical_canonical_failures"],
            "source_validation_loss": main_value["source_validation_loss"],
            "source_validation_loss_metadata": {"router_sensitive": False,
                "invariant_diagnostic_only": True},
            "safety": safety_tuple(safety_value),
            "main_result_sha256": file_hash(main_path), "safety_result_sha256": file_hash(safety_path),
            "copied_safety_sha256": file_hash(copied),
        })

    if any(row["effectiveness_eligible"] for row in rows):
        raise RuntimeError("ROUTER_R1_ORACLE_AGGREGATION_DISCREPANCY")
    aggregate = {"protocol": PROTOCOL, "checkpoint_count": len(rows),
                 "identical_safety_input_count": len(safety_inputs), "rows": rows,
                 "record953_used": False, "heldout_loaded": False, "sealed_blind_loaded": False}
    write_new(target / "safety_aggregate.json", aggregate)
    selection = {
        "protocol": PROTOCOL, "primary_label": "ROUTER_R1_NO_ELIGIBLE_CLEAN_VALIDATION_CHECKPOINT",
        "selected_checkpoint": None, "rows": rows,
        "safety_cannot_override_effectiveness_eligibility": True,
        "heldout_permitted": False, "record953_permitted": False,
        "sealed_blind_permitted": False, "stage2_permitted": False,
    }
    write_new(target / "ROUTER_R1_FINAL_SELECTION.json", selection)
    table = "\n".join(
        f"| {row['step']} | {row['effectiveness']['native']} | {row['effectiveness']['textual']} | "
        f"{row['effectiveness']['visual']} | {row['effectiveness']['paired']} | "
        f"{row['safety']['exact_s0']}/160 | No |" for row in rows)
    (target / "ROUTER_R1_FINAL_DECISION.md").write_text(
        "# Router-R1 Final Validation Decision\n\n"
        "Primary label: `ROUTER_R1_NO_ELIGIBLE_CLEAN_VALIDATION_CHECKPOINT`\n\n"
        "No checkpoint was selected. Safety was evaluated for every checkpoint but cannot override "
        "the frozen effectiveness floor.\n\n"
        "| Step | Native | Textual | Visual | Paired | Safety exact S0 | Eligible |\n"
        "|---:|---:|---:|---:|---:|---:|:---:|\n" + table + "\n\n"
        "- `source_validation_loss`: invariant diagnostic only; router-sensitive = false.\n"
        "- Heldout permitted: **No**\n- Record 953 permitted: **No**\n"
        "- Sealed blind permitted: **No**\n- Stage-2 permitted: **No**\n")
    print(json.dumps({"status": selection["primary_label"], "selected_checkpoint": None}, sort_keys=True))


if __name__ == "__main__":
    main()
