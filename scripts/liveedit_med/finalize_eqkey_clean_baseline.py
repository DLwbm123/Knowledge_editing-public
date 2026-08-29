#!/usr/bin/env python3
"""Finalize clean held-out core/full-system gates from frozen raw results."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.eqkey_clean_fast import (PROTOCOL, exact_mcnemar, wilson_interval)


ROLES = ("native", "textual", "visual", "paired")


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-result", type=Path, required=True)
    parser.add_argument("--safety-result", type=Path, required=True)
    parser.add_argument("--reproducibility", type=Path, required=True)
    parser.add_argument("--split-label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    heldout = json.loads(args.heldout_result.read_text())
    safety = json.loads(args.safety_result.read_text())
    repro = json.loads(args.reproducibility.read_text())
    if heldout.get("protocol") != PROTOCOL or heldout.get("split") != "heldout":
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:heldout")
    if safety.get("selected_step") != heldout.get("step") or not repro.get("passed"):
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:baseline_inputs")
    if any(path.exists() for path in (args.out_dir / "s0_vs_forced_on.json",
                                      args.out_dir / "statistical_tests.json")):
        raise FileExistsError(args.out_dir)

    family_roles = {}
    for row in heldout["rows"]:
        family_roles[row["family_id"]] = {view["eqkey"]: view["role"] for view in row["views"]}
    paired = {role: {"s0": [], "forced": []} for role in ROLES}
    for row in heldout["rows"]:
        for eqkey, role in family_roles[row["family_id"]].items():
            paired[role]["s0"].append(bool(row["s0"][eqkey]["match"]["success"]))
            paired[role]["forced"].append(bool(row["forced_on"][eqkey]["match"]["success"]))
    comparisons, tests = {}, {}
    for role in ROLES:
        total = len(paired[role]["s0"]); s0_count = sum(paired[role]["s0"])
        forced_count = sum(paired[role]["forced"]); improvement = (forced_count - s0_count) / total
        tests[role] = exact_mcnemar(paired[role]["s0"], paired[role]["forced"])
        comparisons[role] = {
            "total": total, "s0_success": s0_count, "forced_on_success": forced_count,
            "absolute_improvement": improvement, "absolute_improvement_percentage_points": improvement * 100,
            "s0_wilson_95": wilson_interval(s0_count, total),
            "forced_on_wilson_95": wilson_interval(forced_count, total),
        }
    full_lane = args.split_label == "EQKEY_PURGED_FULL_FAST_LANE"
    parity = bool(repro["manual_no_cache_cached_hf_parity"] and repro["passed"])
    if full_lane:
        native_core = comparisons["native"]["absolute_improvement"] > 0 and tests["native"]["p_value_two_sided_exact"] < .05
        semantic_core = sum(comparisons[role]["absolute_improvement"] > 0
                            and tests[role]["p_value_two_sided_exact"] < .05
                            for role in ("textual", "visual", "paired")) >= 2
        core_confirmed = bool(native_core and semantic_core and parity)
        rule = "FULL_FAST_LANE_EXACT_MCNEMAR"
    else:
        native_core = comparisons["native"]["absolute_improvement"] >= .20
        semantic_core = sum(comparisons[role]["absolute_improvement"] >= .15
                            for role in ("textual", "visual", "paired")) >= 2
        core_confirmed = bool(native_core and semantic_core and parity)
        rule = "LOW_SAMPLE_DIRECTIONAL"
    write_json(args.out_dir / "s0_vs_forced_on.json", {
        "protocol": PROTOCOL, "selected_step": heldout["step"], "split_label": args.split_label,
        "comparisons": comparisons, "core_rule": rule, "native_core_pass": native_core,
        "semantic_core_pass": semantic_core, "generation_parity_pass": parity,
        "core_effect_confirmed": core_confirmed,
    })
    write_json(args.out_dir / "statistical_tests.json", {
        "protocol": PROTOCOL, "tests": tests, "method": "EXACT_TWO_SIDED_MCNEMAR",
        "confidence_intervals": "WILSON_95", "success_matcher_frozen": True,
    })

    repository_summaries = {}
    for size in (1, 10, 32):
        raw = heldout["repository_sizes"][str(size)]
        current = {
            "protocol": PROTOCOL, "repository_size": size, "role_totals": heldout["role_totals"],
            "routed_success": raw["routed"],
            "routed_rates": {role: raw["routed"][role] / heldout["role_totals"][role] for role in ROLES},
            "forced_on_success": heldout["forced_on"], "s0_success": heldout["s0"],
            "locality_exact_preservation": raw["locality_exact_preservation"],
            "locality_total": raw["locality_total"], "routing_false_positives": raw["routing_false_positives"],
            "target_contaminations": raw["target_contaminations"],
            "clinical_canonical_failures": raw["clinical_canonical_failures"],
            "mean_candidate_count": raw["mean_candidate_count"],
        }
        repository_summaries[str(size)] = current
        write_json(args.out_dir / f"repo_size_{size}.json", current)
    repo32 = repository_summaries["32"]
    hard_gate = safety["hard_negative_exact_s0"] >= math.ceil(.95 * safety["hard_negative_count"])
    locality_gate = repo32["locality_exact_preservation"] == repo32["locality_total"]
    contamination = safety["target_contaminations"] + repo32["target_contaminations"]
    clinical_failures = safety["clinical_canonical_failures"] + repo32["clinical_canonical_failures"]
    routed_gate = (repo32["routed_rates"]["native"] >= .75
                   and all(repo32["routed_rates"][role] >= .70 for role in ("textual", "visual", "paired")))
    full_system_pass = bool(core_confirmed and routed_gate and hard_gate and locality_gate
                            and contamination == 0 and clinical_failures == 0 and parity)
    safety_locality = {
        "protocol": PROTOCOL, "hard_negative_exact_s0": safety["hard_negative_exact_s0"],
        "hard_negative_count": safety["hard_negative_count"], "hard_negative_gate_95": hard_gate,
        "fixed_locality_exact_s0": repo32["locality_exact_preservation"],
        "fixed_locality_count": repo32["locality_total"], "fixed_locality_gate_100": locality_gate,
        "target_contaminations": contamination, "clinical_canonical_failures": clinical_failures,
        "mean_hard_negative_nll_drift": safety["mean_nll_drift"],
        "eqkey_role_audit": safety["eqkey_role_audit"],
    }
    write_json(args.out_dir / "safety_locality.json", safety_locality)
    material_gap = any((heldout["forced_on"][role] - repo32["routed_success"][role])
                       / heldout["role_totals"][role] >= .10 for role in ROLES)
    material_gap = bool(material_gap or not hard_gate or contamination > 0 or clinical_failures > 0)
    if not core_confirmed:
        label = "LIVEEDIT_GENERATOR_NOT_CONFIRMED_ON_EQKEY_CLEAN_SPLIT"
    elif full_system_pass:
        label = "PASS_LIVEEDIT_EQKEY_CLEAN_FULL_SYSTEM"
    else:
        label = "LIVEEDIT_CORE_EFFECT_CONFIRMED_ROUTING_STILL_FAILS"
    router_r1_permitted = bool(core_confirmed and not full_system_pass and material_gap)
    decision = {
        "protocol": PROTOCOL, "primary_label": label, "selected_step": heldout["step"],
        "core_effect_confirmed": core_confirmed, "unadapted_full_system_pass": full_system_pass,
        "material_forced_on_routed_or_safety_gap": material_gap,
        "router_r1_permitted": router_r1_permitted, "record953_used": False,
        "sealed_blind_opened": False, "blind_evaluation_permitted": False, "stage2_permitted": False,
        "repository_summaries": repository_summaries, "safety_locality": safety_locality,
    }
    write_json(args.out_dir / "baseline_decision.json", decision)
    (args.out_dir / "BASELINE_CLEAN_REPORT.md").write_text(
        "# EqKey-Clean Baseline Report\n\n"
        f"Primary label: `{label}`\n\n"
        f"- Selected strict-source checkpoint: **{heldout['step']}**\n"
        f"- LiveEdit core effect confirmed: **{'Yes' if core_confirmed else 'No'}**\n"
        f"- Unadapted full repository passed: **{'Yes' if full_system_pass else 'No'}**\n"
        f"- Router-R1 permitted: **{'Yes' if router_r1_permitted else 'No'}**\n"
        f"- Hard-negative exact S0: **{safety['hard_negative_exact_s0']}/{safety['hard_negative_count']}**\n"
        f"- Fixed locality exact S0: **{repo32['locality_exact_preservation']}/{repo32['locality_total']}**\n"
        f"- Target contamination: **{contamination}**\n"
        f"- Clinical/canonical failures: **{clinical_failures}**\n"
        f"- Three-path parity/reload/replay/rollback: **{'PASS' if repro['passed'] else 'FAIL'}**\n"
        "- Record 953 used for selection: **No**\n"
        "- Sealed blind set opened: **No**\n"
        "- Blind evaluation permitted: **No**\n"
        "- Stage-2 permitted: **No**\n"
    )
    print(json.dumps({key: decision[key] for key in ("primary_label", "selected_step",
        "core_effect_confirmed", "unadapted_full_system_pass", "router_r1_permitted")}, sort_keys=True))


if __name__ == "__main__":
    main()
