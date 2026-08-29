#!/usr/bin/env python3
"""Finalize adapted EqKey-clean Router-R1 held-out gates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL = "LIVEEDIT_MED_EQKEY_CLEAN_ROUTER_R1_V1"
ROLES = ("native", "textual", "visual", "paired")


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--main-result", type=Path, required=True)
    parser.add_argument("--safety-result", type=Path, required=True)
    parser.add_argument("--reproducibility", type=Path, required=True)
    parser.add_argument("--baseline-decision", type=Path, required=True)
    parser.add_argument("--baseline-safety", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    selection=json.loads(args.selection.read_text()); main=json.loads(args.main_result.read_text())
    safety=json.loads(args.safety_result.read_text()); repro=json.loads(args.reproducibility.read_text())
    baseline=json.loads(args.baseline_decision.read_text()); baseline_safety=json.loads(args.baseline_safety.read_text())
    step=selection["selected_step"]
    if step is None or int(main["step"])!=int(step) or int(safety["selected_step"])!=int(step) or not repro["passed"]:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_FINAL_INPUT_MISMATCH")
    repo=main["repository_sizes"]["32"]; totals=main["role_totals"]
    rates={role:repo["routed"][role]/totals[role] for role in ROLES}
    baseline_repo=baseline["repository_summaries"]["32"]
    baseline_rates=baseline_repo["routed_rates"]
    forced_unchanged=main["forced_on"]==selection["forced_on_upper_bound"]
    hard_rate=safety["hard_negative_exact_s0"]/safety["hard_negative_count"]
    baseline_hard_rate=(baseline_safety["hard_negative_exact_s0"]/
                        baseline_safety["hard_negative_count"])
    contamination=repo["target_contaminations"]+safety["target_contaminations"]
    clinical=repo["clinical_canonical_failures"]+safety["clinical_canonical_failures"]
    baseline_contamination=baseline["safety_locality"]["target_contaminations"]
    baseline_clinical=baseline["safety_locality"]["clinical_canonical_failures"]
    full_pass=bool(forced_unchanged and rates["native"]>=.75 and rates["textual"]>=.70
        and rates["visual"]>=.70 and rates["paired"]>=.70 and hard_rate>=.95
        and repo["locality_exact_preservation"]==repo["locality_total"]
        and contamination==0 and clinical==0 and repro["passed"])
    visual_paired_gain=((repo["routed"]["visual"]+repo["routed"]["paired"])
        -(baseline_repo["routed_success"]["visual"]+baseline_repo["routed_success"]["paired"]))/64
    directional=bool(not full_pass and visual_paired_gain>=.10
        and hard_rate-baseline_hard_rate>=.15
        and rates["native"]>=baseline_rates["native"]-.05
        and rates["textual"]>=baseline_rates["textual"]-.05
        and contamination<=baseline_contamination and clinical<baseline_clinical)
    label=("PASS_LIVEEDIT_EQKEY_CLEAN_ROUTER_R1" if full_pass else
           "ROUTER_R1_DIRECTIONAL_GAIN_WITHOUT_FULL_GATE" if directional else
           "ROUTER_R1_NO_GAIN_ON_EQKEY_CLEAN_SPLIT")
    output={"protocol":PROTOCOL,"primary_label":label,"selected_step":step,
        "forced_on_unchanged":forced_unchanged,"routed_success":repo["routed"],
        "routed_rates":rates,"hard_negative_exact_s0":safety["hard_negative_exact_s0"],
        "hard_negative_count":safety["hard_negative_count"],"hard_negative_rate":hard_rate,
        "fixed_locality_exact":repo["locality_exact_preservation"],
        "fixed_locality_total":repo["locality_total"],"target_contamination":contamination,
        "clinical_canonical_failures":clinical,"visual_paired_gain":visual_paired_gain,
        "hard_negative_gain":hard_rate-baseline_hard_rate,"full_gate_passed":full_pass,
        "directional_gain_rule_passed":directional,"reproducibility_passed":repro["passed"],
        "record953_used_for_fitting_or_selection":False,"sealed_blind_loaded":False,
        "blind_evaluation_permitted":False,"stage2_permitted":False}
    write_json(args.out_dir/"heldout_results.json",output)
    (args.out_dir/"ROUTER_R1_CLEAN_REPORT.md").write_text(
        "# EqKey-Clean Router-R1 Report\n\n"
        f"Primary label: `{label}`\n\n"
        f"- Selected R1 checkpoint: **{step}**\n"
        f"- Frozen forced-on upper bounds unchanged: **{'Yes' if forced_unchanged else 'No'}**\n"
        f"- Routed native/textual/visual/paired: **{repo['routed']['native']}/{repo['routed']['textual']}/{repo['routed']['visual']}/{repo['routed']['paired']} of 32**\n"
        f"- Hard-negative exact S0: **{safety['hard_negative_exact_s0']}/{safety['hard_negative_count']}**\n"
        f"- Fixed locality exact S0: **{repo['locality_exact_preservation']}/{repo['locality_total']}**\n"
        f"- Target contamination: **{contamination}**\n"
        f"- Clinical/canonical failures: **{clinical}**\n"
        f"- Parity/reload/fresh/replay/rollback: **{'PASS' if repro['passed'] else 'FAIL'}**\n"
        "- Record 953 used for fitting/selection: **No**\n"
        "- Sealed blind opened: **No**\n"
        "- Blind evaluation permitted: **No**\n"
        "- Stage-2 permitted: **No**\n")
    regression=args.out_dir.parent/"record953_regression";regression.mkdir(exist_ok=True)
    (regression/"DEVELOPMENT_REGRESSION_REPORT.md").write_text(
        "# Record 953 Development Regression\n\n"
        "Status: `NOT_RUN__OPTIONAL_DEVELOPMENT_REGRESSION`\n\n"
        "Record 953 was excluded from fitting, checkpoint selection, hard-negative mining, and promotion. "
        "The adapted held-out label above is frozen and cannot be altered by record 953.\n")
    print(json.dumps({"status":label,"selected_step":step},sort_keys=True))


if __name__=="__main__":
    main()
