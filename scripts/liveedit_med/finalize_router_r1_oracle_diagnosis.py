#!/usr/bin/env python3
"""Finalize Router-R1 validation-only oracle mechanism diagnosis."""
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

from methods.liveedit_med.router_r1_oracle import (
    ORACLES, PROTOCOL, ROLES, mechanism_label, sequential_success_gains,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def file_hash(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def recommendation(label: str) -> tuple[str, str]:
    mapping={
        "PRIMARY_VISUAL_HARD_RECALL_BOTTLENECK":("visual influence-scope redesign",
            "Oracle-Hard was the dominant recovery intervention."),
        "PRIMARY_DISTRACTOR_RESIDUAL_INTERFERENCE":("sparse / top-1 expert execution",
            "Removing distractor residuals was the dominant recovery intervention."),
        "PRIMARY_RELATIVE_SOFTMAX_DILUTION":("relative competition redesign",
            "Removing relative-softmax dilution was the dominant recovery intervention."),
        "PRIMARY_SIGMOID_UNDERSCALING":("decouple routing confidence from execution magnitude",
            "Full-strength execution after sigmoid attenuation was the dominant recovery intervention."),
        "MIXED_POST_RECALL_ROUTING_BOTTLENECK":("null-aware selective top-1 full-strength routing",
            "Multiple post-recall interventions materially contributed; one selective expert or NO_EDIT should be chosen, then executed at full strength."),
        "FROZEN_EXPERT_UPPER_BOUND_LIMIT":("stop router redesign and revisit expert upper bound",
            "Full-strength target-only execution remained the dominant limitation."),
        "ORACLE_DIAGNOSIS_INVALID_ENGINEERING_RUN":("no recommendation",
            "Oracle parity failed, so mechanism conclusions are invalid."),
    }
    return mapping[label]


def feature_driver(value):
    aggregates=value["aggregates"]
    base=aggregates["step_0000"]["32"];final=aggregates["step_0640"]["32"]
    def gap(row, *, norms_from=None, angles_from=None):
        norms=norms_from or row;angles=angles_from or row
        return norms["mean_input_visual_norm"]*(
            norms["mean_distractor_visual_norm"]*angles["mean_visual_distractor_cosine"]
            - norms["mean_sentinel_visual_norm"]*angles["mean_visual_sentinel_cosine"])
    baseline=gap(base);actual=gap(final)
    norm_only=gap(final,angles_from=base);angle_only=gap(base,angles_from=final)
    norm_contribution=norm_only-baseline;angle_contribution=angle_only-baseline
    driver=("feature_norm_drift" if abs(norm_contribution)>abs(angle_contribution)
            else "angular_separation_drift")
    return {"repo32_candidate_count_step0":base["mean_candidate_count"],
            "repo32_candidate_count_step640":final["mean_candidate_count"],
            "unscaled_score_gap_step0":baseline,"unscaled_score_gap_step640":actual,
            "norm_only_counterfactual_change":norm_contribution,
            "angle_only_counterfactual_change":angle_contribution,
            "primary_candidate_inflation_driver":driver,
            "aggregate_counterfactual_is_descriptive_not_exact_per_sample":True}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--out-root",type=Path,required=True)
    args=parser.parse_args();root=args.out_root
    final_path=root/"LIVEEDIT_ROUTER_R1_ORACLE_FINAL_DECISION.md"
    if final_path.exists():raise FileExistsError(final_path)
    selection=json.loads((root/"r1_safety_closure/ROUTER_R1_FINAL_SELECTION.json").read_text())
    if selection.get("selected_checkpoint") is not None or selection.get("primary_label")!="ROUTER_R1_NO_ELIGIBLE_CLEAN_VALIDATION_CHECKPOINT":
        raise RuntimeError("ROUTER_R1_ORACLE_FINAL_SELECTION_MISMATCH")
    cross=json.loads((root/"sample_cross_table/conditional_retention_summary.json").read_text())
    oracle80=json.loads((root/"oracle_step_0080/oracle_aggregate.json").read_text())
    oracle640=json.loads((root/"oracle_step_0640/oracle_aggregate.json").read_text())
    if not (oracle80["O0_exact_frozen_parity"] and oracle80["O4_exact_forced_on_parity"]
            and oracle640["O0_exact_frozen_parity"] and oracle640["O4_exact_forced_on_parity"]):
        raise RuntimeError("ROUTER_R1_ORACLE_FINAL_PARITY_FAILURE")
    label=mechanism_label(oracle640["successes"],o4_parity=True)
    gains=sequential_success_gains(oracle640["successes"])
    role_labels={};role_gains={}
    for role in ROLES:
        successes={oracle:oracle640["metrics"][oracle]["success_by_role"][role] for oracle in ORACLES}
        role_gains[role]=sequential_success_gains(successes)
        role_labels[role]=mechanism_label(successes,o4_parity=True)
    mechanism_root=root/"mechanism_attribution";mechanism_root.mkdir()
    write_json(mechanism_root/"sequential_gain_decomposition.json",{
        "protocol":PROTOCOL,"step":640,"oracle_order":list(ORACLES),
        "successes":oracle640["successes"],"overall_gains":gains,"role_gains":role_gains,
        "strict_mathematical_additivity_claimed":False})
    write_json(mechanism_root/"role_specific_labels.json",{
        "protocol":PROTOCOL,"primary_label":label,"role_specific_labels":role_labels})
    write_json(mechanism_root/"bootstrap_intervals.json",{
        "protocol":PROTOCOL,"descriptive_only":True,"not_for_selection":True,
        "step_80":oracle80["bootstrap_intervals"],"step_640":oracle640["bootstrap_intervals"]})
    (mechanism_root/"MECHANISM_ATTRIBUTION_REPORT.md").write_text(
        "# Router-R1 Mechanism Attribution\n\n"
        f"Primary label: `{label}`\n\n"
        f"Sequential success gains at step 640: `{json.dumps(gains,sort_keys=True)}`. "
        "These are ordered interventions and are not claimed to be mathematically additive outside O0->O4.\n")

    conflict=json.loads((root/"gradient_conflict/conflict_aggregate.json").read_text())
    persistent={name:value["persistent_conflict"] for name,value in conflict["positive_negative_by_group"].items()}
    feature_path=root/"feature_scale/candidate_inflation_analysis.json"
    feature=json.loads(feature_path.read_text());driver=feature_driver(feature)
    feature["final_driver_analysis"]=driver
    # The original diagnostic file is immutable; write finalized analysis separately.
    write_json(root/"feature_scale/candidate_inflation_final_analysis.json",driver)
    direction,reason=recommendation(label)
    r2=root/"router_r2_recommendation";r2.mkdir()
    (r2/"ROUTER_R2_RECOMMENDATION.md").write_text(
        "# Router-R2 Recommendation (Concept Only)\n\n"
        f"Recommended direction: **{direction}**.\n\n{reason}\n\n"
        "Router-R2 was not implemented. If the recommendation is null-aware selective top-1 full-strength routing, "
        "the concept is: select exactly one expert or NO_EDIT; require top-1 to beat NO_EDIT and the runner-up by "
        "calibrated margins; execute the chosen expert at full strength; never sum multiple expert residuals.\n")

    retention=cross["steps"]["640"]
    safety=json.loads((root/"r1_safety_closure/safety_aggregate.json").read_text())
    result_hashes={str(path.relative_to(root)):file_hash(path) for path in sorted(root.rglob("*.json"))}
    summary={
        "protocol":PROTOCOL,"primary_label":label,
        "router_r1_final_label":selection["primary_label"],"selected_checkpoint":None,
        "safety_checkpoint_count":safety["checkpoint_count"],
        "conditional_retention_step640":retention,
        "oracle_step80":oracle80,"oracle_step640":oracle640,
        "candidate_inflation_driver":driver,
        "persistent_positive_negative_gradient_conflict":persistent,
        "router_r2_recommendation":direction,"router_r2_implemented":False,
        "heldout_permitted":False,"record953_permitted":False,
        "sealed_blind_permitted":False,"stage2_permitted":False,
        "result_hashes_before_final_summary":result_hashes,
    }
    write_json(root/"oracle_diagnosis_summary.json",summary)
    rows=[]
    for role in ROLES:
        rates=retention[role]["conditional_retention"]
        rows.append(f"| {role} | {rates['P_R_given_F']} | {rates['P_R_given_F_H']} | {rates['P_R_given_F_H_T']} |")
    final_path.write_text(
        "# LiveEdit-Med Router-R1 Safety Closure and Oracle Diagnosis\n\n"
        f"Primary mechanism label: `{label}`\n\n"
        "## First-page decisions\n\n"
        "- All eight safety evaluations complete: **Yes**\n"
        "- Router-R1 finalized as no eligible checkpoint: **Yes**\n"
        "- Checkpoint selected: **No**\n"
        "- Heldout, record 953, and sealed blind touched: **No**\n"
        f"- O0->O4 successes: **{oracle640['successes']}**\n"
        f"- Sequential recovered positives: **{gains}**\n"
        "- O4 exactly reproduced forced-on: **Yes**\n"
        f"- Candidate inflation driver: **{driver['primary_candidate_inflation_driver']}**\n"
        f"- Persistent positive/negative gradient conflict by group: **{persistent}**\n"
        f"- Single Router-R2 recommendation: **{direction}**\n"
        "- Router-R2 implemented: **No**\n"
        "- Heldout permitted: **No**\n- Record 953 permitted: **No**\n"
        "- Sealed blind permitted: **No**\n- Stage-2 permitted: **No**\n\n"
        "## Correct conditional routing retention at step 640\n\n"
        "| Role | P(R given F) | P(R given F and H) | P(R given F and H and T) |\n"
        "|---|---:|---:|---:|\n"+"\n".join(rows)+"\n\n"
        "All oracle and gradient results are validation/train-only diagnostics and did not enter checkpoint selection.\n")
    print(json.dumps({"status":"ROUTER_R1_ORACLE_DIAGNOSIS_COMPLETE",
                      "primary_label":label,"router_r2_recommendation":direction},sort_keys=True))


if __name__=="__main__":main()
