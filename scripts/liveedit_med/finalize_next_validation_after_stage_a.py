#!/usr/bin/env python3
"""Finalize the frozen A-E validation after a real Stage-A execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DECISION = "PORT_FIDELITY_BLOCKS_METHOD_CONCLUSION"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage-a-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    a = json.loads((args.stage_a_dir / "trace_parity_summary.json").read_text())
    direct = json.loads((args.stage_a_dir / "upstream_forward_from_mid_layer_validation.json").read_text())
    b_path = args.root / "stage_b_official_style_medical/official_style_medical_aggregate.json"
    c_path = args.root / "stage_c_validation_routing/routing_attribution.json"
    d_path = args.root / "stage_d_assistant_only/assistant_only_diagnostic.json"
    e_path = args.root / "stage_e_future_blind/future_blind_manifest.json"
    immutability_path = args.root / "continuation_immutability.json"
    b, c, d, e, immutable = [json.loads(path.read_text()) for path in (b_path, c_path, d_path, e_path, immutability_path)]
    m = b["aggregate"]["32"]["metrics"]
    forced = sum(m[name]["forced_generation_success"] for name in ("native", "textual", "visual", "paired"))
    routed = sum(m[name]["routed_generation_success"] for name in ("native", "textual", "visual", "paired"))
    classes = c["scaling"]["32"]["failure_classes"]
    summary = {
        "decision": DECISION,
        "stage_a": a,
        "stage_a_direct_source_validation": direct,
        "stage_b_repo32": {
            "forced_generation_success": forced,
            "routed_generation_success": routed,
            "hard_safety_exact_s0": m["hard_medical_safety"]["exact_s0"],
            "hard_safety_count": m["hard_medical_safety"]["count"],
            "target_contaminations": m["hard_medical_safety"]["target_contaminations"],
            "image_locality_exact": m["image_locality"]["exact"],
            "text_locality_exact": m["text_locality"]["exact"],
        },
        "stage_c_repo32": c["scaling"]["32"],
        "stage_d": d["validation"],
        "stage_e": {key: e[key] for key in ("status", "selected_count", "input_count", "manifest_hash", "edited_checkpoint_loaded")},
        "immutability": immutable,
        "method_conclusion_permitted": False,
        "router_training_permitted": False,
        "required_next_action": "repair_layer21_training_continuation_then_rerun_stage_a_before_new_source_training",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact_hashes": {str(path.relative_to(args.root)): sha256(path) for path in
                                   (b_path, c_path, d_path, e_path, immutability_path)},
    }
    write_json(args.out_dir / "next_validation_summary.json", summary)
    write_json(args.out_dir / "state_and_bank_hash_ledger.json", immutable)
    lines = [
        "# LiveEdit-Med next validation: final decision", "",
        f"Decision: `{DECISION}`", "",
        "## First-page result", "",
        f"- Stage A official-backbone trace parity: **{a['passed_boundaries']}/{a['required_boundaries']}**, `{a['status']}`.",
        f"- Failed boundaries: **{', '.join(a['failed_boundaries'])}**.",
        f"- Direct pinned-source validation: layer-21 reapplication error **{direct['direct_vs_manual_upstream_max_abs_error']}**; current-port error **{direct['direct_vs_current_port_max_abs_error']}**.",
        f"- Stage B repository-32 forced-on/routed natural generation: **{forced}/256 vs {routed}/256**.",
        f"- Stage C repository-32 positive generation / negative exact S0: **{c['scaling']['32']['positive_generation_success']}/256 / {c['scaling']['32']['negative_exact_s0']}/256**.",
        f"- Stage D source-full / assistant-only success: **{d['validation']['source_full_success']}/256 vs {d['validation']['assistant_only_success']}/256**.",
        f"- Stage E frozen blind set: **{e['selected_count']} edits / {e['input_count']} inputs**, manifest `{e['manifest_hash']}`.",
        f"- Training tree and canonical bank byte-identical: **{immutable['training_byte_identical'] and immutable['canonical_bank_byte_identical']}**.", "",
        "## Interpretation", "",
        "The execution-level inference trace passes through final logits, but the pinned upstream source-loss continuation re-injects the captured layer-21 output at layer 21. The current port starts at layer 22. This changes reliability, generality, locality losses and the complete Adam update.", "",
        "Consequently, Stages B-D remain useful diagnostics for the current port, but they cannot establish a LiveEdit method conclusion or justify router-only adaptation. Do not describe the result as a failed LiveEdit reproduction; it is a port-fidelity failure.", "",
        "## Required next action", "",
        "Repair the layer-21 source-training continuation in a separate implementation, rerun Stage A to 27/27, and only then start a new source-faithful training run. Preserve the present 3200-step run as an immutable diagnostic baseline.", "",
        "## Repository-32 routing classes", "", "```json", json.dumps(classes, indent=2, sort_keys=True), "```", "",
    ]
    (args.out_dir / "LIVEEDIT_MED_NEXT_VALIDATION_FINAL_DECISION.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
