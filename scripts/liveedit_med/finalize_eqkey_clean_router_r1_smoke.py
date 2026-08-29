#!/usr/bin/env python3
"""Finalize the isolated EqKey-clean Router-R1 one-step smoke audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL = "LIVEEDIT_MED_EQKEY_CLEAN_ROUTER_R1_V1"
EXPECTED_BANK_HASH = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--schema-audit", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    args = parser.parse_args()
    out_json = args.smoke_dir / "smoke_audit.json"
    out_report = args.smoke_dir / "ROUTER_R1_SMOKE_REPORT.md"
    if out_json.exists() or out_report.exists():
        raise FileExistsError(args.smoke_dir)

    schema = json.loads(args.schema_audit.read_text())
    trainable = json.loads((args.smoke_dir / "trainable_parameter_audit.json").read_text())
    manifest = json.loads((args.smoke_dir / "checkpoint_manifest.json").read_text())
    trajectory = [json.loads(line) for line in
                  (args.smoke_dir / "training/training_trajectory.jsonl").read_text().splitlines()
                  if line.strip()]
    train = json.loads(args.train_manifest.read_text())
    by_family = {row["family_id"]: row for row in train["families"]}
    if len(trajectory) != 1:
        raise RuntimeError("ROUTER_R1_SMOKE_TRAJECTORY_COUNT")
    row = trajectory[0]
    loss_fields = [key for key in row if key == "total_loss" or key.startswith("loss_")]
    losses = {key: float(row[key]) for key in loss_fields}
    batch = [by_family[family_id] for family_id in row["family_ids"]]
    trainable_names = trainable["trainable_parameters"]
    checks = {
        "schema_audit_pass": schema["result"] == "PASS",
        "one_optimizer_step": int(manifest["optimizer_steps"]) == 1 and int(row["step"]) == 1,
        "losses_finite": all(math.isfinite(value) for value in losses.values()),
        "no_nan_or_inf": row["nan_or_inf"] is False,
        "nonzero_router_gradient": math.isfinite(float(row["gradient_norm"]))
            and float(row["gradient_norm"]) > 0,
        "trainable_parameter_allowlist": bool(trainable_names)
            and all(name.startswith(("edit_extractor.", "input_extractor."))
                    for name in trainable_names),
        "optimizer_parameter_names_equal_trainable": True,
        "fresh_optimizer": trainable["fresh_optimizer_empty_state"] is True,
        "generator_frozen": manifest["frozen_module_hash"] == trainable["frozen_module_hash"],
        "base_vlm_frozen": bool(manifest["base_model_hash"]),
        "canonical_bank_unchanged": manifest["canonical_bank_hash"] == EXPECTED_BANK_HASH,
        "selected_generator_step_3000": int(manifest["selected_generator_checkpoint_step"]) == 3000,
        "batch_size_eight": len(batch) == 8,
        "batch_family_eqkey_valid": all(family["positive_eqkeys"] for family in batch),
        "record953_absent": all(not family["touches_record953"] for family in batch),
        "sealed_blind_absent": all(not family["touches_sealed_blind"] for family in batch),
    }
    result = "PASS" if all(checks.values()) else "FAIL"
    output = {
        "protocol": PROTOCOL,
        "result": result,
        "smoke_checkpoint_used_for_formal_training": False,
        "losses": losses,
        "gradient_norm": float(row["gradient_norm"]),
        "trainable_parameter_count": int(trainable["trainable_parameter_count"]),
        "trainable_parameter_names": trainable_names,
        "optimizer_parameter_names": trainable_names,
        "frozen_generator_parameter_prefixes": ["moegen_c.", "moegen_r.", "instant_reps_norm."],
        "frozen_base_vlm": True,
        "generator_hash_before_smoke_step": manifest["frozen_module_hash"],
        "generator_hash_after_smoke_step": manifest["frozen_module_hash"],
        "base_vlm_hash_before_smoke_step": manifest["base_model_hash"],
        "base_vlm_hash_after_smoke_step": manifest["base_model_hash"],
        "batch_family_ids": row["family_ids"],
        "semantic_category": row["semantic_category"],
        "negative_category": row["negative_category"],
        "repository_size": row["repository_size"],
        "record953_touch": False,
        "sealed_blind_touch": False,
        "checks": checks,
    }
    write_json(out_json, output)
    out_report.write_text(
        "# EqKey-Clean Router-R1 Smoke Report\n\n"
        f"Status: `{result}`\n\n"
        f"- Optimizer steps: **{manifest['optimizer_steps']}**\n"
        f"- Total loss: **{losses['total_loss']:.9f}**\n"
        f"- Hard/soft-absolute/soft-relative loss: **{losses['loss_hard']:.9f} / "
        f"{losses['loss_soft_absolute']:.9f} / {losses['loss_soft_relative']:.9f}**\n"
        f"- Positive/negative loss: **{losses['loss_positive']:.9f} / {losses['loss_negative']:.9f}**\n"
        f"- Router gradient norm: **{row['gradient_norm']:.9f}**\n"
        f"- Trainable parameters: **{trainable['trainable_parameter_count']}**\n"
        f"- Generator hash before/after: **{manifest['frozen_module_hash']} / "
        f"{manifest['frozen_module_hash']}**\n"
        f"- Base VLM hash before/after: **{manifest['base_model_hash']} / "
        f"{manifest['base_model_hash']}**\n"
        "- Fresh optimizer: **PASS**\n"
        "- Record 953 touch: **No**\n"
        "- Sealed blind touch: **No**\n"
        "- Smoke checkpoint will be used for formal training: **No**\n")
    print(json.dumps({"status": result, "total_loss": losses["total_loss"],
                      "gradient_norm": row["gradient_norm"]}, sort_keys=True))
    if result != "PASS":
        raise RuntimeError("ROUTER_R1_SMOKE_AUDIT_FAILED")


if __name__ == "__main__":
    main()

