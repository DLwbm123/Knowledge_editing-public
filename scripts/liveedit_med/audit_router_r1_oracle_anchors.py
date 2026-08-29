#!/usr/bin/env python3
"""Create the immutable preflight audit for Router-R1 oracle diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.router_r1_oracle import PROTOCOL
from methods.liveedit_med.serialization import load_safe_state
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest


STEPS = (80, 160, 240, 320, 400, 480, 560, 640)
EXPECTED_BANK = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"
EXPECTED_FROZEN = "d1d5ce232ad2aeb7a29c4c5586a6af5dcdee064321cfc94c3393d576b1bc2249"
EXPECTED_BASE = "d8b7032a563e32f22fd51eb65d92bbb0177c913d19c5c1e6ce6ad73d0e5ca75d"


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


def hard_stop(reason: str) -> None:
    raise RuntimeError(f"ROUTER_R1_ORACLE_ANCHOR_MISMATCH:{reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-run", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError(args.out_root)
    args.out_root.mkdir(parents=True)

    validation_path = args.parent_run / "purged_split/purged_validation_manifest.json"
    validation = json.loads(validation_path.read_text())
    families = validation.get("families", [])
    if validation.get("split") != "validation" or validation.get("family_count") != 32 or len(families) != 32:
        hard_stop("validation_manifest")
    forbidden = [row["family_id"] for row in families if row.get("touches_record953")
                 or row.get("touches_sealed_blind") or row.get("touches_original_train")
                 or row.get("target_conflict")]
    if forbidden:
        hard_stop("validation_forbidden_family")

    fixed_path = args.parent_run / ".runtime_router_r1/fixed_experts/manifest.json"
    fixed = json.loads(fixed_path.read_text())
    if (fixed.get("canonical_bank_hash") != EXPECTED_BANK
            or fixed.get("generator_checkpoint_step") != 3000
            or fixed.get("record953_used") is not False
            or fixed.get("sealed_blind_loaded") is not False):
        hard_stop("fixed_experts")
    if bank_manifest()["sha256"] != EXPECTED_BANK:
        hard_stop("canonical_bank")

    main_rows = []
    family_order = None
    forced = None
    checkpoint_rows = []
    for step in STEPS:
        directory = args.formal_run / "training" / f"checkpoint_{step:04d}"
        state, manifest = load_safe_state(directory)
        if (int(manifest.get("step", -1)) != step
                or manifest.get("canonical_bank_hash") != EXPECTED_BANK
                or manifest.get("frozen_module_hash") != EXPECTED_FROZEN
                or manifest.get("base_model_hash") != EXPECTED_BASE):
            hard_stop(f"checkpoint_{step}")
        checkpoint_rows.append({
            "step": step, "tensor_count": len(state), "tensor_hashes_verified": True,
            "manifest_sha256": file_hash(directory / "manifest.json"),
            "model_sha256": file_hash(directory / "model.safetensors"),
            "frozen_module_hash": manifest["frozen_module_hash"],
            "base_model_hash": manifest["base_model_hash"],
            "canonical_bank_hash": manifest["canonical_bank_hash"],
        })
        del state
        result_path = args.formal_run / "evaluation/validation_main" / f"checkpoint_{step:04d}.json"
        result = json.loads(result_path.read_text())
        ids = [row["family_id"] for row in result["rows"]]
        if result.get("split") != "validation" or result.get("family_count") != 32:
            hard_stop(f"main_{step}")
        if family_order is None:
            family_order = ids
            forced = result["forced_on"]
        if ids != family_order or result["forced_on"] != forced:
            hard_stop(f"main_drift_{step}")
        main_rows.append({"step": step, "sha256": file_hash(result_path),
                          "forced_on": result["forced_on"],
                          "repository_32": result["repository_sizes"]["32"]})

    if family_order != [row["family_id"] for row in families]:
        hard_stop("validation_order")
    if forced != {"native": 19, "textual": 17, "visual": 18, "paired": 18}:
        hard_stop("forced_upper_bound")

    blind_source = json.loads((args.parent_run / "blind_set_seal_audit.json").read_text())
    if (blind_source.get("passed") is not True or blind_source.get("edited_checkpoint_loaded") is not False
            or blind_source.get("outcomes_enumerated") is not False):
        hard_stop("blind_seal")
    anchor_source = json.loads((args.parent_run / "anchor_and_immutability_audit.json").read_text())
    if (anchor_source.get("status") != "PASS" or anchor_source.get("record953_edited_for_selection") is not False
            or anchor_source.get("blind_edited_checkpoint_loaded") is not False):
        hard_stop("prior_anchor")

    safety_inputs = {
        "validation_manifest": {"path": str(validation_path), "sha256": file_hash(validation_path)},
        "family_components": {"path": str(args.parent_run / "eqkey_family_audit/family_components.jsonl"),
                              "sha256": file_hash(args.parent_run / "eqkey_family_audit/family_components.jsonl")},
        "fixed_experts_manifest": {"path": str(fixed_path), "sha256": file_hash(fixed_path)},
    }
    audit = {
        "protocol": PROTOCOL, "status": "PASS", "formal_run": str(args.formal_run),
        "parent_run": str(args.parent_run), "canonical_bank_hash": EXPECTED_BANK,
        "frozen_module_hash": EXPECTED_FROZEN, "base_model_hash": EXPECTED_BASE,
        "validation_family_count": 32, "validation_family_order": family_order,
        "forced_on_upper_bound": forced, "checkpoints": checkpoint_rows,
        "main_results": main_rows, "safety_inputs": safety_inputs,
        "record953_excluded": True, "sealed_blind_unopened": True,
        "heldout_loaded": False, "optimizer_loaded": False,
    }
    write_new(args.out_root / "anchor_and_immutability_audit.json", audit)
    write_new(args.out_root / "blind_set_seal_audit.json", {
        "protocol": PROTOCOL, "passed": True,
        "selection_manifest_hash": blind_source["selection_manifest_hash"],
        "sealed_manifest_hash": blind_source["sealed_manifest_hash"],
        "edited_checkpoint_loaded": False, "outcomes_enumerated": False,
        "source_audit_sha256": file_hash(args.parent_run / "blind_set_seal_audit.json"),
    })
    with (args.out_root / "state_and_bank_hash_ledger.jsonl").open("x") as handle:
        for row in checkpoint_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_new(args.out_root / "run_manifest.json", {
        "protocol": PROTOCOL, "status": "ANCHORS_VERIFIED__SAFETY_CLOSURE_PENDING",
        "formal_run": str(args.formal_run), "record953_used": False,
        "heldout_loaded": False, "sealed_blind_loaded": False,
        "training_performed": False, "checkpoint_selection_performed_by_oracle": False,
        "router_r2_implemented": False, "stage2_permitted": False,
    })
    print(json.dumps({"status": "ROUTER_R1_ORACLE_ANCHORS_PASS",
                      "out_root": str(args.out_root), "checkpoint_count": len(checkpoint_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
