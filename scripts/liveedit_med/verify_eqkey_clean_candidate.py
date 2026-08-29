#!/usr/bin/env python3
"""Fresh-process parity, replay, and rollback for an EqKey-clean candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.eqkey_clean_fast import PROTOCOL
from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import sample_to_model_row
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH
from methods.liveedit_med.serialization import load_safe_state, tensor_hashes
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, eos_ids
from scripts.engram.run_engram_v2_stage0abc_diagnostics import hf_cached_greedy_trace
from scripts.engram.stage0_generation_audit_utils import (build_canonical_inputs,
    manual_cached_greedy_trace, manual_greedy_trace)
from scripts.liveedit_med.evaluate_eqkey_clean_checkpoint import (expert_for_family, repository,
    repository_ids, sample)
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import load_clean_model
from scripts.liveedit_med.run_posthoc_stage_q import remove_hook, route_context
from scripts.liveedit_med.eqkey_fixed_expert_utils import load_fixed_experts


def compact(trace):
    return {key: trace.get(key) for key in ("raw_output", "token_ids", "stop_reason", "eos_step", "cap_hit")}


@torch.inference_mode()
def worker(args) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_GPU_VISIBILITY_MISMATCH")
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads(args.heldout_manifest.read_text())
    families = manifest["families"]
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:bank")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    modules.load_state_dict(state, strict=True); modules.eval()
    fixed_manifest_hash = None
    if args.fixed_experts is None:
        experts = {family["family_id"]: expert_for_family(modules, family, model.lm_device)
                   for family in families}
    else:
        experts, _fixed_manifest, fixed_manifest_hash = load_fixed_experts(
            args.fixed_experts, families, model.lm_device, checkpoint_manifest)
    family = min(families, key=lambda row: (row["allocation_hash"], row["family_id"]))
    family_id = family["family_id"]
    members = repository_ids(family_id, 32, [row["family_id"] for row in families])
    repo = repository(members, experts)
    native = next(row for row in family["canonical_views"]
                  if row["eqkey"] == family["canonical_native_eqkey"])
    current_sample = sample(native)
    clean_canonical = build_canonical_inputs(model, sample_to_model_row(current_sample))
    clean_before = manual_cached_greedy_trace(model, clean_canonical, 128, eos_ids(model), top_k=5)
    traces = {}; cached_route = None
    for mode in ("no_cache", "cached", "hf", "replay"):
        canonical, route, hook = route_context(model, block, modules, current_sample, repo)
        try:
            if mode == "no_cache":
                trace = manual_greedy_trace(model, canonical, 128, eos_ids(model), top_k=5)
            elif mode == "hf":
                trace = hf_cached_greedy_trace(model, canonical, 128)
            else:
                trace = manual_cached_greedy_trace(model, canonical, 128, eos_ids(model), top_k=5)
        finally:
            remove_hook(hook)
        traces[mode] = compact(trace)
        if mode == "cached":
            cached_route = route
    clean_after = manual_cached_greedy_trace(model, clean_canonical, 128, eos_ids(model), top_k=5)
    expert_hashes = tensor_hashes({
        "moe_c": experts[family_id]["moe_c"].cpu(), "moe_r": experts[family_id]["moe_r"].cpu(),
        "eqr": experts[family_id]["eqr"].cpu(), "evr": experts[family_id]["evr"].cpu(),
    })
    output = {
        "protocol": PROTOCOL, "process_index": args.process_index,
        "selected_step": int(checkpoint_manifest["step"]), "family_id": family_id,
        "route": cached_route, "traces": traces, "expert_tensor_hashes": expert_hashes,
        "manual_no_cache_cached_hf_parity": traces["no_cache"]["token_ids"] == traces["cached"]["token_ids"] == traces["hf"]["token_ids"],
        "replay": traces["cached"] == traces["replay"],
        "rollback": compact(clean_before) == compact(clean_after),
        "checkpoint_manifest_sha256": hashlib.sha256((args.checkpoint / "manifest.json").read_bytes()).hexdigest(),
        "canonical_bank_hash": bank_manifest()["sha256"], "record953_loaded": False,
        "sealed_blind_loaded": False,
        "fixed_experts_used": args.fixed_experts is not None,
        "fixed_expert_manifest_sha256": fixed_manifest_hash,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def finalize(args) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = [json.loads(path.read_text()) for path in args.process]
    if len(rows) != 2:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:repro_process_count")
    fresh = (rows[0]["route"] == rows[1]["route"] and rows[0]["traces"] == rows[1]["traces"]
             and rows[0]["expert_tensor_hashes"] == rows[1]["expert_tensor_hashes"]
             and rows[0]["checkpoint_manifest_sha256"] == rows[1]["checkpoint_manifest_sha256"])
    result = {
        "protocol": PROTOCOL,
        "manual_no_cache_cached_hf_parity": all(row["manual_no_cache_cached_hf_parity"] for row in rows),
        "reload": fresh, "fresh_process": fresh,
        "route_decision_parity": rows[0]["route"] == rows[1]["route"],
        "output_token_parity": rows[0]["traces"] == rows[1]["traces"],
        "generated_expert_hash_parity": rows[0]["expert_tensor_hashes"] == rows[1]["expert_tensor_hashes"],
        "replay": all(row["replay"] for row in rows), "rollback": all(row["rollback"] for row in rows),
        "canonical_bank_unchanged": all(row["canonical_bank_hash"] == EXPECTED_BANK_HASH for row in rows),
        "record953_loaded": False, "sealed_blind_loaded": False, "process_results": rows,
    }
    required = ("manual_no_cache_cached_hf_parity", "reload", "fresh_process", "route_decision_parity",
                "output_token_parity", "generated_expert_hash_parity", "replay", "rollback",
                "canonical_bank_unchanged")
    result["passed"] = all(result[name] for name in required)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS" if result["passed"] else "FAIL", "checks": list(required)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("worker")
    p.add_argument("--heldout-manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--process-index", type=int, required=True)
    p.add_argument("--fixed-experts", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("finalize")
    p.add_argument("--process", type=Path, action="append", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); (worker if args.mode == "worker" else finalize)(args)


if __name__ == "__main__":
    main()
