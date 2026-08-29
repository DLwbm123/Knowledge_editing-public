#!/usr/bin/env python3
"""Online parity for hash-referenced cache rows in the EqKey-clean pilot."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.eqkey_clean_fast import PROTOCOL
from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, eos_ids
from scripts.engram.run_record953_routed_banked_lora_v1_1 import visible_input_audit
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_cached_greedy_trace
from scripts.liveedit_med.cache_router_r1 import capture, sha256_file
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import (MAX_NEW_TOKENS,
    compact_trace, load_clean_model)
from methods.liveedit_med.posthoc_validation import sample_to_model_row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample_family(families: list[dict[str, Any]], predicate) -> dict[str, Any]:
    return min((row for row in families if predicate(row)),
               key=lambda row: (row["allocation_hash"], row["family_id"]))


def cached_variant(tensors: Mapping[str, torch.Tensor], role: str) -> dict[str, torch.Tensor]:
    prefix = role + "__"
    return {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-components", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_GPU_VISIBILITY_MISMATCH")
    families = read_jsonl(args.family_components)
    validation = json.loads(args.validation_manifest.read_text())["families"]
    heldout = json.loads(args.heldout_manifest.read_text())["families"]
    chosen = {
        "clean_train": sample_family(families, lambda row: row["touches_original_train"]),
        "purged_validation": min(validation, key=lambda row: (row["allocation_hash"], row["family_id"])),
        "purged_heldout": min(heldout, key=lambda row: (row["allocation_hash"], row["family_id"])),
    }
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:bank")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    if int(checkpoint_manifest["step"]) != 3200:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:cache_checkpoint")
    modules.load_state_dict(state, strict=True); modules.eval()

    results = []
    for split, family in chosen.items():
        family_rows = []
        for view in family["canonical_views"]:
            sample = {"image": view["image_path"], "prompt": view["question"], "target": view["target"]}
            tensors = load_file(view["cache_file_path"], device="cpu")
            cached = cached_variant(tensors, view["role"])
            online = capture(model, block, sample)
            tensor_equal = {name: bool(torch.equal(cached[name], value)) for name, value in online.items()}
            visible = visible_input_audit(model, {"question": sample["prompt"], "image_path": sample["image"],
                "image_sha256": sha256_file(Path(sample["image"]))})
            canonical = build_canonical_inputs(model, sample_to_model_row(sample))
            trace = compact_trace(manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS,
                                                              eos_ids(model), top_k=1))
            generation_equal = (trace["token_ids"] == view["clean_generation"]["token_ids"]
                                and trace["stop_reason"] == view["clean_generation"]["stop_reason"])
            family_rows.append({
                "eqkey": view["eqkey"], "role": view["role"], "tensor_equal": tensor_equal,
                "all_tensors_equal": all(tensor_equal.values()),
                "eqkey_equal": visible["router_input_equivalence_key"] == view["eqkey"],
                "generation_equal": generation_equal,
            })
        native = next(row for row in family["canonical_views"]
                      if row["eqkey"] == family["canonical_native_eqkey"])
        native_tensors = load_file(native["cache_file_path"], device="cpu")
        native_online = capture(model, block, {"image": native["image_path"], "prompt": native["question"],
                                               "target": native["target"]})
        hidden = native_online["hidden"].float().unsqueeze(0).to(model.lm_device)
        vision = hidden[:, native_online["vision_mask"].to(model.lm_device).bool()]
        question = hidden[:, native_online["question_mask"].to(model.lm_device).bool()]
        answer = hidden[:, native_online["answer_mask"].to(model.lm_device).bool()]
        _eqr, _evr, moe_c, moe_r = modules.generated_edit(vision, question, answer)
        expert_equal = {
            "moe_c": bool(torch.equal(moe_c.cpu(), native_tensors["expert__moe_c"])),
            "moe_r": bool(torch.equal(moe_r.cpu(), native_tensors["expert__moe_r"])),
        }
        results.append({"split": split, "family_id": family["family_id"], "views": family_rows,
                        "expert_equal": expert_equal, "all_expert_tensors_equal": all(expert_equal.values())})
    passed = all(row["all_expert_tensors_equal"] and all(
        view["all_tensors_equal"] and view["eqkey_equal"] and view["generation_equal"]
        for view in row["views"]) for row in results)
    payload = {"protocol": PROTOCOL, "status": "PASS" if passed else "EQKEY_FAST_LANE_CACHE_PARITY_FAILURE",
               "checkpoint_step": 3200, "samples": results, "old_cache_mutated": False,
               "record953_loaded": False, "sealed_blind_loaded": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "sample_splits": list(chosen)}, sort_keys=True))
    if not passed:
        raise RuntimeError("EQKEY_FAST_LANE_CACHE_PARITY_FAILURE")


if __name__ == "__main__":
    main()
