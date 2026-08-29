#!/usr/bin/env python3
"""Freeze generated expert tensors before EqKey-clean Router-R1 training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH
from methods.liveedit_med.serialization import load_safe_state, tensor_hashes
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.liveedit_med.cache_router_r1 import sha256_file
from scripts.liveedit_med.evaluate_eqkey_clean_checkpoint import expert_for_family


PROTOCOL = "LIVEEDIT_MED_EQKEY_CLEAN_ROUTER_R1_V1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    train = json.loads(args.train_manifest.read_text())
    validation = json.loads(args.validation_manifest.read_text())
    heldout = json.loads(args.heldout_manifest.read_text())
    families = []
    split_by_id = {}
    for split, source in (("train", train), ("validation", validation), ("heldout", heldout)):
        for family in source["families"]:
            family_id = family["family_id"]
            if family_id in split_by_id:
                raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_EXPERT_SPLIT_COLLISION")
            split_by_id[family_id] = split
            families.append(family)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    modules.load_state_dict(state, strict=True); modules.eval()
    args.out.mkdir(parents=True)
    records = []
    for complete, family in enumerate(families, 1):
        family_id = family["family_id"]
        expert = expert_for_family(modules, family, torch.device("cpu"))
        tensors = {name: expert[name].detach().float().cpu().contiguous()
                   for name in ("eqr", "evr", "moe_c", "moe_r")}
        path = args.out / f"family_{family_id}.safetensors"
        save_file(tensors, str(path))
        loaded = load_file(str(path), device="cpu")
        if loaded.keys() != tensors.keys() or any(not torch.equal(loaded[key], tensors[key]) for key in tensors):
            raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_EXPERT_RELOAD_FAILURE")
        records.append({"family_id": family_id, "split": split_by_id[family_id],
            "file_path": str(path.resolve()), "file_sha256": sha256_file(path),
            "tensor_hashes": tensor_hashes(tensors), "source_cache_sha256": expert["source_cache_sha256"]})
        if complete % 50 == 0 or complete == len(families):
            print(json.dumps({"event": "eqkey_r1_fixed_experts", "complete": complete,
                              "total": len(families)}), flush=True)
    manifest = {"protocol": PROTOCOL, "generator_checkpoint_step": int(checkpoint_manifest["step"]),
        "generator_checkpoint_manifest_sha256": sha256_file(args.checkpoint / "manifest.json"),
        "count": len(records), "split_counts": {split: sum(row["split"] == split for row in records)
            for split in ("train", "validation", "heldout")}, "records": records,
        "global_expert_hash": canonical_hash([{"family_id": row["family_id"],
            "tensor_hashes": row["tensor_hashes"]} for row in records]),
        "all_generated_expert_tensors_frozen": True, "canonical_bank_hash": EXPECTED_BANK_HASH,
        "record953_used": False, "sealed_blind_loaded": False}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
