"""Load immutable generated experts for EqKey-clean Router-R1 evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file


PROTOCOL = "LIVEEDIT_MED_EQKEY_CLEAN_ROUTER_R1_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_experts(directory: Path, families: Sequence[Mapping[str, Any]],
                       device: torch.device, checkpoint_manifest: Mapping[str, Any]):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != PROTOCOL or not manifest.get("all_generated_expert_tensors_frozen"):
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_FIXED_EXPERT_PROTOCOL")
    generator_step = int(checkpoint_manifest.get("selected_generator_checkpoint_step",
                                                  checkpoint_manifest["step"]))
    if int(manifest["generator_checkpoint_step"]) != generator_step:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_FIXED_EXPERT_CHECKPOINT_DRIFT")
    by_id = {row["family_id"]: row for row in manifest["records"]}
    result = {}
    for family in families:
        family_id = family["family_id"]
        if family_id not in by_id:
            raise RuntimeError(f"EQKEY_CLEAN_ROUTER_R1_FIXED_EXPERT_MISSING:{family_id}")
        row = by_id[family_id]; path = Path(row["file_path"])
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_FIXED_EXPERT_HASH")
        tensors = load_file(str(path), device="cpu")
        result[family_id] = {key: tensors[key].float().to(device)
                             for key in ("eqr", "evr", "moe_c", "moe_r")}
    return result, manifest, sha256_file(manifest_path)
