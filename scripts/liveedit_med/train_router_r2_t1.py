#!/usr/bin/env python3
"""Train the single permitted Router-R2 T1 K+1 NO_EDIT variant.

The model/generator and immutable experts are not loaded for optimization.
Only the Router-R1 input/edit extractors plus one zero-initialized NO_EDIT text
prototype are trainable.  The unchanged visual-hard branch is retained at its
R1 weight 1; K+1 cross entropy replaces the R1 text absolute/relative losses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.router_r1 import (CHECKPOINT_STEPS, EXPECTED_BANK_HASH,
    assert_router_only, configure_router_only, frozen_module_state,
    negative_visual_loss, positive_visual_loss, repository_size)
from methods.liveedit_med.router_r2 import (PROTOCOL, kplus1_cross_entropy,
    kplus1_label, kplus1_text_logits)
from methods.liveedit_med.serialization import load_safe_state, save_safe_state, tensor_hashes
from methods.liveedit_med.trace_parity import state_dict_sha256
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_llavamed_record953_lora_positive_control import seed_everything
from scripts.liveedit_med.train_eqkey_clean_router_r1 import (
    fixed_experts, role_view, stable_repository, training_variant)
from scripts.liveedit_med.train_liveedit_med_v4_source import spans


ROLES = ("native", "textual", "visual", "paired")
CATEGORIES = ("same_image_different_question", "same_question_different_image",
              "visual_nearest", "text_nearest", "joint_near_miss")
EXPECTED_FROZEN = "d1d5ce232ad2aeb7a29c4c5586a6af5dcdee064321cfc94c3393d576b1bc2249"
EXPECTED_BASE = "d8b7032a563e32f22fd51eb65d92bbb0177c913d19c5c1e6ce6ad73d0e5ca75d"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def query(modules, row):
    vision, question, _answer = spans(row)
    return (modules.input_extractor.extract_query(question.float()),
            modules.input_extractor.extract_vision(question.float(), vision.float()),
            modules.input_extractor.extract_from_visprot(question.float()))


def checkpoint(run_dir: Path, step: int, epoch: int, modules, no_edit_key,
               optimizer, frozen_hash: str, expert_hash: str):
    assert_router_only(modules)
    if state_dict_sha256(frozen_module_state(modules)) != frozen_hash:
        raise RuntimeError("ROUTER_R2_T1_FROZEN_MODULE_MUTATION")
    state = {f"modules.{name}": value for name, value in modules.state_dict().items()}
    state["no_edit_text_key"] = no_edit_key.detach()
    directory = run_dir / f"checkpoint_{step:04d}"
    manifest = save_safe_state(directory, state, {
        "protocol": PROTOCOL, "candidate": "R2_T1_EXPLICIT_NOEDIT_KPLUS1",
        "step": step, "epoch": epoch, "router_only": True,
        "initialization": "Router-R1 pre-training step-0 state = strict-source generator checkpoint 3000",
        "no_edit_initialization": "all_zeros", "no_edit_shape": list(no_edit_key.shape),
        "trainable_module_names": ["input_extractor", "edit_extractor", "no_edit_text_key"],
        "objective": "balanced_positive_negative_standard_K_plus_1_cross_entropy_plus_visual_hard",
        "retained_r1_losses": {"visual_hard": 1.0},
        "removed_r1_losses": ["text_absolute", "text_relative", "positive_nll", "negative_kl"],
        "forbidden_losses_absent": ["energy", "contrastive", "angular_margin", "cosine_normalization", "full_strength_execution"],
        "frozen_module_hash": frozen_hash, "frozen_expert_hash": expert_hash,
        "base_model_hash": EXPECTED_BASE, "canonical_bank_hash": EXPECTED_BANK_HASH,
        "tensor_hashes_at_save": tensor_hashes(state),
    })
    return {"step": step, "directory": directory.name,
            "manifest_sha256": __import__("hashlib").sha256((directory / "manifest.json").read_bytes()).hexdigest(),
            "tensor_hashes": manifest["tensor_hashes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--fixed-experts", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("ROUTER_R2_T1_GPU_VISIBILITY_MISMATCH")
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True)
    seed_everything(); device = torch.device("cuda")
    train = json.loads(args.train_manifest.read_text())
    split = json.loads(args.calibration_manifest.read_text())
    fit_ids = {row["family_id"] for row in split["calibration_fit"]}
    families = [row for row in train["families"] if row["family_id"] in fit_ids]
    all_ids = [row["family_id"] for row in families]
    family_by_id = {row["family_id"]: row for row in families}
    hard = {row["family_id"]: row for row in json.loads(args.hard_manifest.read_text())["records"]}
    if len(all_ids) != 374 or set(all_ids) != fit_ids or set(all_ids) - set(hard):
        raise RuntimeError("ROUTER_R2_T1_TRAIN_SET")
    experts, expert_manifest = fixed_experts(args.fixed_experts)
    # The immutable expert manifest is a deliberate superset: it also contains
    # the 64 purged evaluation experts.  T1 may load that frozen manifest but
    # must train on, and construct repositories from, the 467 calibration
    # train families only.
    if set(all_ids) - set(experts):
        raise RuntimeError("ROUTER_R2_T1_EXPERT_SET")
    modules = LiveEditMedicalModules(LiveEditMedicalConfig(learning_rate=5e-5)).to(device).float()
    initial, initial_manifest = load_safe_state(args.initial_checkpoint)
    if int(initial_manifest["step"]) != 3000:
        raise RuntimeError("ROUTER_R2_T1_NOT_R1_STEP0_STATE")
    modules.load_state_dict(initial, strict=True)
    trainable_names = configure_router_only(modules)
    frozen_hash = state_dict_sha256(frozen_module_state(modules))
    if frozen_hash != EXPECTED_FROZEN:
        raise RuntimeError("ROUTER_R2_STOP__FROZEN_HASH_MISMATCH")
    no_edit_key = torch.nn.Parameter(torch.zeros(1, 4, 1024, device=device))
    parameters = [value for value in modules.parameters() if value.requires_grad] + [no_edit_key]
    optimizer = torch.optim.Adam(parameters, lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=0)
    write_json(args.run_dir.parent / "TRAINING_MANIFEST.json", {
        "protocol": PROTOCOL, "candidate": "R2_T1_EXPLICIT_NOEDIT_KPLUS1",
        "initial_checkpoint": str(args.initial_checkpoint), "initial_step": 3000,
        "r1_semantic_initialization": "pre-training step-0",
        "steps": 640, "checkpoints": list(CHECKPOINT_STEPS), "seed": 42,
        "training_partition": "calibration_fit_only",
        "training_family_count": len(all_ids),
        "calibration_manifest_sha256": hashlib.sha256(
            args.calibration_manifest.read_bytes()).hexdigest(),
        "calibration_lock_family_count": len(split["calibration_lock"]),
        "calibration_lock_loaded_for_training": False,
        "batch_size_families": 8, "positive_roles_per_family": list(ROLES),
        "negative_subtypes_per_family": list(CATEGORIES),
        "balanced_ce": "0.5*mean(positive_CE)+0.5*mean(negative_CE)",
        "retained_r1_losses": {"visual_hard": 1.0},
        "trainable_module_parameter_names": list(trainable_names),
        "new_trainable_parameter": "no_edit_text_key", "new_parameter_shape": [1, 4, 1024],
        "frozen_module_hash": frozen_hash,
        "frozen_expert_hash": expert_manifest["global_expert_hash"],
        "base_model_hash": EXPECTED_BASE, "canonical_bank_hash": EXPECTED_BANK_HASH,
        "validation_loaded": False, "heldout_loaded": False, "record953_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False})
    trajectory = args.run_dir / "training_trajectory.jsonl"; trajectory.touch(exist_ok=False)
    rng = np.random.default_rng(42); step = epoch = 0; checkpoints = []
    while step < 640:
        epoch += 1
        for begin in range(0, len(all_ids), 8):
            if step >= 640: break
            if begin == 0:
                order = rng.permutation(len(all_ids))
            step += 1; size = repository_size(step)
            batch_ids = [all_ids[int(index)] for index in order[begin:begin + 8]]
            positive_ce=[]; negative_ce=[]; positive_hard=[]; negative_hard=[]
            optimizer.zero_grad(set_to_none=True)
            for fid in batch_ids:
                members = stable_repository(fid, size, all_ids)
                eqr = torch.cat([experts[mid]["eqr"].to(device) for mid in members])
                evr = torch.cat([experts[mid]["evr"].to(device) for mid in members])
                family = family_by_id[fid]
                for role in ROLES:
                    view = role_view(family, role)
                    row = training_variant(load_file(view["cache_file_path"], device="cpu"), role, device)
                    text, visual, sentinel = query(modules, row)
                    logits = kplus1_text_logits(text, eqr, no_edit_key)
                    positive_ce.append(kplus1_cross_entropy(logits, kplus1_label(
                        repository_size=size, target_repository_index=0, no_edit=False)))
                    positive_hard.append(positive_visual_loss(visual, evr, sentinel, 0))
                hard_tensors = load_file(hard[fid]["file_path"], device="cpu")
                for category in CATEGORIES:
                    row = training_variant(hard_tensors, category, device)
                    text, visual, sentinel = query(modules, row)
                    logits = kplus1_text_logits(text, eqr, no_edit_key)
                    negative_ce.append(kplus1_cross_entropy(logits, kplus1_label(
                        repository_size=size, target_repository_index=None, no_edit=True)))
                    negative_hard.append(negative_visual_loss(visual, evr, sentinel))
            loss_positive_ce = torch.stack(positive_ce).mean()
            loss_negative_ce = torch.stack(negative_ce).mean()
            loss_visual_positive = torch.stack(positive_hard).mean()
            loss_visual_negative = torch.stack(negative_hard).mean()
            loss_kplus1 = .5 * (loss_positive_ce + loss_negative_ce)
            loss_visual = .5 * (loss_visual_positive + loss_visual_negative)
            total = loss_kplus1 + loss_visual
            if not torch.isfinite(total):
                raise RuntimeError("ROUTER_R2_T1_NONFINITE_LOSS")
            total.backward(); assert_router_only(modules)
            grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad):
                raise RuntimeError("ROUTER_R2_T1_NONFINITE_GRADIENT")
            optimizer.step()
            row = {"protocol": PROTOCOL, "step": step, "epoch": epoch,
                "repository_size": size, "family_ids": batch_ids,
                "loss_total": float(total.detach()), "loss_kplus1": float(loss_kplus1.detach()),
                "loss_positive_ce": float(loss_positive_ce.detach()),
                "loss_negative_ce": float(loss_negative_ce.detach()),
                "loss_visual_hard": float(loss_visual.detach()),
                "gradient_norm": float(grad), "nan_or_inf": False}
            with trajectory.open("a") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")
            if step % 20 == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            if step in CHECKPOINT_STEPS:
                checkpoints.append(checkpoint(args.run_dir, step, epoch, modules, no_edit_key,
                    optimizer, frozen_hash, expert_manifest["global_expert_hash"]))
    write_json(args.run_dir.parent / "CHECKPOINT_SET.json", {
        "protocol": PROTOCOL, "complete": True, "steps": 640,
        "checkpoints": checkpoints, "validation_loaded": False,
        "heldout_loaded": False, "record953_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False})
    print(json.dumps({"status": "ROUTER_R2_T1_TRAINING_COMPLETE",
                      "steps": step, "checkpoint_count": len(checkpoints)}, sort_keys=True))


if __name__ == "__main__":
    main()
