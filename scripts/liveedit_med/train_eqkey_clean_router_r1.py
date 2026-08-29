#!/usr/bin/env python3
"""Train structure-preserving Router-R1 on EqKey-clean family caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.router_r1 import (CHECKPOINT_STEPS, EXPECTED_BANK_HASH,
    assert_router_only, canonical_hash, configure_router_only, frozen_module_state,
    negative_category, negative_text_absolute_loss, negative_visual_loss,
    positive_text_losses, positive_visual_loss, repository_size, router_state,
    semantic_category)
from methods.liveedit_med.serialization import load_safe_state, save_safe_state, tensor_hashes
from methods.liveedit_med.source_training_continuation import SourceTrainingContinuationMode
from methods.liveedit_med.trace_parity import state_dict_sha256
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, state_weight_hash
from scripts.engram.run_llavamed_record953_lora_positive_control import seed_everything
from scripts.liveedit_med.router_r1_schema import resolve_locality_source
from scripts.liveedit_med.train_liveedit_med_v4_source import (
    grouped_residual_losses,
    load_training_model,
    spans,
)
from scripts.liveedit_med.train_router_r1 import recursive_hash


PROTOCOL = "LIVEEDIT_MED_EQKEY_CLEAN_ROUTER_R1_V1"
STRICT_MODE = SourceTrainingContinuationMode.STRICT_SOURCE_REAPPLY_LAYER21


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_repository(target: str, size: int, all_ids: list[str]) -> list[str]:
    ordered = sorted((value for value in all_ids if value != target),
                     key=lambda value: (canonical_hash([target, value]), value))
    result = [target, *ordered[:size - 1]]
    if len(result) != size or len(set(result)) != size:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_INVALID_REPOSITORY")
    return result


def role_view(family: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    rows = [row for row in family["canonical_views"] if row["role"] == role]
    if not rows:
        raise RuntimeError(f"EQKEY_CLEAN_ROUTER_R1_MISSING_ROLE:{family['family_id']}:{role}")
    return min(rows, key=lambda row: (row["selection_hash"], row["eqkey"]))


def training_variant(tensors: Mapping[str, torch.Tensor], name: str, device: torch.device):
    """Map frozen cache names to the source-training row contract."""
    prefix = name + "__"
    raw = {key[len(prefix):]: value.to(device)
           for key, value in tensors.items() if key.startswith(prefix)}
    if not raw:
        raise RuntimeError(f"EQKEY_CLEAN_ROUTER_R1_MISSING_VARIANT:{name}")
    return {"hidden": raw["hidden"].float(), "labels": raw["labels"],
            "attention": raw["attention"], "vision": raw["vision_mask"].bool(),
            "prompt": raw["question_mask"].bool(), "answer": raw["answer_mask"].bool(),
            "base_answer_logits": raw["base_answer_logits"].float()}


def load_view(view: Mapping[str, Any], device: torch.device):
    return training_variant(load_file(view["cache_file_path"], device="cpu"), view["role"], device)


def load_locality(family: Mapping[str, Any], device: torch.device):
    source = resolve_locality_source(family)
    return training_variant(load_file(str(source.cache_file_path), device="cpu"),
                            "image_locality", device)


def query_keys(modules, row: Mapping[str, torch.Tensor]):
    """Apply the frozen Router-R1 query equations to source-training rows."""
    vision, question, _answer = spans(row)
    return (modules.input_extractor.extract_query(question),
            modules.input_extractor.extract_vision(question, vision),
            modules.input_extractor.extract_from_visprot(question),
            modules.edit_extractor.extract_query(question))


def fixed_experts(path: Path) -> tuple[dict[str, dict[str, torch.Tensor]], Mapping[str, Any]]:
    manifest = json.loads((path / "manifest.json").read_text())
    result = {}
    for row in manifest["records"]:
        result[row["family_id"]] = {key: value.float().cpu().contiguous()
            for key, value in load_file(row["file_path"], device="cpu").items()}
        if sha256_file(Path(row["file_path"])) != row["file_sha256"]:
            raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_EXPERT_HASH_MISMATCH")
    return result, manifest


def save_checkpoint(run_dir: Path, step: int, epoch: int, modules, optimizer,
                    frozen_hash: str, expert_hash: str, base_hash: str,
                    generator_step: int) -> dict[str, Any]:
    assert_router_only(modules)
    if state_dict_sha256(frozen_module_state(modules)) != frozen_hash:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_FROZEN_MODULE_MUTATION")
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_BANK_MUTATION")
    directory = run_dir / f"checkpoint_{step:04d}"
    manifest = save_safe_state(directory, modules.state_dict(), {
        "protocol": PROTOCOL, "step": step, "epoch": epoch, "router_only": True,
        "selected_generator_checkpoint_step": generator_step,
        "trainable_tensor_hashes": tensor_hashes(router_state(modules)),
        "frozen_module_hash": frozen_hash, "frozen_expert_hash": expert_hash,
        "base_model_hash": base_hash, "canonical_bank_hash": EXPECTED_BANK_HASH,
        "optimizer_state_hash": recursive_hash(optimizer.state_dict()),
        "source_training_continuation": STRICT_MODE.value,
        "inference_continuation": "official_layer21_output_hook_then_layer22",
    })
    return {"step": step, "epoch": epoch, "directory": directory.name,
        "manifest_sha256": sha256_file(directory / "manifest.json"),
        "tensor_hashes": manifest["tensor_hashes"],
        "optimizer_state_hash": manifest["optimizer_state_hash"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--fixed-experts", type=Path, required=True)
    parser.add_argument("--hard-cache-manifest", type=Path, required=True)
    parser.add_argument("--strict-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--suffix-physical-gpu", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=640)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != f"{args.physical_gpu},{args.suffix_physical_gpu}":
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_TWO_GPU_VISIBILITY_MISMATCH")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything()
    train_manifest = json.loads(args.train_manifest.read_text())
    families = train_manifest["families"]
    family_by_id = {row["family_id"]: row for row in families}
    family_ids = [row["family_id"] for row in families]
    if len(family_ids) < 32 or len(set(family_ids)) != len(family_ids):
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_TRAIN_FAMILY_SET")
    expert_bank, expert_manifest = fixed_experts(args.fixed_experts)
    if set(family_ids) - set(expert_bank):
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_MISSING_FIXED_EXPERT")
    hard_manifest = json.loads(args.hard_cache_manifest.read_text())
    hard = {row["family_id"]: row for row in hard_manifest["records"]}
    if set(family_ids) != set(hard):
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_HARD_CACHE_SET")
    model, _bank, suffix_device = load_training_model(args.physical_gpu,
        args.suffix_physical_gpu, STRICT_MODE)
    base_hash = state_weight_hash(model)
    for parameter in model.llava_model.parameters():
        parameter.requires_grad_(False)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig(
        learning_rate=5e-5, source_training_continuation_mode=STRICT_MODE)).to(model.lm_device).float()
    initial_state, strict_manifest = load_safe_state(args.strict_checkpoint)
    generator_step = int(strict_manifest["step"])
    if generator_step != int(expert_manifest["generator_checkpoint_step"]):
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_GENERATOR_EXPERT_MISMATCH")
    modules.load_state_dict(initial_state, strict=True)
    trainable_names = configure_router_only(modules); modules.train()
    frozen_hash = state_dict_sha256(frozen_module_state(modules))
    optimizer = torch.optim.Adam([p for p in modules.parameters() if p.requires_grad],
                                 lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=0)
    if optimizer.state:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_OLD_OPTIMIZER_STATE")
    (args.run_dir.parent / "trainable_parameter_audit.json").write_text(json.dumps({
        "protocol": PROTOCOL, "trainable_parameters": list(trainable_names),
        "trainable_parameter_count": sum(p.numel() for p in modules.parameters() if p.requires_grad),
        "only_edit_and_input_extractors_trainable": True, "fresh_optimizer_empty_state": True,
        "frozen_module_hash": frozen_hash, "frozen_expert_hash": expert_manifest["global_expert_hash"],
    }, indent=2, sort_keys=True) + "\n")
    (args.run_dir / "router_r1_config.yaml").write_text(
        "protocol: LIVEEDIT_MED_EQKEY_CLEAN_ROUTER_R1_V1\noptimizer: Adam\n"
        "learning_rate: 5.0e-5\nweight_decay: 0\nbatch_size: 8\noptimizer_steps: 640\n"
        "seed: 42\nrepository_cycle: [1, 4, 8, 16, 32]\n"
        "checkpoint_steps: [80, 160, 240, 320, 400, 480, 560, 640]\n"
        "loss_weights: {visual_hard: 1, text_absolute: 1, text_relative: 1, positive_nll: 1, negative_kl: 1}\n")
    trajectory = args.run_dir / "training_trajectory.jsonl"; trajectory.write_text("")
    frozen_ledger = args.run_dir / "frozen_hash_ledger.jsonl"; frozen_ledger.write_text("")
    rng = np.random.default_rng(42); step = 0; epoch = 0; checkpoints = []
    while step < args.max_steps:
        epoch += 1
        order = rng.permutation(len(family_ids))
        for begin in range(0, len(order), 8):
            if step >= args.max_steps:
                break
            step += 1
            ids = [family_ids[int(index)] for index in order[begin:begin + 8]]
            size = repository_size(step); semantic = semantic_category(step); negative = negative_category(step)
            optimizer.zero_grad(set_to_none=True)
            all_eqr=[]; all_evr=[]; all_c=[]; all_r=[]; target_rows=[]
            route_losses={"hard":[], "absolute":[], "relative":[]}
            for family_id in ids:
                members = stable_repository(family_id, size, family_ids)
                repo_begin = len(all_eqr)
                for member_id in members:
                    fixed = expert_bank[member_id]
                    all_eqr.append(fixed["eqr"].to(model.lm_device)); all_evr.append(fixed["evr"].to(model.lm_device))
                    all_c.append(fixed["moe_c"].to(model.lm_device)); all_r.append(fixed["moe_r"].to(model.lm_device))
                family = family_by_id[family_id]
                native_row = load_view(role_view(family, "native"), model.lm_device)
                semantic_row = load_view(role_view(family, semantic), model.lm_device)
                negative_row = training_variant(load_file(hard[family_id]["file_path"], device="cpu"),
                                                negative, model.lm_device)
                locality_row = load_locality(family, model.lm_device)
                target_rows.append((native_row, semantic_row, negative_row, locality_row,
                                    repo_begin, repo_begin + size))
            eqrs=torch.cat(all_eqr); evrs=torch.cat(all_evr); moe_cs=torch.cat(all_c); moe_rs=torch.cat(all_r)
            masks=[]; native_rows=[]; semantic_rows=[]; negative_rows=[]; locality_rows=[]
            for native_row, semantic_row, negative_row, locality_row, repo_begin, repo_end in target_rows:
                mask=torch.zeros(len(all_eqr),dtype=torch.bool,device=model.lm_device); mask[repo_begin:repo_end]=True
                masks.extend([mask,mask,mask,mask]); native_rows.append(native_row); semantic_rows.append(semantic_row)
                negative_rows.append(negative_row); locality_rows.append(locality_row)
                repo_eqr,repo_evr=eqrs[repo_begin:repo_end],evrs[repo_begin:repo_end]
                for row in (native_row,semantic_row):
                    iqr,ivr,sentinel,_=query_keys(modules,row)
                    loc_edit=modules.edit_extractor.extract_query(spans(locality_row)[1])
                    route_losses["hard"].append(positive_visual_loss(ivr,repo_evr,sentinel,0))
                    absolute,relative=positive_text_losses(iqr,repo_eqr,0,loc_edit)
                    route_losses["absolute"].append(absolute); route_losses["relative"].append(relative)
                for row in (negative_row,locality_row):
                    iqr,ivr,sentinel,_=query_keys(modules,row)
                    route_losses["hard"].append(negative_visual_loss(ivr,repo_evr,sentinel))
                    route_losses["absolute"].append(negative_text_absolute_loss(iqr,repo_eqr))
            output=grouped_residual_losses(modules,model,[
                ("native",native_rows,masks[0::4],False),("semantic",semantic_rows,masks[1::4],False),
                ("hard_negative",negative_rows,masks[2::4],True),("locality",locality_rows,masks[3::4],True),
            ],moe_cs,moe_rs,eqrs,suffix_device,STRICT_MODE)
            losses={"hard":torch.stack(route_losses["hard"]).mean(),
                "soft_absolute":torch.stack(route_losses["absolute"]).mean(),
                "soft_relative":torch.stack(route_losses["relative"]).mean(),
                "positive":output["native"]+output["semantic"],
                "negative":output["hard_negative"]+output["locality"]}
            total=sum(losses.values())
            if not torch.isfinite(total) or any(not torch.isfinite(value) for value in losses.values()):
                raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_NONFINITE_LOSS")
            total.backward(); assert_router_only(modules)
            grad_norm=torch.nn.utils.clip_grad_norm_([p for p in modules.parameters() if p.requires_grad],1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_NONFINITE_GRADIENT")
            optimizer.step()
            row={"protocol":PROTOCOL,"epoch":epoch,"step":step,"family_ids":ids,
                "repository_size":size,"semantic_category":semantic,"negative_category":negative,
                "total_loss":float(total.detach()),**{f"loss_{k}":float(v.detach()) for k,v in losses.items()},
                "gradient_norm":float(grad_norm),"nan_or_inf":False,"learning_rate":5e-5}
            with trajectory.open("a") as handle: handle.write(json.dumps(row,sort_keys=True)+"\n")
            if step%20==0: print(json.dumps({k:row[k] for k in ("epoch","step","repository_size","total_loss",
                "loss_hard","loss_soft_absolute","loss_soft_relative","loss_positive","loss_negative")},sort_keys=True),flush=True)
            if step in CHECKPOINT_STEPS:
                checkpoints.append(save_checkpoint(args.run_dir,step,epoch,modules,optimizer,frozen_hash,
                    expert_manifest["global_expert_hash"],base_hash,generator_step))
                with frozen_ledger.open("a") as handle: handle.write(json.dumps({"step":step,
                    "frozen_module_hash":frozen_hash,"frozen_expert_hash":expert_manifest["global_expert_hash"],
                    "base_model_hash":base_hash,"canonical_bank_hash":EXPECTED_BANK_HASH,"unchanged":True},sort_keys=True)+"\n")
    if args.max_steps==640 and tuple(row["step"] for row in checkpoints)!=CHECKPOINT_STEPS:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_INCOMPLETE_CHECKPOINTS")
    if state_dict_sha256(frozen_module_state(modules))!=frozen_hash or state_weight_hash(model)!=base_hash:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_FROZEN_STATE_MUTATION")
    (args.run_dir.parent/"checkpoint_manifest.json").write_text(json.dumps({"protocol":PROTOCOL,
        "complete":step==640,"optimizer_steps":step,"checkpoints":checkpoints,
        "selected_generator_checkpoint_step":generator_step,"frozen_module_hash":frozen_hash,
        "frozen_expert_hash":expert_manifest["global_expert_hash"],"base_model_hash":base_hash,
        "canonical_bank_hash":bank_manifest()["sha256"]},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"EQKEY_CLEAN_ROUTER_R1_TRAINING_COMPLETE" if step==640 else "EQKEY_CLEAN_ROUTER_R1_SMOKE_COMPLETE",
                      "steps":step,"checkpoints":len(checkpoints)},sort_keys=True))


if __name__ == "__main__":
    main()
