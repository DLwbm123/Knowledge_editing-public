#!/usr/bin/env python3
"""Evaluate frozen Router-R1 step 640 under Router-R2 E0 execution.

This program accepts only the precommitted clean-train calibration manifest.
It deliberately has no validation/heldout/record-953/blind command-line input.
Workers are isolated single-GPU processes and write append-only JSONL progress.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import normalize_answer, sample_to_model_row, unrestricted_match
from methods.liveedit_med.router_r2 import PROTOCOL, execution_plan, top1_route_statistics
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.source_ops import apply_low_rank_expert_residual, route_repository
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_natural_generation_recovery import clinical_preservation
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, eos_ids, state_weight_hash
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_cached_greedy_trace
from scripts.liveedit_med.eqkey_fixed_expert_utils import load_fixed_experts, sha256_file
from scripts.liveedit_med.evaluate_eqkey_clean_checkpoint import repository, sample
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import capture_prompt, compact_trace, load_clean_model
from scripts.liveedit_med.evaluate_router_r1_checkpoint import variant
from scripts.liveedit_med.evaluate_router_r1_oracle_ladder import generate as oracle_generate, parity_modes
from scripts.liveedit_med.train_eqkey_clean_router_r1 import role_view, stable_repository


ROLES = ("native", "textual", "visual", "paired")
CATEGORIES = ("same_image_different_question", "same_question_different_image",
              "visual_nearest", "text_nearest", "joint_near_miss")
EXPECTED_BANK = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"
EXPECTED_FROZEN = "d1d5ce232ad2aeb7a29c4c5586a6af5dcdee064321cfc94c3393d576b1bc2249"
EXPECTED_BASE = "d8b7032a563e32f22fd51eb65d92bbb0177c913d19c5c1e6ce6ad73d0e5ca75d"
MAX_NEW_TOKENS = 128


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def trace_result(trace: Mapping[str, Any], target: str) -> dict[str, Any]:
    result = compact_trace(trace)
    result["match"] = unrestricted_match(
        trace["raw_output"], target,
        eos=trace["stop_reason"] == "eos", cap_hit=trace["cap_hit"])
    return result


def route_audit(r1, r2, repo_ids: list[str]) -> dict[str, Any]:
    candidates = torch.where(r1.candidate_mask)[0].detach().cpu().tolist()
    return {
        "candidate_mask": [bool(value) for value in r1.candidate_mask.detach().cpu().tolist()],
        "candidate_ids": [repo_ids[index] for index in candidates],
        "candidate_count": r2.candidate_count,
        "visual_scores": r1.visual_scores.detach().float().cpu().tolist(),
        "sentinel_score": r1.sentinel_score.detach().float().cpu().tolist(),
        "text_scores": ([] if not hasattr(r1, "text_scores") else
                        r1.text_scores.detach().float().cpu().tolist()),
        "relative_weights": ([] if not hasattr(r1, "relative_weights") else
                             r1.relative_weights.detach().float().cpu().tolist()),
        "absolute_weights": ([] if not hasattr(r1, "absolute_weights") else
                             r1.absolute_weights.detach().float().cpu().tolist()),
        "original_final_weights": ([] if not hasattr(r1, "final_weights") else
                                   r1.final_weights.detach().float().cpu().tolist()),
        "selected_repository_index": r2.selected_repository_index,
        "selected_family_id": (None if r2.selected_repository_index is None else
                               repo_ids[r2.selected_repository_index]),
        "selected_original_coefficient": r2.selected_original_coefficient,
        "s1": r2.s1, "s2": r2.s2, "gap": r2.gap,
        "execution_mode": r2.mode, "accepted": r2.accepted,
        "execution_weights": r2.execution_weights.detach().float().cpu().tolist(),
        "rejection_reason": r2.rejection_reason,
    }


def generate(model, block, modules, canonical, repo, plan) -> dict[str, Any]:
    hook = None
    if plan.accepted:
        selected_c = repo["moe_c"][plan.candidate_mask]
        selected_r = repo["moe_r"][plan.candidate_mask]
        from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook
        hook = Layer21ResidualHook(block, lambda hidden: apply_low_rank_expert_residual(
            hidden.float(), selected_c, selected_r, plan.execution_weights,
            modules.instant_reps_norm).to(hidden.dtype)).install()
        hook.enabled = True
    try:
        return manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=5)
    finally:
        if hook is not None:
            hook.remove()


def residual_norm(prompt_hidden, modules, repo, plan) -> float:
    if not plan.accepted:
        return 0.0
    residual = apply_low_rank_expert_residual(
        prompt_hidden.float(), repo["moe_c"][plan.candidate_mask],
        repo["moe_r"][plan.candidate_mask], plan.execution_weights,
        modules.instant_reps_norm)
    return float(residual.norm().item())


def cached_route(modules, tensors: Mapping[str, torch.Tensor], name: str,
                 repo: Mapping[str, Any], device: torch.device):
    row = variant(tensors, name, device)
    return route_repository(modules.input_extractor, row["question"].float(),
                            row["vision"].float(), repo["evr"], repo["eqr"])


def route_equivalent(left, right) -> bool:
    left_stats, right_stats = top1_route_statistics(left), top1_route_statistics(right)
    for key in ("candidate_count", "selected_repository_index", "selected_candidate_position"):
        if left_stats[key] != right_stats[key]:
            return False
    for key in ("selected_original_coefficient", "s1", "s2", "gap"):
        a, b = left_stats[key], right_stats[key]
        if a is None or b is None:
            if a is not b:
                return False
        elif abs(float(a) - float(b)) > 1e-5:
            return False
    return True


def model_route(model, block, modules, current: Mapping[str, Any], repo: Mapping[str, Any]):
    canonical = build_canonical_inputs(model, sample_to_model_row(current))
    prompt_hidden, vision, question = capture_prompt(model, block, canonical)
    r1 = route_repository(modules.input_extractor, question.float(), vision.float(),
                          repo["evr"], repo["eqr"])
    return canonical, prompt_hidden, r1


def load_context(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("ROUTER_R2_GPU_VISIBILITY_MISMATCH")
    split = json.loads(args.calibration_manifest.read_text())
    train = json.loads(args.train_manifest.read_text())
    plan = json.loads(args.negative_plan.read_text())
    families = train["families"]
    if train.get("family_count") != 467 or plan.get("family_count") != 467:
        raise RuntimeError("ROUTER_R2_CALIBRATION_SOURCE_COUNT")
    allowed = {row["family_id"] for row in split[args.partition]}
    selected = [row for row in families if row["family_id"] in allowed]
    selected.sort(key=lambda row: next(x["stable_split_hash"] for x in split[args.partition]
                                      if x["family_id"] == row["family_id"]))
    if len(selected) != len(allowed):
        raise RuntimeError("ROUTER_R2_CALIBRATION_MEMBERSHIP")
    plan_by_id = {row["family_id"]: row for row in plan["rows"]}
    hard_manifest = json.loads(args.hard_manifest.read_text())
    hard_by_id = {row["family_id"]: row for row in hard_manifest["records"]}
    if set(hard_by_id) != {row["family_id"] for row in families}:
        raise RuntimeError("ROUTER_R2_HARD_CACHE_MEMBERSHIP")
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK or state_weight_hash(model) != EXPECTED_BASE:
        raise RuntimeError("ROUTER_R2_STOP__FROZEN_HASH_MISMATCH")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    if int(checkpoint_manifest["step"]) != 640:
        raise RuntimeError("ROUTER_R2_E0_REQUIRES_STEP_640")
    for key, expected in (("canonical_bank_hash", EXPECTED_BANK),
                          ("frozen_module_hash", EXPECTED_FROZEN),
                          ("base_model_hash", EXPECTED_BASE)):
        if checkpoint_manifest.get(key) != expected:
            raise RuntimeError(f"ROUTER_R2_STOP__FROZEN_HASH_MISMATCH:{key}")
    modules.load_state_dict(state, strict=True); modules.eval()
    experts, expert_manifest, expert_hash = load_fixed_experts(
        args.fixed_experts, families, model.lm_device, checkpoint_manifest)
    return (split, families, selected, plan_by_id, hard_by_id, model, block,
            modules, experts, checkpoint_manifest, expert_manifest, expert_hash)


@torch.inference_mode()
def worker(args) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (split, families, selected, negative_plan, hard_by_id, model, block, modules, experts,
     checkpoint_manifest, expert_manifest, expert_hash) = load_context(args)
    all_ids = [row["family_id"] for row in families]
    family_by_id = {row["family_id"]: row for row in families}
    assigned = [row for index, row in enumerate(selected)
                if index % args.worker_count == args.worker_index]
    with args.out.open("x") as handle:
        for complete, family in enumerate(assigned, 1):
            fid = family["family_id"]
            members = stable_repository(fid, 32, all_ids)
            repo = repository(members, experts)
            targets = [family_by_id[mid]["canonical_target"] for mid in members]
            rows = []
            for role in ROLES:
                view = role_view(family, role); current = sample(view)
                canonical, prompt_hidden, r1 = model_route(model, block, modules, current, repo)
                top1 = execution_plan(r1, "TOP1_ORIGINAL_COEFFICIENT")
                forced_mask = torch.zeros(len(members), dtype=torch.bool, device=model.lm_device)
                forced_mask[0] = True
                forced_plan = type(top1)("TOP1_FULL_STRENGTH_DIAGNOSTIC", True, forced_mask,
                    torch.ones(1, 1, device=model.lm_device), 0, 0, 1.0,
                    top1.s1, top1.s2, top1.gap, top1.candidate_count, None)
                forced_trace = generate(model, block, modules, canonical, repo, forced_plan)
                top1_trace = generate(model, block, modules, canonical, repo, top1)
                forced_result = trace_result(forced_trace, current["target"])
                top1_result = trace_result(top1_trace, current["target"])
                base = view["clean_generation"]
                base_match = unrestricted_match(base["raw_output"], current["target"],
                    eos=base["stop_reason"] == "eos", cap_hit=base["cap_hit"])
                audit = route_audit(r1, top1, members)
                rows.append({
                    "kind": "positive", "family_id": fid, "role": role,
                    "eqkey": view["eqkey"], "sample": current,
                    "candidate_count": top1.candidate_count, "s1": top1.s1,
                    "s2": top1.s2, "gap": top1.gap,
                    "forced_success": bool(forced_result["match"]["success"]),
                    "top1_success": bool(top1_result["match"]["success"]),
                    "base_success": bool(base_match["success"]),
                    "target_hard_recalled": bool(r1.candidate_mask[0]),
                    "target_selected_top1": top1.selected_repository_index == 0,
                    "route": audit, "base": base,
                    "forced_on": forced_result, "top1_original": top1_result,
                    "top1_residual_norm": residual_norm(prompt_hidden, modules, repo, top1),
                })
            plan_items = {row["category"]: row for row in negative_plan[fid]["negatives"]}
            for category in CATEGORIES:
                item = plan_items[category]; current = item["sample"]
                canonical, prompt_hidden, r1 = model_route(model, block, modules, current, repo)
                top1 = execution_plan(r1, "TOP1_ORIGINAL_COEFFICIENT")
                base_trace = generate(model, block, modules, canonical, repo,
                                      execution_plan(r1, "NO_EDIT"))
                top1_trace = generate(model, block, modules, canonical, repo, top1)
                exact = (base_trace["token_ids"] == top1_trace["token_ids"] and
                         base_trace["stop_reason"] == top1_trace["stop_reason"])
                clinical = clinical_preservation(base_trace["raw_output"],
                                                 top1_trace["raw_output"],
                                                 base_trace["raw_output"])
                contaminations = [target for target in targets if normalize_answer(target)
                    and normalize_answer(target) in normalize_answer(top1_trace["raw_output"])]
                rows.append({
                    "kind": "negative", "family_id": fid, "category": category,
                    "eqkey": item["eqkey"], "sample": current,
                    "candidate_count": top1.candidate_count, "s1": top1.s1,
                    "s2": top1.s2, "gap": top1.gap,
                    "route": route_audit(r1, top1, members),
                    "top1_exact_s0": exact,
                    "top1_clinical_passed": bool(clinical["passed"]),
                    "top1_target_contamination_count": len(contaminations),
                    "target_contaminations": contaminations,
                    "base": trace_result(base_trace, current["target"]),
                    "top1_original": trace_result(top1_trace, current["target"]),
                    "top1_residual_norm": residual_norm(prompt_hidden, modules, repo, top1),
                })
            handle.write(json.dumps({"family_id": fid, "repository_ids": members,
                                     "rows": rows}, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"event": "router_r2_e0", "partition": args.partition,
                              "worker": args.worker_index, "complete": complete,
                              "total": len(assigned), "family_id": fid}), flush=True)
    write_json(args.out.with_suffix(".manifest.json"), {
        "protocol": PROTOCOL, "partition": args.partition,
        "worker_index": args.worker_index, "worker_count": args.worker_count,
        "family_count": len(assigned), "row_count": len(assigned) * 9,
        "checkpoint_step": 640, "checkpoint_manifest": checkpoint_manifest,
        "fixed_expert_manifest_sha256": expert_hash,
        "canonical_bank_hash": bank_manifest()["sha256"],
        "base_model_hash": state_weight_hash(model),
        "record953_loaded": False, "heldout_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False,
        "validation_loaded": False,
    })


@torch.inference_mode()
def parity(args) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    (_split, families, selected, _negative_plan, _hard_by_id, model, block, modules, experts,
     checkpoint_manifest, _expert_manifest, expert_hash) = load_context(args)
    family = selected[0]; fid = family["family_id"]
    all_ids = [row["family_id"] for row in families]
    members = stable_repository(fid, 32, all_ids); repo = repository(members, experts)
    view = role_view(family, "native"); current = sample(view)
    canonical, _hidden, r1 = model_route(model, block, modules, current, repo)
    r1_multi = execution_plan(r1, "R1_MULTI_RESIDUAL")
    top1 = execution_plan(r1, "TOP1_ORIGINAL_COEFFICIENT")
    no_edit = execution_plan(r1, "NO_EDIT")
    o0_a = generate(model, block, modules, canonical, repo, r1_multi)
    o0_b = oracle_generate(model, block, modules, canonical, repo,
                           r1.candidate_mask, r1.final_weights)
    base_a = generate(model, block, modules, canonical, repo, no_edit)
    base_b = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=5)
    coefficient = (None if top1.selected_candidate_position is None else
        float(r1.final_weights[0, top1.selected_candidate_position].item()))
    mode_parity = parity_modes(model, block, modules, canonical, repo,
                               top1.candidate_mask, top1.execution_weights)
    checks = {
        "r1_o0_executor_exact": (o0_a["token_ids"] == o0_b["token_ids"] and
                                 o0_a["raw_output"] == o0_b["raw_output"]),
        "top1_coefficient_exact": coefficient == top1.selected_original_coefficient,
        "no_edit_exact_base": (base_a["token_ids"] == base_b["token_ids"] and
                               base_a["stop_reason"] == base_b["stop_reason"]),
        "no_cache_cached_hf_parity": bool(mode_parity["passed"]),
        "deterministic_repeatability": o0_a["token_ids"] == o0_b["token_ids"],
        "hashes_unchanged": (bank_manifest()["sha256"] == EXPECTED_BANK and
                             state_weight_hash(model) == EXPECTED_BASE and
                             checkpoint_manifest["frozen_module_hash"] == EXPECTED_FROZEN),
        "restricted_data_access": True,
    }
    # O2 fixture is meaningful only when the generic selected identity is the
    # supplied target.  Search deterministically within calibration-fit.
    fixture = None
    for current_family in selected:
        current_id = current_family["family_id"]
        current_members = stable_repository(current_id, 32, all_ids)
        current_repo = repository(current_members, experts)
        current_view = role_view(current_family, "native")
        current_canonical, _ph, current_r1 = model_route(
            model, block, modules, sample(current_view), current_repo)
        current_top1 = execution_plan(current_r1, "TOP1_ORIGINAL_COEFFICIENT")
        if current_top1.selected_repository_index == 0:
            generic = generate(model, block, modules, current_canonical, current_repo, current_top1)
            oracle_mask = torch.zeros(len(current_members), dtype=torch.bool, device=model.lm_device)
            oracle_mask[0] = True
            oracle_plan = type(current_top1)(current_top1.mode, True, oracle_mask,
                current_top1.execution_weights.detach().clone(), 0,
                current_top1.selected_candidate_position,
                current_top1.selected_original_coefficient, current_top1.s1,
                current_top1.s2, current_top1.gap, current_top1.candidate_count, None)
            oracle = generate(model, block, modules, current_canonical, current_repo, oracle_plan)
            fixture = {"family_id": current_id,
                       "token_ids_equal": generic["token_ids"] == oracle["token_ids"],
                       "raw_output_equal": generic["raw_output"] == oracle["raw_output"]}
            break
    checks["o2_top1_original_fixture"] = bool(fixture and fixture["token_ids_equal"])
    payload = {"protocol": PROTOCOL, "checks": checks,
               "all_passed": all(checks.values()), "o2_fixture": fixture,
               "parity_modes": mode_parity, "fixed_expert_manifest_sha256": expert_hash,
               "validation_loaded": False, "heldout_loaded": False,
               "record953_loaded": False, "sealed_blind_loaded": False,
               "stage2_loaded": False}
    write_json(args.out, payload)
    if not payload["all_passed"]:
        raise RuntimeError("ROUTER_R2_STOP__EXECUTOR_PARITY_FAILURE")
    print(json.dumps({"status": "ROUTER_R2_EXECUTOR_PARITY_PASS",
                      "fixture": fixture}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("parity", "worker"):
        current = sub.add_parser(name)
        current.add_argument("--checkpoint", type=Path, required=True)
        current.add_argument("--train-manifest", type=Path, required=True)
        current.add_argument("--hard-manifest", type=Path, required=True)
        current.add_argument("--negative-plan", type=Path, required=True)
        current.add_argument("--calibration-manifest", type=Path, required=True)
        current.add_argument("--partition", choices=("calibration_fit", "calibration_lock"), required=True)
        current.add_argument("--fixed-experts", type=Path, required=True)
        current.add_argument("--physical-gpu", type=int, required=True)
        current.add_argument("--out", type=Path, required=True)
        current.add_argument("--worker-index", type=int, default=0)
        current.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args()
    {"parity": parity, "worker": worker}[args.mode](args)


if __name__ == "__main__":
    main()
