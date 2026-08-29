#!/usr/bin/env python3
"""Run the frozen validation-only O0--O4 Router-R1 oracle ladder."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook, resolve_layer21_block
from methods.liveedit_med.posthoc_validation import BaseRoutePlan, plan_audit, unrestricted_match
from methods.liveedit_med.router_r1_oracle import (
    ORACLES, PROTOCOL, ROLES, bootstrap_gain_intervals, ensure_target_candidate,
    mechanism_label, oracle_coefficients, original_text_weights,
    sequential_success_gains, tensor_norm_rms, zero_safe_cosine,
)
from methods.liveedit_med.serialization import load_safe_state, tensor_hashes
from methods.liveedit_med.source_ops import (
    SIM_SCALE, RoutePlan, apply_low_rank_expert_residual, route_repository,
)
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, eos_ids, state_weight_hash
from scripts.engram.run_engram_v2_stage0abc_diagnostics import hf_cached_greedy_trace
from scripts.engram.stage0_generation_audit_utils import (
    build_canonical_inputs, manual_cached_greedy_trace, manual_greedy_trace,
)
from scripts.liveedit_med.eqkey_fixed_expert_utils import load_fixed_experts, sha256_file
from scripts.liveedit_med.evaluate_eqkey_clean_checkpoint import repository, repository_ids, sample
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import (
    capture_prompt, compact_trace, load_clean_model, sample_to_model_row,
)


MAX_NEW_TOKENS = 128
EXPECTED_BANK = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"


def write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def flat(values):
    return values[0] if values and isinstance(values[0], list) else values


def near(left: Any, right: Any, tolerance: float = 1e-5) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def trace_summary(trace: Mapping[str, Any], target: str) -> dict[str, Any]:
    match = unrestricted_match(trace["raw_output"], target,
                               eos=trace["stop_reason"] == "eos", cap_hit=trace["cap_hit"])
    first = trace["trajectory"][0] if trace["trajectory"] else {}
    return {
        **compact_trace(trace), "match": match,
        "first_target_token_rank": first.get("target_rank"),
        "first_target_token_margin": first.get("margin"),
        "first_target_token_probability": first.get("target_probability"),
        "first_target_token_id": first.get("target_id"),
        "first_selected_token_id": first.get("selected_id"),
    }


def install_residual_hook(block, modules, repo, mask: torch.Tensor, weights: torch.Tensor):
    if int(mask.sum()) == 0:
        return None
    selected_c, selected_r = repo["moe_c"][mask], repo["moe_r"][mask]
    hook = Layer21ResidualHook(block, lambda hidden: apply_low_rank_expert_residual(
        hidden.float(), selected_c, selected_r, weights,
        modules.instant_reps_norm).to(hidden.dtype)).install()
    hook.enabled = True
    return hook


def remove_hook(hook) -> None:
    if hook is not None:
        hook.remove()


@torch.inference_mode()
def generate(model, block, modules, canonical, repo, mask, weights):
    hook = install_residual_hook(block, modules, repo, mask, weights)
    try:
        return manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=5)
    finally:
        remove_hook(hook)


@torch.inference_mode()
def parity_modes(model, block, modules, canonical, repo, mask, weights):
    traces = {}
    for name in ("no_cache", "cached", "hf"):
        hook = install_residual_hook(block, modules, repo, mask, weights)
        try:
            if name == "no_cache":
                trace = manual_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=5)
            elif name == "hf":
                trace = hf_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS)
            else:
                trace = manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS, eos_ids(model), top_k=5)
        finally:
            remove_hook(hook)
        traces[name] = {key: trace.get(key) for key in (
            "raw_output", "token_ids", "stop_reason", "eos_step", "cap_hit")}
    return {"traces": traces,
            "passed": traces["no_cache"]["token_ids"] == traces["cached"]["token_ids"] == traces["hf"]["token_ids"]}


@torch.inference_mode()
def target_nll(model, block, modules, current_sample, repo, mask, weights):
    row = sample_to_model_row(current_sample)
    inputs, labels, masks = model._build_batch(row)
    hook = install_residual_hook(block, modules, repo, mask, weights)
    try:
        output = model.llava_model(inputs_embeds=inputs,
            attention_mask=masks["attention_mask"].long(), labels=labels,
            return_dict=True, use_cache=False)
    finally:
        remove_hook(hook)
    return float(output.loss.item())


@torch.inference_mode()
def residual_diagnostics(prompt_hidden, modules, repo, mask, weights, target_index):
    one_mask = torch.zeros_like(mask); one_mask[target_index] = True
    target_full = apply_low_rank_expert_residual(
        prompt_hidden.float(), repo["moe_c"][one_mask], repo["moe_r"][one_mask],
        torch.ones(1, 1, device=prompt_hidden.device), modules.instant_reps_norm)
    if bool(mask[target_index]):
        selected = torch.where(mask)[0]
        location = int(torch.where(selected == target_index)[0][0])
        target_weight = weights[:, location:location + 1]
        target_applied = apply_low_rank_expert_residual(
            prompt_hidden.float(), repo["moe_c"][one_mask], repo["moe_r"][one_mask],
            target_weight, modules.instant_reps_norm)
    else:
        target_weight = torch.zeros(1, 1, device=prompt_hidden.device)
        target_applied = torch.zeros_like(prompt_hidden.float())
    if int(mask.sum()):
        fused = apply_low_rank_expert_residual(
            prompt_hidden.float(), repo["moe_c"][mask], repo["moe_r"][mask],
            weights, modules.instant_reps_norm)
    else:
        fused = torch.zeros_like(prompt_hidden.float())
    distractor = fused - target_applied
    td_cos, td_degenerate = zero_safe_cosine(target_applied, distractor)
    tf_cos, tf_degenerate = zero_safe_cosine(target_applied, fused)
    target_norm = float(target_applied.norm().item())
    full_norm = float(target_full.norm().item())
    return {
        "target_full_strength": tensor_norm_rms(target_full),
        "target_applied": tensor_norm_rms(target_applied),
        "distractor_residual_sum": tensor_norm_rms(distractor),
        "fused": tensor_norm_rms(fused),
        "cos_target_distractor_sum": td_cos,
        "cos_target_distractor_sum_degenerate": td_degenerate,
        "cos_target_applied_fused": tf_cos,
        "cos_target_applied_fused_degenerate": tf_degenerate,
        "norm_ratio_fused_over_target": (float(fused.norm().item()) / target_norm) if target_norm else None,
        "norm_ratio_target_applied_over_full": (target_norm / full_norm) if full_norm else None,
        "target_execution_coefficient": float(target_weight.item()),
    }


def plan_values(modules, question, vision, repo, target_index):
    normal = route_repository(modules.input_extractor, question.float(), vision.float(),
                              repo["evr"], repo["eqr"])
    normal_mask = (normal.candidate_mask.detach().clone().bool()
                   if normal.candidate_mask is not None else torch.zeros(len(repo["ids"]), dtype=torch.bool, device=question.device))
    if isinstance(normal, BaseRoutePlan):
        normal_weights = torch.empty(1, 0, device=question.device)
    else:
        normal_weights = normal.final_weights.detach()
    input_text = modules.input_extractor.extract_query(question.float())
    all_raw = torch.einsum("ned,med->nme", input_text, repo["eqr"]).mean(2) * SIM_SCALE
    o1_mask = ensure_target_candidate(normal_mask, target_index)
    o1_text = original_text_weights(all_raw[:, o1_mask])
    selected = torch.where(o1_mask)[0]
    target_location = int(torch.where(selected == target_index)[0][0])
    coefficients = oracle_coefficients(o1_text["final"][:, target_location],
                                       o1_text["sigmoid"][:, target_location])
    one_mask = torch.zeros_like(o1_mask); one_mask[target_index] = True
    masks = {"O0": normal_mask, "O1": o1_mask, "O2": one_mask, "O3": one_mask, "O4": one_mask}
    weights = {"O0": normal_weights, "O1": o1_text["final"], **coefficients}
    return normal, all_raw, o1_text, target_location, masks, weights


def route_metadata(name, normal, repo, all_raw, o1_text, target_location, mask, weights, target_index):
    ids = [rid for rid, selected in zip(repo["ids"], mask.tolist()) if selected]
    values = flat(weights.detach().float().cpu().tolist())
    target_raw = float(all_raw[0, target_index].item())
    result = {
        "oracle": name, "candidate_ids": ids, "candidate_count": len(ids),
        "final_weights": values, "sum_final_weights": float(weights.sum().item()),
        "target_raw_text_score": target_raw, "target_sigmoid": float(torch.sigmoid(all_raw[0, target_index]).item()),
        "target_softmax_O1": float(o1_text["softmax"][0, target_location].item()),
        "target_final_weight_O1": float(o1_text["final"][0, target_location].item()),
        "target_present": bool(mask[target_index]),
        "candidate_mask": [bool(value) for value in mask.detach().cpu().tolist()],
        "visual_scores": (normal.visual_scores.detach().float().cpu().tolist()
                          if normal.visual_scores is not None else []),
        "sentinel_score": (normal.sentinel_score.detach().float().cpu().tolist()
                           if normal.sentinel_score is not None else None),
        "O1_raw_text_scores": flat(o1_text["raw"].detach().float().cpu().tolist()),
        "O1_sigmoid_weights": flat(o1_text["sigmoid"].detach().float().cpu().tolist()),
        "O1_softmax_weights": flat(o1_text["softmax"].detach().float().cpu().tolist()),
        "O1_final_weights": flat(o1_text["final"].detach().float().cpu().tolist()),
    }
    if name == "O0":
        result["frozen_route"] = plan_audit(normal, repo["ids"])
    if name == "O1":
        result.update({"raw_text_scores": flat(o1_text["raw"].detach().float().cpu().tolist()),
                       "sigmoid_weights": flat(o1_text["sigmoid"].detach().float().cpu().tolist()),
                       "softmax_weights": flat(o1_text["softmax"].detach().float().cpu().tolist())})
    return result


def parity_with_frozen(actual, expected):
    return (actual["token_ids"] == expected["token_ids"]
            and actual["stop_reason"] == expected["stop_reason"]
            and bool(actual["match"]["success"]) == bool(expected["match"]["success"]))


def logits_metric_parity(actual, expected):
    wanted = expected.get("target_token_metrics", [])
    if not wanted:
        return actual["first_target_token_rank"] is None
    first = wanted[0]
    return (actual["first_target_token_rank"] == first.get("target_rank")
            and near(actual["first_target_token_margin"], first.get("margin"))
            and near(actual["first_target_token_probability"], first.get("target_probability")))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--fixed-experts", type=Path, required=True)
    parser.add_argument("--frozen-main", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit-families", type=int)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("ROUTER_R1_ORACLE_GPU_VISIBILITY_MISMATCH")
    args.out_dir.mkdir(parents=True)

    manifest = json.loads(args.validation_manifest.read_text())
    families = manifest["families"]
    if manifest.get("split") != "validation" or manifest.get("family_count") != 32:
        raise RuntimeError("ROUTER_R1_ORACLE_VALIDATION_MANIFEST")
    if any(row.get("touches_record953") or row.get("touches_sealed_blind") for row in families):
        raise RuntimeError("ROUTER_R1_ORACLE_FORBIDDEN_FAMILY")
    target_families = families if args.limit_families is None else families[:args.limit_families]
    if not target_families:
        raise RuntimeError("ROUTER_R1_ORACLE_EMPTY_TARGET_SET")
    frozen = json.loads(args.frozen_main.read_text())
    frozen_by_id = {row["family_id"]: row for row in frozen["rows"]}

    model, _bank = load_clean_model(args.physical_gpu)
    clean_hash = state_weight_hash(model)
    if bank_manifest()["sha256"] != EXPECTED_BANK:
        raise RuntimeError("ROUTER_R1_ORACLE_ANCHOR_MISMATCH")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    modules.load_state_dict(state, strict=True); modules.eval()
    step = int(checkpoint_manifest["step"])
    if step not in (80, 640) or int(frozen["step"]) != step:
        raise RuntimeError("ROUTER_R1_ORACLE_CHECKPOINT_SCOPE")
    module_hashes_before = tensor_hashes({name: value.cpu() for name, value in modules.state_dict().items()})
    experts, fixed_manifest, fixed_hash = load_fixed_experts(
        args.fixed_experts, families, model.lm_device, checkpoint_manifest)
    ids = [row["family_id"] for row in families]

    per_input, residual_rows, parity_rows = [], [], []
    for family_index, family in enumerate(target_families):
        family_id = family["family_id"]
        members = repository_ids(family_id, 32, ids)
        repo = repository(members, experts)
        target_index = repo["ids"].index(family_id)
        frozen_family = frozen_by_id[family_id]
        role_views = {view["role"]: view for view in family["canonical_views"]}
        for role in ROLES:
            view = role_views[role]; eqkey = view["eqkey"]; current = sample(view)
            canonical = build_canonical_inputs(model, sample_to_model_row(current))
            prompt_hidden, vision, question = capture_prompt(model, block, canonical)
            normal, all_raw, o1_text, target_location, masks, weights = plan_values(
                modules, question, vision, repo, target_index)
            oracle_outputs = {}
            frozen_routed = frozen_family["repositories"]["32"]["routed"][eqkey]
            frozen_forced = frozen_family["forced_on"][eqkey]
            for oracle in ORACLES:
                trace = generate(model, block, modules, canonical, repo, masks[oracle], weights[oracle])
                generation = trace_summary(trace, current["target"])
                generation["target_sequence_nll"] = target_nll(
                    model, block, modules, current, repo, masks[oracle], weights[oracle])
                route = route_metadata(oracle, normal, repo, all_raw, o1_text,
                                       target_location, masks[oracle], weights[oracle], target_index)
                residual = residual_diagnostics(prompt_hidden, modules, repo,
                                                masks[oracle], weights[oracle], target_index)
                oracle_outputs[oracle] = {"generation": generation, "route": route,
                                          "residual": residual}
                residual_rows.append({"step": step, "family_id": family_id, "eqkey": eqkey,
                                      "role": role, "oracle": oracle, **residual})
            o0_ok = parity_with_frozen(oracle_outputs["O0"]["generation"], frozen_routed)
            o4_ok = parity_with_frozen(oracle_outputs["O4"]["generation"], frozen_forced)
            o4_logits = logits_metric_parity(oracle_outputs["O4"]["generation"], frozen_forced)
            o4_residual = near(oracle_outputs["O4"]["residual"]["fused"]["norm"],
                               frozen_forced["residual_norms"]["fused_residual_norm"], 1e-4)
            if not o0_ok:
                raise RuntimeError(f"ORACLE_BASELINE_ROUTED_PARITY_FAILURE:{step}:{family_id}:{role}")
            if not (o4_ok and o4_logits and o4_residual):
                raise RuntimeError(f"ORACLE_TARGET_ONLY_FORCED_ON_PARITY_FAILURE:{step}:{family_id}:{role}")
            row = {
                "protocol": PROTOCOL, "step": step, "family_id": family_id,
                "canonical_record_id": family["canonical_record_id"], "eqkey": eqkey,
                "role": role, "target": current["target"],
                "repository_ids": repo["ids"], "target_expert_index": target_index,
                "expert_tensor_hashes": {name: tensor_hashes({name: experts[family_id][name].cpu()})[name]
                                         for name in ("eqr", "evr", "moe_c", "moe_r")},
                "oracles": oracle_outputs, "O0_frozen_parity": o0_ok,
                "O4_forced_on_token_success_parity": o4_ok,
                "O4_forced_on_logit_metric_parity": o4_logits,
                "O4_forced_on_residual_parity": o4_residual,
                "target_used_only_as_diagnostic_identity_and_matcher": True,
                "target_added_to_prompt": False, "diagnostic_only": True,
                "not_for_selection": True, "heldout_loaded": False,
                "record953_loaded": False, "sealed_blind_loaded": False,
            }
            per_input.append(row)
            if family_index == 0 and role == "native":
                for oracle in ORACLES:
                    parity = parity_modes(model, block, modules, canonical, repo,
                                          masks[oracle], weights[oracle])
                    parity_rows.append({"step": step, "family_id": family_id,
                                        "role": role, "oracle": oracle, **parity})
                    if not parity["passed"]:
                        raise RuntimeError(f"ROUTER_R1_ORACLE_GENERATION_PARITY_FAILURE:{step}:{oracle}")
            print(json.dumps({"event": "router_r1_oracle", "step": step,
                              "complete": len(per_input), "total": len(target_families) * 4,
                              "family_id": family_id, "role": role}), flush=True)

    successes, aggregate = {}, {}
    for oracle in ORACLES:
        role_counts = {role: sum(row["oracles"][oracle]["generation"]["match"]["success"]
                                 for row in per_input if row["role"] == role) for role in ROLES}
        successes[oracle] = sum(role_counts.values())
        rows = [row["oracles"][oracle]["generation"] for row in per_input]
        margins = [row["first_target_token_margin"] for row in rows
                   if row["first_target_token_margin"] is not None]
        aggregate[oracle] = {
            "success_by_role": role_counts, "success_overall": successes[oracle],
            "first_token_rank1_count": sum(row["first_target_token_rank"] == 1 for row in rows),
            "positive_margin_count": sum((row["first_target_token_margin"] or -math.inf) > 0 for row in rows),
            "mean_first_token_margin": sum(margins) / len(margins),
            "mean_target_nll": sum(row["target_sequence_nll"] for row in rows) / len(rows),
        }
    family_bootstrap = []
    for family in target_families:
        current = [row for row in per_input if row["family_id"] == family["family_id"]]
        family_bootstrap.append({oracle: sum(row["oracles"][oracle]["generation"]["match"]["success"]
                                             for row in current) for oracle in ORACLES})
    oracle_aggregate = {
        "protocol": PROTOCOL, "step": step, "input_count": len(per_input),
        "oracle_order": list(ORACLES), "metrics": aggregate,
        "successes": successes, "sequential_success_gains": sequential_success_gains(successes),
        "bootstrap_intervals": bootstrap_gain_intervals(family_bootstrap),
        "mechanism_label": mechanism_label(successes, o4_parity=True),
        "O0_exact_frozen_parity": all(row["O0_frozen_parity"] for row in per_input),
        "O4_exact_forced_on_parity": all(row["O4_forced_on_token_success_parity"]
                                          and row["O4_forced_on_logit_metric_parity"]
                                          and row["O4_forced_on_residual_parity"] for row in per_input),
        "manual_no_cache_cached_hf_parity": all(row["passed"] for row in parity_rows),
        "fixed_expert_manifest_sha256": fixed_hash,
        "checkpoint_manifest_sha256": sha256_file(args.checkpoint / "manifest.json"),
        "diagnostic_only": True, "not_for_selection": True,
        "heldout_loaded": False, "record953_loaded": False, "sealed_blind_loaded": False,
    }
    if tensor_hashes({name: value.cpu() for name, value in modules.state_dict().items()}) != module_hashes_before:
        raise RuntimeError("ROUTER_R1_ORACLE_MODULE_MUTATION")
    if state_weight_hash(model) != clean_hash or bank_manifest()["sha256"] != EXPECTED_BANK:
        raise RuntimeError("ROUTER_R1_ORACLE_BASE_OR_BANK_MUTATION")
    write_jsonl(args.out_dir / "oracle_per_input.jsonl", per_input)
    write_jsonl(args.out_dir / "residual_diagnostics.jsonl", residual_rows)
    write_json(args.out_dir / "oracle_aggregate.json", oracle_aggregate)
    write_json(args.out_dir / "generation_parity_audit.json", {
        "protocol": PROTOCOL, "step": step, "rows": parity_rows,
        "passed": all(row["passed"] for row in parity_rows),
    })
    print(json.dumps({"status": "ROUTER_R1_ORACLE_CHECKPOINT_COMPLETE",
                      "step": step, "successes": successes,
                      "mechanism_label": oracle_aggregate["mechanism_label"]}, sort_keys=True))


if __name__ == "__main__":
    main()
