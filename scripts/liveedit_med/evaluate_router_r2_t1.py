#!/usr/bin/env python3
"""Evaluate one frozen Router-R2 T1 checkpoint on calibration only.

The worker reuses the already frozen E0 base/forced generations and generates
only the T1 single-residual path.  It deliberately exposes no evaluation-set
arguments.  Every worker is an isolated single-GPU process with an append-only
JSONL output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import normalize_answer, sample_to_model_row, unrestricted_match
from methods.liveedit_med.router_r2 import (PROTOCOL, kplus1_text_logits,
    t1_execution_plan)
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.source_ops import route_repository
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_natural_generation_recovery import clinical_preservation
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, state_weight_hash
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs
from scripts.liveedit_med.eqkey_fixed_expert_utils import load_fixed_experts, sha256_file
from scripts.liveedit_med.evaluate_eqkey_clean_checkpoint import repository, sample
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import capture_prompt, load_clean_model
from scripts.liveedit_med.evaluate_router_r2_e0 import (CATEGORIES, EXPECTED_BANK,
    EXPECTED_BASE, EXPECTED_FROZEN, ROLES, generate, residual_norm, route_audit,
    trace_result, write_json)
from scripts.liveedit_med.train_eqkey_clean_router_r1 import role_view, stable_repository


def load_e0_reference(path: Path, partition: str) -> tuple[dict[str, Any], dict[str, str]]:
    selection = json.loads(path.read_text())
    key = "worker_shards" if partition == "calibration_fit" else "lock_worker_shards"
    expected = selection[key]
    families: dict[str, Any] = {}
    for raw_path, expected_hash in expected.items():
        shard = Path(raw_path)
        if sha256_file(shard) != expected_hash:
            raise RuntimeError("ROUTER_R2_T1_E0_REFERENCE_HASH_MISMATCH")
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["family_id"] in families:
                raise RuntimeError("ROUTER_R2_T1_E0_REFERENCE_DUPLICATE")
            families[row["family_id"]] = row
    return families, expected


def load_context(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("ROUTER_R2_T1_GPU_VISIBILITY_MISMATCH")
    split = json.loads(args.calibration_manifest.read_text())
    train = json.loads(args.train_manifest.read_text())
    negative_plan = json.loads(args.negative_plan.read_text())
    families = train["families"]
    if train.get("family_count") != 467 or negative_plan.get("family_count") != 467:
        raise RuntimeError("ROUTER_R2_T1_CALIBRATION_SOURCE_COUNT")
    allowed_rows = split[args.partition]
    allowed = {row["family_id"] for row in allowed_rows}
    stable = {row["family_id"]: row["stable_split_hash"] for row in allowed_rows}
    selected = sorted((row for row in families if row["family_id"] in allowed),
                      key=lambda row: stable[row["family_id"]])
    if len(selected) != len(allowed):
        raise RuntimeError("ROUTER_R2_T1_CALIBRATION_MEMBERSHIP")
    plan_by_id = {row["family_id"]: row for row in negative_plan["rows"]}
    hard = json.loads(args.hard_manifest.read_text())
    if {row["family_id"] for row in hard["records"]} != {row["family_id"] for row in families}:
        raise RuntimeError("ROUTER_R2_T1_HARD_CACHE_MEMBERSHIP")
    reference, reference_hashes = load_e0_reference(args.e0_selection, args.partition)
    if set(reference) != allowed:
        raise RuntimeError("ROUTER_R2_T1_E0_REFERENCE_MEMBERSHIP")

    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK or state_weight_hash(model) != EXPECTED_BASE:
        raise RuntimeError("ROUTER_R2_STOP__FROZEN_HASH_MISMATCH")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    step = int(checkpoint_manifest["step"])
    if step not in (80, 160, 240, 320, 400, 480, 560, 640) \
            or checkpoint_manifest.get("candidate") != "R2_T1_EXPLICIT_NOEDIT_KPLUS1":
        raise RuntimeError("ROUTER_R2_T1_CHECKPOINT_PROTOCOL")
    for key, expected in (("canonical_bank_hash", EXPECTED_BANK),
                          ("frozen_module_hash", EXPECTED_FROZEN),
                          ("base_model_hash", EXPECTED_BASE)):
        if checkpoint_manifest.get(key) != expected:
            raise RuntimeError(f"ROUTER_R2_STOP__FROZEN_HASH_MISMATCH:{key}")
    module_state = {name[len("modules."):]: value for name, value in state.items()
                    if name.startswith("modules.")}
    if set(state) != {f"modules.{name}" for name in module_state} | {"no_edit_text_key"}:
        raise RuntimeError("ROUTER_R2_T1_CHECKPOINT_TENSOR_SET")
    modules.load_state_dict(module_state, strict=True); modules.eval()
    no_edit_key = state["no_edit_text_key"].float().to(model.lm_device)
    expert_context = {**checkpoint_manifest, "selected_generator_checkpoint_step": 3000}
    experts, expert_manifest, expert_hash = load_fixed_experts(
        args.fixed_experts, families, model.lm_device, expert_context)
    if expert_manifest["global_expert_hash"] != checkpoint_manifest["frozen_expert_hash"]:
        raise RuntimeError("ROUTER_R2_T1_EXPERT_HASH_MISMATCH")
    return (split, families, selected, plan_by_id, reference, reference_hashes,
            model, block, modules, no_edit_key, experts, checkpoint_manifest,
            expert_manifest, expert_hash)


def model_route(model, block, modules, no_edit_key, current: Mapping[str, Any], repo):
    canonical = build_canonical_inputs(model, sample_to_model_row(current))
    prompt_hidden, vision, question = capture_prompt(model, block, canonical)
    r1 = route_repository(modules.input_extractor, question.float(), vision.float(),
                          repo["evr"], repo["eqr"])
    input_text = modules.input_extractor.extract_query(question.float())
    logits = kplus1_text_logits(input_text, repo["eqr"], no_edit_key)
    t1, decision = t1_execution_plan(
        r1, logits[:, :-1], logits[:, -1:], tau_null=-1.0e30, tau_gap=-1.0e30)
    return canonical, prompt_hidden, r1, t1, decision, logits


def t1_route_audit(r1, t1, decision, logits, members):
    result = route_audit(r1, t1, members)
    result.update({
        "expert_logits": logits[:, :-1].detach().float().cpu().tolist(),
        "no_edit_logit": float(decision["null_logit"]),
        "null_margin": float(decision["null_margin"]),
        "expert_gap": float(decision["expert_gap"]),
        "hard_gate_admitted": bool(decision["hard_gate_admitted"]),
        "expert_is_overall_top": bool(decision["expert_is_overall_top"]),
        "unthresholded_t1_accepted": bool(decision["accepted"]),
    })
    return result


def reference_rows(reference_family: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    positive = {row["role"]: row for row in reference_family["rows"] if row["kind"] == "positive"}
    negative = {row["category"]: row for row in reference_family["rows"] if row["kind"] == "negative"}
    if set(positive) != set(ROLES) or set(negative) != set(CATEGORIES):
        raise RuntimeError("ROUTER_R2_T1_E0_REFERENCE_SCHEMA")
    return positive, negative


@torch.inference_mode()
def worker(args) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (_split, families, selected, negative_plan, reference, reference_hashes,
     model, block, modules, no_edit_key, experts, checkpoint_manifest,
     _expert_manifest, expert_hash) = load_context(args)
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
            positive_ref, negative_ref = reference_rows(reference[fid])
            rows = []
            for role in ROLES:
                view = role_view(family, role); current = sample(view); ref = positive_ref[role]
                canonical, prompt_hidden, r1, t1, decision, logits = model_route(
                    model, block, modules, no_edit_key, current, repo)
                if t1.accepted:
                    top1_result = trace_result(generate(model, block, modules, canonical, repo, t1),
                                               current["target"])
                    top1_success = bool(top1_result["match"]["success"])
                else:
                    top1_result = {**ref["base"], "match": {"success": bool(ref["base_success"])}}
                    top1_success = bool(ref["base_success"])
                rows.append({
                    "kind": "positive", "family_id": fid, "role": role,
                    "eqkey": view["eqkey"], "sample": current,
                    "candidate_count": t1.candidate_count,
                    "z1": float(decision["z1"]), "z2": float(decision["z2"]),
                    "null_logit": float(decision["null_logit"]),
                    "null_margin": float(decision["null_margin"]),
                    "expert_gap": float(decision["expert_gap"]),
                    "hard_gate_admitted": bool(decision["hard_gate_admitted"]),
                    "expert_is_overall_top": bool(decision["expert_is_overall_top"]),
                    "forced_success": bool(ref["forced_success"]),
                    "top1_success": top1_success, "base_success": bool(ref["base_success"]),
                    "target_hard_recalled": bool(r1.candidate_mask[0]),
                    "target_selected_top1": int(decision["selected_repository_index"]) == 0,
                    "route": t1_route_audit(r1, t1, decision, logits, members),
                    "base": ref["base"], "forced_on": ref["forced_on"],
                    "top1_original": top1_result,
                    "top1_residual_norm": residual_norm(prompt_hidden, modules, repo, t1),
                })
            plan_items = {row["category"]: row for row in negative_plan[fid]["negatives"]}
            for category in CATEGORIES:
                item = plan_items[category]; current = item["sample"]; ref = negative_ref[category]
                canonical, prompt_hidden, r1, t1, decision, logits = model_route(
                    model, block, modules, no_edit_key, current, repo)
                if t1.accepted:
                    trace = generate(model, block, modules, canonical, repo, t1)
                    top1_result = trace_result(trace, current["target"])
                    exact = (top1_result["token_ids"] == ref["base"]["token_ids"] and
                             top1_result["stop_reason"] == ref["base"]["stop_reason"])
                    clinical = clinical_preservation(ref["base"]["raw_output"],
                                                     top1_result["raw_output"],
                                                     ref["base"]["raw_output"])
                    contaminations = [target for target in targets if normalize_answer(target)
                        and normalize_answer(target) in normalize_answer(top1_result["raw_output"])]
                else:
                    top1_result = ref["base"]; exact = True
                    clinical = {"passed": True}; contaminations = []
                rows.append({
                    "kind": "negative", "family_id": fid, "category": category,
                    "eqkey": item["eqkey"], "sample": current,
                    "candidate_count": t1.candidate_count,
                    "z1": float(decision["z1"]), "z2": float(decision["z2"]),
                    "null_logit": float(decision["null_logit"]),
                    "null_margin": float(decision["null_margin"]),
                    "expert_gap": float(decision["expert_gap"]),
                    "hard_gate_admitted": bool(decision["hard_gate_admitted"]),
                    "expert_is_overall_top": bool(decision["expert_is_overall_top"]),
                    "route": t1_route_audit(r1, t1, decision, logits, members),
                    "top1_exact_s0": exact,
                    "top1_clinical_passed": bool(clinical["passed"]),
                    "top1_target_contamination_count": len(contaminations),
                    "target_contaminations": contaminations,
                    "base": ref["base"], "top1_original": top1_result,
                    "top1_residual_norm": residual_norm(prompt_hidden, modules, repo, t1),
                })
            handle.write(json.dumps({"family_id": fid, "repository_ids": members,
                                     "rows": rows}, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"event": "router_r2_t1", "step": checkpoint_manifest["step"],
                              "partition": args.partition, "worker": args.worker_index,
                              "complete": complete, "total": len(assigned)}), flush=True)
    write_json(args.out.with_suffix(".manifest.json"), {
        "protocol": PROTOCOL, "candidate": "R2_T1_EXPLICIT_NOEDIT_KPLUS1",
        "partition": args.partition, "worker_index": args.worker_index,
        "worker_count": args.worker_count, "family_count": len(assigned),
        "row_count": len(assigned) * 9, "checkpoint_step": checkpoint_manifest["step"],
        "checkpoint_manifest_sha256": sha256_file(args.checkpoint / "manifest.json"),
        "fixed_expert_manifest_sha256": expert_hash,
        "e0_reference_shards": reference_hashes,
        "canonical_bank_hash": bank_manifest()["sha256"],
        "base_model_hash": state_weight_hash(model),
        "restricted_to_calibration": True, "evaluation_data_loaded": False,
        "record953_loaded": False, "sealed_blind_loaded": False,
        "stage2_loaded": False,
    })


@torch.inference_mode()
def parity(args) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    (_split, families, selected, _negative_plan, reference, _reference_hashes,
     model, block, modules, no_edit_key, experts, checkpoint_manifest,
     _expert_manifest, expert_hash) = load_context(args)
    family = selected[0]; fid = family["family_id"]
    members = stable_repository(fid, 32, [row["family_id"] for row in families])
    repo = repository(members, experts); view = role_view(family, "native")
    canonical, _hidden, r1, t1, decision, _logits = model_route(
        model, block, modules, no_edit_key, sample(view), repo)
    no_edit_plan, no_edit_decision = t1_execution_plan(
        r1, torch.full((1, len(members)), -1.0, device=model.lm_device),
        torch.full((1, 1), 1.0, device=model.lm_device), tau_null=0.0, tau_gap=0.0)
    base = generate(model, block, modules, canonical, repo, no_edit_plan)
    positive_ref, _negative_ref = reference_rows(reference[fid])
    ref = positive_ref["native"]["base"]
    coefficient_exact = True
    if t1.accepted:
        local = t1.selected_candidate_position
        coefficient_exact = (local is not None and
            t1.selected_original_coefficient == float(r1.final_weights[0, local].item()) and
            t1.execution_weights.shape == (1, 1) and
            float(t1.execution_weights[0, 0].item()) == t1.selected_original_coefficient)
    checks = {
        "checkpoint_hash_verified": True,
        "explicit_no_edit_wins_and_rejects": (not no_edit_decision["expert_is_overall_top"]
                                               and not no_edit_plan.accepted),
        "no_edit_exact_frozen_base": (base["token_ids"] == ref["token_ids"] and
                                      base["stop_reason"] == ref["stop_reason"]),
        "selected_original_coefficient_exact": coefficient_exact,
        "single_or_zero_residual": int(t1.candidate_mask.sum().item()) <= 1,
        "hashes_unchanged": (bank_manifest()["sha256"] == EXPECTED_BANK and
                             state_weight_hash(model) == EXPECTED_BASE and
                             checkpoint_manifest["frozen_module_hash"] == EXPECTED_FROZEN),
        "restricted_data_access": True,
    }
    payload = {"protocol": PROTOCOL, "candidate": "R2_T1_EXPLICIT_NOEDIT_KPLUS1",
        "checkpoint_step": checkpoint_manifest["step"], "checks": checks,
        "all_passed": all(checks.values()), "decision_fixture": decision,
        "fixed_expert_manifest_sha256": expert_hash,
        "evaluation_data_loaded": False, "record953_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False}
    write_json(args.out, payload)
    if not payload["all_passed"]:
        raise RuntimeError("ROUTER_R2_STOP__EXECUTOR_PARITY_FAILURE")
    print(json.dumps({"status": "ROUTER_R2_T1_PARITY_PASS",
                      "step": checkpoint_manifest["step"]}, sort_keys=True))


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
        current.add_argument("--e0-selection", type=Path, required=True)
        current.add_argument("--physical-gpu", type=int, required=True)
        current.add_argument("--out", type=Path, required=True)
        current.add_argument("--worker-index", type=int, default=0)
        current.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args(); {"parity": parity, "worker": worker}[args.mode](args)


if __name__ == "__main__":
    main()
