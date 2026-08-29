#!/usr/bin/env python3
"""Construct and evaluate EqKey-clean held-out hard negatives."""
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
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.eqkey_clean_fast import PROTOCOL, canonical_hash, normalize_target
from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import normalize_answer
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_natural_generation_recovery import clinical_preservation
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, eos_ids
from scripts.engram.run_record953_routed_banked_lora_v1_1 import visible_input_audit
from scripts.engram.equivalence_aware_router_utils import router_input_equivalence_key
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs, manual_cached_greedy_trace
from scripts.liveedit_med.evaluate_eqkey_clean_checkpoint import (expert_for_family, repository, sample)
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import (MAX_NEW_TOKENS,
    capture_teacher_forced, compact_trace, load_clean_model, routed_generation)
from scripts.liveedit_med.evaluate_router_r1_checkpoint import variant
from scripts.liveedit_med.run_official_style_medical_aggregate import clean_tf, routed_tf
from methods.liveedit_med.posthoc_validation import sample_to_model_row
from scripts.liveedit_med.eqkey_fixed_expert_utils import load_fixed_experts


CATEGORIES = ("same_image_different_question", "same_question_different_image",
              "visual_nearest", "text_nearest", "joint_near_miss")


def native_view(family: Mapping[str, Any]) -> Mapping[str, Any]:
    return next(row for row in family["canonical_views"]
                if row["eqkey"] == family["canonical_native_eqkey"])


def stable_others(target: str, families: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted((row for row in families if row["family_id"] != target),
                  key=lambda row: (canonical_hash([target, row["family_id"]]), row["family_id"]))


def feature(view: Mapping[str, Any], name: str, device: torch.device) -> torch.Tensor:
    tensors = load_file(view["cache_file_path"], device="cpu")
    cached = variant(tensors, view["role"], device)
    return F.normalize(cached[name].float().mean(1), dim=1)[0].cpu()


def ranked(target: str, families: list[Mapping[str, Any]], features: Mapping[str, torch.Tensor]):
    current = features[target]
    return sorted((row for row in families if row["family_id"] != target),
                  key=lambda row: (-float(torch.dot(current, features[row["family_id"]]).item()),
                                   canonical_hash([target, row["family_id"]]), row["family_id"]))


def candidate_sample(category: str, target: Mapping[str, Any], left: Mapping[str, Any],
                     right: Mapping[str, Any] | None = None) -> dict[str, str]:
    target_view, left_view = native_view(target), native_view(left)
    right_view = native_view(right or left)
    if category in ("same_image_different_question", "visual_nearest"):
        return {"image": target_view["image_path"], "prompt": left_view["question"],
                "target": target["canonical_target"]}
    if category in ("same_question_different_image", "text_nearest"):
        return {"image": left_view["image_path"], "prompt": target_view["question"],
                "target": target["canonical_target"]}
    return {"image": left_view["image_path"], "prompt": right_view["question"],
            "target": target["canonical_target"]}


def valid_other(target: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    return normalize_target(target["canonical_target"]) != normalize_target(other["canonical_target"])


def teacher_loss(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                 labels[:, 1:].reshape(-1), ignore_index=-100).item())


def answer_kl(clean_logits: torch.Tensor, routed_logits: torch.Tensor,
              labels: torch.Tensor) -> float:
    answer = torch.where(labels[0] != -100)[0]
    predictors = answer - 1
    if not len(predictors) or bool((predictors < 0).any()):
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:safety_predictors")
    return float(F.kl_div(F.log_softmax(routed_logits[0, predictors].float(), dim=-1),
                          F.softmax(clean_logits[0, predictors].float(), dim=-1),
                          reduction="batchmean").item())


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--family-components", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--limit-families", type=int)
    parser.add_argument("--fixed-experts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_GPU_VISIBILITY_MISMATCH")
    manifest = json.loads(args.heldout_manifest.read_text())
    manifest_split = str(manifest.get("split"))
    if manifest_split not in ("validation", "heldout"):
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:safety_split")
    repository_families = manifest["families"]
    families = (repository_families if args.limit_families is None
                else repository_families[:args.limit_families])
    all_families = [json.loads(line) for line in args.family_components.read_text().splitlines() if line.strip()]
    if len(repository_families) != 32:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:heldout_count")
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:bank")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    modules.load_state_dict(state, strict=True); modules.eval()
    fixed_manifest = None; fixed_manifest_hash = None
    if args.fixed_experts is None:
        experts = {family["family_id"]: expert_for_family(modules, family, model.lm_device)
                   for family in repository_families}
    else:
        experts, fixed_manifest, fixed_manifest_hash = load_fixed_experts(
            args.fixed_experts, repository_families, model.lm_device, checkpoint_manifest)
    repo = repository([family["family_id"] for family in repository_families], experts)
    visual_features = {family["family_id"]: feature(native_view(family), "vision", model.lm_device)
                       for family in repository_families}
    text_features = {family["family_id"]: feature(native_view(family), "question", model.lm_device)
                     for family in repository_families}
    joint_features = {family["family_id"]: F.normalize(torch.cat(
        [visual_features[family["family_id"]], text_features[family["family_id"]]]), dim=0)
        for family in repository_families}
    positive_eqkeys = {eqkey for family in all_families for eqkey in family["positive_eqkeys"]}
    train_positive = {eqkey for family in all_families if family["touches_original_train"]
                      for eqkey in family["positive_eqkeys"]}
    validation_positive = {eqkey for family in all_families if family["touches_original_validation"]
                           for eqkey in family["positive_eqkeys"]}
    heldout_positive = {eqkey for family in all_families if family["touches_original_heldout"]
                        for eqkey in family["positive_eqkeys"]}
    used_negative_eqkeys: set[str] = set()
    image_audits: dict[str, dict[str, Any]] = {}

    def mixed_visible_audit(current: Mapping[str, Any]) -> dict[str, Any]:
        image_path = str(current["image"])
        if image_path not in image_audits:
            image_audits[image_path] = visible_input_audit(model, {
                "question": current["prompt"], "image_path": image_path,
                "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest()})
        base = image_audits[image_path]
        question = str(current["prompt"])
        prompt = model._conversation_prompt(question, None)
        input_ids = model.tokenizer_image_token(prompt, model.llava_tokenizer,
            model.IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.lm_device)
        ids = [int(value) for value in input_ids[0].tolist()]
        mask = [1] * len(ids)
        result = dict(base)
        result.update({
            "router_input_equivalence_key": router_input_equivalence_key(
                base["processed_pixel_tensor_sha256"], base["image_sizes_equivalence_field"], ids, mask),
            "normalized_original_question": " ".join(question.casefold().split()),
            "question_token_ids": [int(value) for value in model.llava_tokenizer(
                question, add_special_tokens=False).input_ids],
            "rendered_canonical_routing_prompt": prompt,
            "routing_input_ids": ids, "attention_mask": mask,
        })
        return result
    targets = [family["canonical_target"] for family in repository_families]
    rows = []
    for complete, family in enumerate(families, 1):
        family_id = family["family_id"]
        orders = {
            "same_image_different_question": stable_others(family_id, repository_families),
            "same_question_different_image": stable_others(family_id, repository_families),
            "visual_nearest": ranked(family_id, repository_families, visual_features),
            "text_nearest": ranked(family_id, repository_families, text_features),
            "joint_near_miss": ranked(family_id, repository_families, joint_features),
        }
        negatives = []
        for category in CATEGORIES:
            candidates = [row for row in orders[category] if valid_other(family, row)]
            selected = None
            for index, left in enumerate(candidates):
                right = candidates[(index + 1) % len(candidates)] if category == "joint_near_miss" else None
                options = [(candidate_sample(category, family, left, right), right)]
                # Some families share the same raw question or image without
                # sharing a complete positive EqKey.  A single-axis candidate
                # list can therefore be exhausted by the earlier globally
                # unique negatives.  Continue deterministically through clean
                # cross-family image/question pairs; EqKey exclusion below
                # remains the authoritative safety boundary.
                for offset in range(1, len(candidates)):
                    alternate = candidates[(index + offset) % len(candidates)]
                    left_view, alternate_view = native_view(left), native_view(alternate)
                    options.extend([
                        ({"image": left_view["image_path"], "prompt": alternate_view["question"],
                          "target": family["canonical_target"]}, alternate),
                        ({"image": alternate_view["image_path"], "prompt": left_view["question"],
                          "target": family["canonical_target"]}, alternate),
                    ])
                for current, actual_right in options:
                    image_hash = hashlib.sha256(Path(current["image"]).read_bytes()).hexdigest()
                    visible = mixed_visible_audit(current)
                    eqkey = visible["router_input_equivalence_key"]
                    if eqkey not in positive_eqkeys and eqkey not in used_negative_eqkeys:
                        selected = (current, eqkey, left, actual_right)
                        used_negative_eqkeys.add(eqkey); break
                if selected is not None:
                    break
            if selected is None:
                raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:negative_exhausted:{family_id}:{category}")
            current, eqkey, left, right = selected
            canonical = build_canonical_inputs(model, sample_to_model_row(current))
            clean = compact_trace(manual_cached_greedy_trace(model, canonical, MAX_NEW_TOKENS,
                                                              eos_ids(model), top_k=1))
            clean_loss = capture_teacher_forced(model, block, current)["loss"]
            clean_logits, clean_labels = clean_tf(model, current)
            routed = routed_generation(model, block, modules, current, repo)
            routed_logits, routed_labels, route, norms = routed_tf(model, block, modules, current, repo)
            routed_loss = teacher_loss(routed_logits, routed_labels)
            exact = clean["token_ids"] == routed["token_ids"] and clean["stop_reason"] == routed["stop_reason"]
            clinical = clinical_preservation(clean["raw_output"], routed["raw_output"], clean["raw_output"])
            contaminations = [target for target in targets if normalize_answer(target)
                              and normalize_answer(target) in normalize_answer(routed["raw_output"])]
            negatives.append({
                "category": category, "eqkey": eqkey, "sample": current,
                "provenance": {"left_family_id": left["family_id"],
                               "right_family_id": None if right is None else right["family_id"]},
                "s0": clean, "routed": routed, "route_teacher_forced": route,
                "residual_norms_teacher_forced": norms, "exact_s0": exact,
                "clinical": clinical, "target_contaminations": contaminations,
                "s0_nll": clean_loss, "routed_nll": routed_loss, "nll_drift": routed_loss - clean_loss,
                "kl_to_s0": answer_kl(clean_logits, routed_logits, clean_labels),
            })
        rows.append({"family_id": family_id, "negatives": negatives})
        print(json.dumps({"event": "eqkey_clean_safety", "complete": complete,
                          "total": len(families), "family_id": family_id}), flush=True)
    values = [negative for row in rows for negative in row["negatives"]]
    audit = {
        "positive_positive_family_partition": len(positive_eqkeys) == sum(len(family["positive_eqkeys"])
                                                                           for family in all_families),
        "positive_negative_disjoint": not bool(positive_eqkeys & used_negative_eqkeys),
        "negative_negative_unique": len(used_negative_eqkeys) == len(values),
        "negative_train_collision_absent": not bool(train_positive & used_negative_eqkeys),
        "negative_validation_collision_absent": not bool(validation_positive & used_negative_eqkeys),
        "negative_heldout_collision_absent": not bool(heldout_positive & used_negative_eqkeys),
    }
    payload = {
        "protocol": PROTOCOL, "split": manifest_split,
        "selected_step": int(checkpoint_manifest["step"]),
        "hard_negative_count": len(values), "hard_negative_exact_s0": sum(row["exact_s0"] for row in values),
        "clinical_canonical_failures": sum(not row["clinical"]["passed"] for row in values),
        "target_contaminations": sum(len(row["target_contaminations"]) for row in values),
        "mean_nll_drift": sum(row["nll_drift"] for row in values) / max(1, len(values)),
        "mean_negative_locality_kl": sum(row["kl_to_s0"] for row in values) / max(1, len(values)),
        "category_exact_s0": {category: sum(row["exact_s0"] for row in values if row["category"] == category)
                              for category in CATEGORIES},
        "eqkey_role_audit": audit, "rows": rows, "record953_loaded": False,
        "sealed_blind_loaded": False, "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH,
        "fixed_experts_used": args.fixed_experts is not None,
        "fixed_expert_manifest_sha256": fixed_manifest_hash,
        "fixed_expert_generator_step": None if fixed_manifest is None else fixed_manifest["generator_checkpoint_step"],
    }
    if not all(audit.values()) or not payload["canonical_bank_unchanged"]:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:safety_audit")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("selected_step", "hard_negative_count",
        "hard_negative_exact_s0", "clinical_canonical_failures", "target_contaminations")}, sort_keys=True))


if __name__ == "__main__":
    main()
