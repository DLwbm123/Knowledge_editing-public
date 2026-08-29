#!/usr/bin/env python3
"""Read-only Router-R1 gradient-conflict and feature-scale diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.router_r1 import (
    configure_router_only, negative_text_absolute_loss, negative_visual_loss,
    positive_text_losses, positive_visual_loss,
)
from methods.liveedit_med.router_r1_oracle import PROTOCOL, zero_safe_cosine
from methods.liveedit_med.serialization import load_safe_state, tensor_hashes
from methods.liveedit_med.source_ops import SIM_SCALE
from methods.liveedit_med.source_training_continuation import SourceTrainingContinuationMode
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, state_weight_hash
from scripts.liveedit_med.train_eqkey_clean_router_r1 import (
    fixed_experts, load_locality, load_view, query_keys, role_view, stable_repository,
    training_variant,
)
from scripts.liveedit_med.train_liveedit_med_v4_source import grouped_residual_losses, load_training_model, spans


STRICT_MODE = SourceTrainingContinuationMode.STRICT_SOURCE_REAPPLY_LAYER21
EXPECTED_BANK = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"
LOSSES = ("L_hard", "L_soft_abs", "L_soft_rel", "L_positive", "L_negative")
SIZES = (1, 4, 8, 16, 32)


def write_json(path: Path, value) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def write_jsonl(path: Path, rows) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parameter_group(name: str) -> str | None:
    if name.startswith("edit_extractor."):
        return "edit_extractor"
    if name == "input_extractor.vis_rep_prot":
        return "input_extractor.vis_rep_prot"
    if name.startswith(("input_extractor.eqe1", "input_extractor.layer_norm1",
                        "input_extractor.ca_query_info_ext1", "input_extractor.ca_vision_info_ext")):
        return "input_extractor.visual_branch"
    if name.startswith(("input_extractor.eqe2", "input_extractor.layer_norm2",
                        "input_extractor.ca_query_info_ext2")):
        return "input_extractor.text_branch"
    return None


def flatten_gradients(modules) -> dict[str, torch.Tensor]:
    pieces: dict[str, list[torch.Tensor]] = {name: [] for name in (
        "edit_extractor", "input_extractor.visual_branch",
        "input_extractor.text_branch", "input_extractor.vis_rep_prot")}
    for name, parameter in modules.named_parameters():
        group = parameter_group(name)
        if group is None or not parameter.requires_grad:
            continue
        value = parameter.grad
        pieces[group].append((torch.zeros_like(parameter) if value is None else value).detach().float().cpu().reshape(-1))
    return {name: torch.cat(values) for name, values in pieces.items()}


def dot_decomposition(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool]:
    a, b = left.detach().float().reshape(-1), right.detach().float().reshape(-1)
    dot = float(torch.dot(a, b).item()); cosine, degenerate = zero_safe_cosine(a, b)
    reconstructed = float(a.norm().item() * b.norm().item() * cosine)
    return {"dot": dot, "norm_product": float(a.norm().item() * b.norm().item()),
            "cosine": cosine, "reconstructed_dot": reconstructed,
            "absolute_error": abs(dot - reconstructed), "degenerate": degenerate}


def load_diagnostic_rows(family_ids, family_by_id, hard, device):
    result = {}
    for family_id in family_ids:
        family = family_by_id[family_id]
        result[family_id] = {
            "native": load_view(role_view(family, "native"), device),
            "semantic": load_view(role_view(family, "textual"), device),
            "negative": training_variant(load_file(hard[family_id]["file_path"], device="cpu"),
                                         "same_image_different_question", device),
            "locality": load_locality(family, device),
        }
    return result


def losses_and_features(modules, model, suffix_device, family_ids, all_ids,
                        cached_rows, expert_bank, size):
    all_eqr=[]; all_evr=[]; all_c=[]; all_r=[]; targets=[]
    for family_id in family_ids:
        members = stable_repository(family_id, size, all_ids)
        begin = len(all_eqr)
        for member in members:
            fixed = expert_bank[member]
            all_eqr.append(fixed["eqr"].to(model.lm_device)); all_evr.append(fixed["evr"].to(model.lm_device))
            all_c.append(fixed["moe_c"].to(model.lm_device)); all_r.append(fixed["moe_r"].to(model.lm_device))
        targets.append((family_id, members, begin, begin + size))
    eqrs=torch.cat(all_eqr); evrs=torch.cat(all_evr); moe_cs=torch.cat(all_c); moe_rs=torch.cat(all_r)
    hard_losses=[]; abs_losses=[]; rel_losses=[]; masks=[]
    native_rows=[]; semantic_rows=[]; negative_rows=[]; locality_rows=[]; feature_rows=[]
    for family_id, members, begin, end in targets:
        rows=cached_rows[family_id]; mask=torch.zeros(len(all_eqr),dtype=torch.bool,device=model.lm_device)
        mask[begin:end]=True; masks.extend([mask,mask,mask,mask])
        native_rows.append(rows["native"]); semantic_rows.append(rows["semantic"])
        negative_rows.append(rows["negative"]); locality_rows.append(rows["locality"])
        repo_eqr,repo_evr=eqrs[begin:end],evrs[begin:end]
        for current in (rows["native"],rows["semantic"]):
            iqr,ivr,sentinel,_=query_keys(modules,current)
            loc_edit=modules.edit_extractor.extract_query(spans(rows["locality"])[1])
            hard_losses.append(positive_visual_loss(ivr,repo_evr,sentinel,0))
            absolute,relative=positive_text_losses(iqr,repo_eqr,0,loc_edit)
            abs_losses.append(absolute);rel_losses.append(relative)
        for current in (rows["negative"],rows["locality"]):
            iqr,ivr,sentinel,_=query_keys(modules,current)
            hard_losses.append(negative_visual_loss(ivr,repo_evr,sentinel))
            abs_losses.append(negative_text_absolute_loss(iqr,repo_eqr))

        iqr,ivr,sentinel,_=query_keys(modules,rows["native"])
        visual_scores=torch.einsum("bed,med->bm",ivr,repo_evr) * SIM_SCALE / ivr.shape[1]
        sentinel_score=torch.einsum("bed,bed->b",ivr,sentinel) * SIM_SCALE / ivr.shape[1]
        text_scores=torch.einsum("bed,med->bm",iqr,repo_eqr) * SIM_SCALE / iqr.shape[1]
        candidates=(visual_scores>sentinel_score[:,None])[0]
        target_visual=dot_decomposition(ivr,repo_evr[0:1]);target_text=dot_decomposition(iqr,repo_eqr[0:1])
        distract_visual=[dot_decomposition(ivr,repo_evr[index:index+1]) for index in range(1,size)]
        distract_text=[dot_decomposition(iqr,repo_eqr[index:index+1]) for index in range(1,size)]
        feature_rows.append({
            "family_id":family_id,"repository_size":size,"repository_ids":members,
            "candidate_count":int(candidates.sum()),"target_candidate":bool(candidates[0]),
            "norm_input_visual":float(ivr.norm()),"norm_target_edit_visual":float(repo_evr[0].norm()),
            "norm_distractor_visual_mean":sum(float(repo_evr[i].norm()) for i in range(1,size))/max(1,size-1),
            "norm_sentinel_visual":float(sentinel.norm()),"norm_input_text":float(iqr.norm()),
            "norm_target_edit_text":float(repo_eqr[0].norm()),
            "norm_distractor_text_mean":sum(float(repo_eqr[i].norm()) for i in range(1,size))/max(1,size-1),
            "visual_target":target_visual,"visual_sentinel":dot_decomposition(ivr,sentinel),
            "visual_distractors":distract_visual,"text_target":target_text,
            "text_distractors":distract_text,"raw_visual_target_score":float(visual_scores[0,0]),
            "raw_sentinel_score":float(sentinel_score[0]),
            "raw_visual_distractor_scores":[float(value) for value in visual_scores[0,1:]],
            "raw_text_target_score":float(text_scores[0,0]),
            "raw_text_distractor_scores":[float(value) for value in text_scores[0,1:]],
        })
    route_losses={"L_hard":torch.stack(hard_losses).mean(),
                  "L_soft_abs":torch.stack(abs_losses).mean(),
                  "L_soft_rel":torch.stack(rel_losses).mean()}
    output=grouped_residual_losses(modules,model,[
        ("native",native_rows,masks[0::4],False),("semantic",semantic_rows,masks[1::4],False),
        ("hard_negative",negative_rows,masks[2::4],True),("locality",locality_rows,masks[3::4],True),
    ],moe_cs,moe_rs,eqrs,suffix_device,STRICT_MODE)
    route_losses["L_positive"]=output["native"]+output["semantic"]
    route_losses["L_negative"]=output["hard_negative"]+output["locality"]
    return route_losses,feature_rows


def gradient_diagnostics(modules, losses, state_name, size):
    vectors={};norm_rows=[]
    for index,name in enumerate(LOSSES):
        modules.zero_grad(set_to_none=True)
        losses[name].backward(retain_graph=index < len(LOSSES)-1)
        vectors[name]=flatten_gradients(modules)
        for group,vector in vectors[name].items():
            norm_rows.append({"router_state":state_name,"repository_size":size,
                              "loss":name,"parameter_group":group,
                              "loss_value":float(losses[name].detach()),
                              "gradient_norm":float(vector.norm())})
    modules.zero_grad(set_to_none=True)
    cosine_rows=[]
    for group in next(iter(vectors.values())):
        for left,right in combinations(LOSSES,2):
            cosine,degenerate=zero_safe_cosine(vectors[left][group],vectors[right][group])
            a,b=vectors[left][group],vectors[right][group]
            nonzero=(a!=0)|(b!=0)
            opposite=float(((a*b<0)&nonzero).sum().item()/max(1,int(nonzero.sum())))
            projection=max(0.0,-float(torch.dot(a,b)))/max(float(b.norm()),1e-12)
            cosine_rows.append({"router_state":state_name,"repository_size":size,
                "parameter_group":group,"left_loss":left,"right_loss":right,
                "gradient_cosine":cosine,"degenerate":degenerate,
                "opposite_sign_coordinate_fraction":opposite,
                "conflict_projection_magnitude":projection})
        positive=vectors["L_positive"][group];negative=vectors["L_negative"][group]
        cosine,degenerate=zero_safe_cosine(positive,negative)
        cosine_rows.append({"router_state":state_name,"repository_size":size,
            "parameter_group":group,"left_loss":"positive_aggregate",
            "right_loss":"negative_aggregate","gradient_cosine":cosine,
            "degenerate":degenerate,
            "opposite_sign_coordinate_fraction":float((positive*negative<0).sum().item()/max(1,positive.numel())),
            "conflict_projection_magnitude":max(0.0,-float(torch.dot(positive,negative)))/max(float(negative.norm()),1e-12)})
    del vectors
    return norm_rows,cosine_rows


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-manifest",type=Path,required=True)
    parser.add_argument("--hard-cache-manifest",type=Path,required=True)
    parser.add_argument("--fixed-experts",type=Path,required=True)
    parser.add_argument("--state",action="append",required=True,help="name=checkpoint_directory")
    parser.add_argument("--training-trajectory",type=Path,required=True)
    parser.add_argument("--physical-gpu",type=int,required=True)
    parser.add_argument("--suffix-physical-gpu",type=int,required=True)
    parser.add_argument("--out-root",type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=f"{args.physical_gpu},{args.suffix_physical_gpu}":
        raise RuntimeError("ROUTER_R1_GRADIENT_GPU_VISIBILITY_MISMATCH")
    gradient_root=args.out_root/"gradient_conflict";feature_root=args.out_root/"feature_scale"
    if gradient_root.exists() or feature_root.exists():raise FileExistsError(args.out_root)
    gradient_root.mkdir(parents=True);feature_root.mkdir(parents=True)

    manifest=json.loads(args.train_manifest.read_text());families=manifest["families"]
    family_by_id={row["family_id"]:row for row in families};all_ids=[row["family_id"] for row in families]
    first=json.loads(args.training_trajectory.read_text().splitlines()[0]);family_ids=first["family_ids"]
    if len(family_ids)!=8 or any(value not in family_by_id for value in family_ids):
        raise RuntimeError("ROUTER_R1_GRADIENT_BATCH_MISMATCH")
    hard_manifest=json.loads(args.hard_cache_manifest.read_text());hard={row["family_id"]:row for row in hard_manifest["records"]}
    expert_bank,expert_manifest=fixed_experts(args.fixed_experts)
    states={item.split("=",1)[0]:Path(item.split("=",1)[1]) for item in args.state}
    if tuple(states)!=("step_0000","step_0080","step_0320","step_0640"):
        raise RuntimeError("ROUTER_R1_GRADIENT_STATE_SET")
    model,_bank,suffix_device=load_training_model(args.physical_gpu,args.suffix_physical_gpu,STRICT_MODE)
    base_hash=state_weight_hash(model)
    for parameter in model.llava_model.parameters():parameter.requires_grad_(False)
    cached_rows=load_diagnostic_rows(family_ids,family_by_id,hard,model.lm_device)
    batch_manifest={"protocol":PROTOCOL,"selection":"formal_training_trajectory_step_1_family_ids",
        "family_ids":family_ids,"semantic_category":"textual",
        "negative_category":"same_image_different_question","train_only":True,
        "validation_used":False,"heldout_used":False,"record953_used":False,"sealed_blind_used":False,
        "repository_sizes":list(SIZES),"router_states":list(states),
        "fixed_expert_hash":expert_manifest["global_expert_hash"]}
    write_json(gradient_root/"diagnostic_batch_manifest.json",batch_manifest)

    norm_rows=[];cosine_rows=[];feature_rows=[];state_ledger=[]
    for state_name,path in states.items():
        modules=LiveEditMedicalModules(LiveEditMedicalConfig(
            learning_rate=5e-5,source_training_continuation_mode=STRICT_MODE)).to(model.lm_device).float()
        state,checkpoint_manifest=load_safe_state(path);modules.load_state_dict(state,strict=True)
        configure_router_only(modules);modules.train()
        before=tensor_hashes({name:value.detach().cpu() for name,value in modules.state_dict().items()})
        for size in SIZES:
            losses,features=losses_and_features(modules,model,suffix_device,family_ids,all_ids,
                                               cached_rows,expert_bank,size)
            current_norms,current_cosines=gradient_diagnostics(modules,losses,state_name,size)
            norm_rows.extend(current_norms);cosine_rows.extend(current_cosines)
            feature_rows.extend({"router_state":state_name,**row} for row in features)
            if any(parameter.grad is not None for parameter in modules.parameters()):
                raise RuntimeError("ROUTER_R1_GRADIENT_NOT_RESET")
            del losses
            torch.cuda.empty_cache()
            print(json.dumps({"event":"router_r1_gradient_feature","state":state_name,
                              "repository_size":size}),flush=True)
        after=tensor_hashes({name:value.detach().cpu() for name,value in modules.state_dict().items()})
        if before!=after:raise RuntimeError(f"ROUTER_R1_GRADIENT_STATE_MUTATION:{state_name}")
        state_ledger.append({"router_state":state_name,"checkpoint":str(path),
                             "checkpoint_step":checkpoint_manifest.get("step"),"hashes_unchanged":True})
        del modules,state;torch.cuda.empty_cache()
    if state_weight_hash(model)!=base_hash or bank_manifest()["sha256"]!=EXPECTED_BANK:
        raise RuntimeError("ROUTER_R1_GRADIENT_BASE_OR_BANK_MUTATION")

    write_jsonl(gradient_root/"gradient_norms.jsonl",norm_rows)
    write_jsonl(gradient_root/"gradient_cosines.jsonl",cosine_rows)
    aggregate={}
    for group in sorted({row["parameter_group"] for row in cosine_rows}):
        rows=[row for row in cosine_rows if row["parameter_group"]==group
              and row["left_loss"]=="positive_aggregate"]
        values=[row["gradient_cosine"] for row in rows if not row["degenerate"]]
        aggregate[group]={"measurements":len(rows),"nondegenerate":len(values),
            "negative_count":sum(value<0 for value in values),
            "mean_cosine":sum(values)/len(values) if values else None,
            "persistent_conflict":bool(values and sum(value<0 for value in values)>=math.ceil(.75*len(values)))}
    write_json(gradient_root/"conflict_aggregate.json",{
        "protocol":PROTOCOL,"positive_negative_by_group":aggregate,
        "optimizer_created":False,"optimizer_step_performed":False,"all_gradients_reset":True,
        "states_unchanged":state_ledger,"heldout_used":False,"record953_used":False,"sealed_blind_used":False})
    (gradient_root/"GRADIENT_CONFLICT_REPORT.md").write_text(
        "# Router-R1 Read-Only Gradient Conflict Report\n\n"
        "No optimizer was created or stepped. Persistent conflict requires negative cosine in at least 75% "
        "of non-degenerate state/repository measurements. See `conflict_aggregate.json`.\n")

    key_rows=[];score_rows=[]
    for row in feature_rows:
        key_rows.append({key:value for key,value in row.items() if key.startswith("norm_") or key in (
            "router_state","repository_size","family_id","candidate_count","target_candidate")})
        score_rows.append({key:value for key,value in row.items() if key not in key_rows[-1] or key in (
            "router_state","repository_size","family_id","candidate_count","target_candidate")})
    write_jsonl(feature_root/"key_norms.jsonl",key_rows)
    write_jsonl(feature_root/"cosine_and_score_decomposition.jsonl",score_rows)
    inflation={}
    for state_name in states:
        inflation[state_name]={}
        for size in SIZES:
            rows=[row for row in feature_rows if row["router_state"]==state_name and row["repository_size"]==size]
            inflation[state_name][str(size)]={
                "mean_candidate_count":sum(row["candidate_count"] for row in rows)/len(rows),
                "target_recall":sum(row["target_candidate"] for row in rows)/len(rows),
                "mean_input_visual_norm":sum(row["norm_input_visual"] for row in rows)/len(rows),
                "mean_target_visual_norm":sum(row["norm_target_edit_visual"] for row in rows)/len(rows),
                "mean_distractor_visual_norm":sum(row["norm_distractor_visual_mean"] for row in rows)/len(rows),
                "mean_sentinel_visual_norm":sum(row["norm_sentinel_visual"] for row in rows)/len(rows),
                "mean_visual_target_cosine":sum(row["visual_target"]["cosine"] for row in rows)/len(rows),
                "mean_visual_sentinel_cosine":sum(row["visual_sentinel"]["cosine"] for row in rows)/len(rows),
                "mean_visual_distractor_cosine":sum(
                    sum(item["cosine"] for item in row["visual_distractors"])/max(1,len(row["visual_distractors"]))
                    for row in rows)/len(rows),
                "mean_visual_target_score":sum(row["raw_visual_target_score"] for row in rows)/len(rows),
                "mean_sentinel_score":sum(row["raw_sentinel_score"] for row in rows)/len(rows),
                "mean_visual_distractor_score":sum(
                    sum(row["raw_visual_distractor_scores"])/max(1,len(row["raw_visual_distractor_scores"]))
                    for row in rows)/len(rows),
            }
    write_json(feature_root/"candidate_inflation_analysis.json",{
        "protocol":PROTOCOL,"aggregates":inflation,
        "interpretation_pending_finalizer":True,"routing_features_normalized_or_temperature_changed":False})
    (feature_root/"FEATURE_SCALE_REPORT.md").write_text(
        "# Router-R1 Feature Scale Diagnostic\n\n"
        "Read-only key norms, cosines, scores, and exact dot-product decompositions were recorded. "
        "No L2 normalization or temperature change was applied.\n")
    print(json.dumps({"status":"ROUTER_R1_GRADIENT_FEATURE_DIAGNOSTICS_COMPLETE",
                      "states":len(states),"repository_sizes":len(SIZES)},sort_keys=True))


if __name__=="__main__":
    main()
