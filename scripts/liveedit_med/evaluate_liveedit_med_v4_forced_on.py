#!/usr/bin/env python3
"""Stage F forced-on generated-expert evaluation for external record 953."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
for item in (ROOT,ROOT/"scripts"):
    if str(item) not in sys.path: sys.path.insert(0,str(item))

from methods.liveedit_med.llavamed_adapter import Layer21ResidualHook, resolve_layer21_block
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.source_ops import apply_low_rank_expert_residual
from methods.liveedit_med.trainer import LiveEditMedicalConfig,LiveEditMedicalModules
from scripts.engram.lora_positive_control_utils import positive_control_match
from scripts.engram.run_engram_v2_one_shot_natural_generation_rescue import full_generation_parity
from scripts.engram.run_engram_v2_stage0_generation_audit import apply_prefix,clone_sample_with_target,eos_ids,load_model_views_bank
from scripts.engram.run_llavamed_record953_lora_positive_control import CAP,RECORD_ID,TARGET,seed_everything
from scripts.engram.stage0_generation_audit_utils import build_canonical_inputs,manual_greedy_trace


def sample(image,prompt,target): return {"image_path":[str(image)],"prompt":[prompt],"target":[target]}


@torch.inference_mode()
def capture(model,block,row):
    canonical=build_canonical_inputs(model,row); values=[]; handle=block.register_forward_hook(lambda _m,_a,o: values.append((o[0] if isinstance(o,(tuple,list)) else o).detach()))
    model.llava_model(input_ids=canonical.full_ids,images=canonical.image,attention_mask=torch.ones_like(canonical.full_ids),return_dict=True,use_cache=False);handle.remove()
    if len(values)!=1: raise RuntimeError("LIVEEDIT_MED_CAPTURE_COUNT")
    hidden=values[0]; visual_start=int(torch.where(canonical.full_ids[0]==model.IMAGE_TOKEN_INDEX)[0][0]); visual=hidden[:,visual_start:visual_start+576]; answer_n=int(canonical.target_ids.numel()); question=hidden[:,visual_start+576:hidden.shape[1]-answer_n]; answer=hidden[:,-answer_n:]
    return canonical,visual,question,answer


def generated(model,canonical,aliases):
    trace=manual_greedy_trace(model,canonical,CAP,eos_ids(model),top_k=1);match=positive_control_match(trace["raw_output"],TARGET,eos=trace["stop_reason"]=="eos",cap_hit=trace["cap_hit"],aliases=aliases);return {**trace,"match":match}


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--physical-gpu",type=int,default=2);a=p.parse_args();seed_everything()
    model,views,bank,records=load_model_views_bank(a.physical_gpu);apply_prefix(model,bank,0);_name,block=resolve_layer21_block(model)
    modules=LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float();state,manifest=load_safe_state(a.checkpoint);modules.load_state_dict(state,strict=True);modules.eval()
    target=views[RECORD_ID]["target"]; canonical,visual,question,answer=capture(model,block,target);eqr,evr,moe_c,moe_r=modules.generated_edit(visual.float(),question.float(),answer.float())
    hook=Layer21ResidualHook(block,lambda hidden:apply_low_rank_expert_residual(hidden.float(),moe_c,moe_r,torch.ones(1,1,device=hidden.device),modules.instant_reps_norm).to(hidden.dtype)).install();hook.enabled=True
    row=records[RECORD_ID]; image_root=Path(target["image_path"][0]).parents[1]
    tests={
      "native":target,
      "textual":sample(target["image_path"][0],row["rephrase"],row["alt"]),
      "visual":sample(image_root/row["image_rephrase"],row["src"],row["alt"]),
      "paired":sample(image_root/row["image_rephrase"],row["port_new"][0]["Q&A"]["Question"],row["alt"]),
    }
    aliases=[str(x) for x in row.get("accepted_answers",[])];outputs={}
    for name,value in tests.items(): outputs[name]=generated(model,build_canonical_inputs(model,value),aliases)
    parity=full_generation_parity(model,canonical);hook.remove()
    native=outputs["native"];passed=bool(native["match"]["success"] and parity["passed"])
    result={"label":"LIVEEDIT_GENERATOR_FORCED_ON_PASS" if passed else "LIVEEDIT_GENERATOR_TRANSFER_FAILURE","passed":passed,"checkpoint_manifest":manifest,"outputs":outputs,"native_three_path_parity":parity["passed"],"generated_expert_shapes":{"eqr":list(eqr.shape),"evr":list(evr.shape),"moe_c":list(moe_c.shape),"moe_r":list(moe_r.shape)}}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps({"label":result["label"],"output":native["raw_output"]}))


if __name__=="__main__":main()
