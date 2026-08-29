#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from m3bench_repro.inference import LlavaMedAdapter
from m3bench_repro.data.m3bench_base_questions import expand_metadata
def main():
 p=argparse.ArgumentParser(); p.add_argument("--model",required=True,type=Path); p.add_argument("--vision-tower",required=True,type=Path); p.add_argument("--output",type=Path,default=Path("outputs/gates/llava_preprocessing_determinism.json")); a=p.parse_args()
 rows=expand_metadata(); selected=[]
 availability={}
 for dataset in ("SLAKE","VQA-RAD"):
  seen=set()
  for r in rows:
   if r["dataset"]==dataset and r["image_id"] not in seen:
    from PIL import Image
    with Image.open(r["absolute_image_path"]) as im:
     if im.width!=im.height: selected.append(r); seen.add(r["image_id"])
    if len([x for x in selected if x["dataset"]==dataset])==10: break
  availability[dataset]=len([x for x in selected if x["dataset"]==dataset])
  if availability[dataset] < 10:
   # SLAKE's released images are all square (642/642); audit ten square controls
   # because the upstream random pad branch is unreachable for that dataset.
   for r in rows:
    if r["dataset"]==dataset and r["image_id"] not in seen:
     selected.append(r); seen.add(r["image_id"])
     if len([x for x in selected if x["dataset"]==dataset])==10: break
 adapter=LlavaMedAdapter(a.model,a.vision_tower); adapter.load(); results=[]
 for r in selected:
  ts=[]
  for _ in range(3): ts.append(adapter.prepare_inputs(r["absolute_image_path"],r["question"]))
  x=[t["images"] if isinstance(t["images"],torch.Tensor) else t["images"][0] for t in ts]; results.append({"dataset":r["dataset"],"image_id":r["image_id"],"seed":ts[0]["preprocessing_seed"],"image_sha256":ts[0]["image_sha256"],"shape":list(x[0].shape),"finite":all(torch.isfinite(y).all().item() for y in x),"exact":all(torch.equal(x[0],y) for y in x[1:]),"max_abs_diff":max(float((x[0]-y).abs().max()) for y in x[1:])})
 out={"choice":"REPRODUCTION_CHOICE__ORDER_INDEPENDENT_UPSTREAM_PREPROCESSING","non_square_available":availability,"slake_square_only_note":"SLAKE has zero non-square released images; ten square controls cover its reachable preprocessing path.","image_aspect_ratio":adapter.model.config.image_aspect_ratio,"mm_use_im_start_end":bool(adapter.model.config.mm_use_im_start_end),"mm_use_im_patch_token":bool(adapter.model.config.mm_use_im_patch_token),"image_processor":{"size":adapter.image_processor.size,"crop_size":adapter.image_processor.crop_size},"results":results,"passed":len(results)==20 and all(x["exact"] and x["finite"] for x in results)}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"passed":out["passed"],"n":len(results)}));
 if not out["passed"]: raise SystemExit(1)
if __name__=="__main__": main()
