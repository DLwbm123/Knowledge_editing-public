#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,random
from collections import defaultdict
from pathlib import Path
def load(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--base",type=Path,default=Path("manifests/llava_med_all_base_questions.jsonl")); p.add_argument("--output",type=Path,default=Path("manifests/llava_gate_b_100_seed42.json")); a=p.parse_args(); rows=load(a.base); rng=random.Random(42); selected=[]
 for ds in ("SLAKE","VQA-RAD"):
  buckets=defaultdict(list)
  for r in rows:
   if r["dataset"]==ds: buckets[str(r.get("modality") or "unknown")].append(r)
  for b in buckets.values(): rng.shuffle(b)
  keys=sorted(buckets); i=0
  while len([x for x in selected if x["dataset"]==ds])<50:
   k=keys[i%len(keys)];
   if buckets[k]: selected.append(buckets[k].pop())
   i+=1
 for i,r in enumerate(selected): r.update({"gate_b_index":i,"selection_seed":42,"selection_strata":{"modality":r.get("modality"),"body_part":r.get("body_part"),"answer_style":"yes_no" if r["gold_answer"].strip().lower() in {"yes","no","y","n"} else "free_form"}})
 core=json.dumps(selected,ensure_ascii=False,sort_keys=True,separators=(",",":")); h=hashlib.sha256(core.encode()).hexdigest()
 for r in selected:r["manifest_sha256"]=h
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(selected,ensure_ascii=False,indent=2)+"\n"); a.output.with_suffix(".sha256").write_text(h+"  "+a.output.name+"\n"); repeat=[r for ds in ("SLAKE","VQA-RAD") for r in [x for x in selected if x["dataset"]==ds][:10]]; Path("manifests/llava_gate_b_repeat20.json").write_text(json.dumps(repeat,ensure_ascii=False,indent=2)+"\n"); print(h)
if __name__=="__main__":main()
