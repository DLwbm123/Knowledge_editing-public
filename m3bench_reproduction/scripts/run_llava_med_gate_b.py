#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,time,traceback
from pathlib import Path
import torch,yaml,transformers
from m3bench_repro.inference import LlavaMedAdapter
def h(x): return hashlib.sha256(x).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,default=Path("manifests/llava_gate_b_100_seed42.json"));p.add_argument("--repeat",type=Path,default=Path("manifests/llava_gate_b_repeat20.json"));p.add_argument("--config",type=Path,default=Path("configs/models/llava_med_generation.yaml"));p.add_argument("--output",type=Path,default=Path("outputs/gates/llava_gate_b"));a=p.parse_args(); c=yaml.safe_load(a.config.read_text()); ch=h(json.dumps(c,sort_keys=True,separators=(",",":")).encode()); rows=json.loads(a.manifest.read_text()); repeats=json.loads(a.repeat.read_text()); a.output.mkdir(parents=True,exist_ok=True); (a.output/"run_config.yaml").write_text(a.config.read_text()); (a.output/"environment.txt").write_text(f"torch={torch.__version__}\ntransformers={transformers.__version__}\n")
 adapter=LlavaMedAdapter(c["model_path"],c["vision_tower_path"]);adapter.load(); modelhash=h("".join(f"{n}:{tuple(v.shape)}" for n,v in adapter.model.state_dict().items()).encode()); gen={k:c[k] for k in ("do_sample","num_beams","use_cache","max_new_tokens")}
 def one(r,rep):
  t=time.monotonic(); b=adapter.prepare_inputs(r["absolute_image_path"],r["question"]); result=adapter.generate_with_result(r["absolute_image_path"],r["question"],gen); ids=list(result.raw_token_ids); eos=adapter.tokenizer.eos_token_id; non=[x for x in ids if x not in {eos,adapter.tokenizer.bos_token_id,adapter.tokenizer.pad_token_id}]; return {**r,"prompt":b["prompt"],"prompt_sha256":h(b["prompt"].encode()),"input_ids":b["input_ids"][0].tolist(),"image_token_count":int((b["input_ids"]==-200).sum()),"image_token_positions":[i for i,x in enumerate(b["input_ids"][0].tolist()) if x==-200],"image_sha256":b["image_sha256"],"preprocessing_seed":b["preprocessing_seed"],"image_tensor_shape":list(b["images"].shape),"image_tensor_dtype":str(b["images"].dtype),"image_tensor_finite":bool(torch.isfinite(b["images"]).all()),"generation_output_contract":"continuation_only","raw_generated_token_ids":ids,"raw_sequence_length":len(ids),"generated_non_special_token_count":len(non),"decoded_with_special_tokens":adapter.tokenizer.decode(ids,skip_special_tokens=False),"model_answer":result.decoded_text,"first_generated_token_id":ids[1] if ids and ids[0]==adapter.tokenizer.bos_token_id else (ids[0] if ids else None),"first_eos_position":next((i for i,x in enumerate(ids) if x==eos),None),"hit_max_new_tokens":len(ids)>=c["max_new_tokens"],"generation_config_sha256":ch,"model_index_sha256":modelhash,"model_source_commit":"30697ca50b5c29a8e955c99330b259776aef27b9","elapsed_seconds":time.monotonic()-t,"repetition_index":rep,"inference_status":"success","exception":None,"judge_status":"not_judged","is_correct":None}
 out=[]
 for r in rows: out.append(one(r,1))
 (a.output/"predictions.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in out))
 rep=[]
 for r in repeats:
  rep.extend([one(r,2),one(r,3)])
 (a.output/"repeated_predictions.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rep))
 allx=out+rep; ok=len(out)==100 and all(x["model_answer"] and x["generated_non_special_token_count"] and x["image_token_count"]==1 and x["image_tensor_finite"] and not x["hit_max_new_tokens"] for x in allx)
 for r in repeats:
  z=[x for x in allx if x["gate_b_index"]==r["gate_b_index"]]; ok &= len(z)==3 and len({tuple(x["raw_generated_token_ids"]) for x in z})==1 and len({x["model_answer"] for x in z})==1
 s={"label":"LLAVA_GATE_B_PASS" if ok else "LLAVA_NEXT_STOP__GATE_B_FAILED","passed":bool(ok),"completed":len(out),"repeat_records":len(rep),"config_sha256":ch,"model_index_sha256":modelhash};(a.output/"summary.json").write_text(json.dumps(s,indent=2)+"\n");print(json.dumps(s));
 if not ok:raise SystemExit(1)
if __name__=="__main__":main()
