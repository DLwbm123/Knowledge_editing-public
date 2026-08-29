#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from m3bench_repro.data.m3bench_base_questions import expand_metadata,audit_records,canonical_json
def main():
 p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=Path("manifests/llava_med_all_base_questions.jsonl")); a=p.parse_args(); records=expand_metadata(); audit=audit_records(records)
 if not audit["valid"]:
  a.output.parent.mkdir(parents=True,exist_ok=True); a.output.with_suffix(".summary.json").write_text(json.dumps(audit,indent=2)+"\n"); raise SystemExit(json.dumps(audit,indent=2))
 a.output.parent.mkdir(parents=True,exist_ok=True); data="".join(canonical_json(r)+"\n" for r in records); a.output.write_text(data,encoding="utf-8"); h=hashlib.sha256(data.encode()).hexdigest(); summary={**audit,"manifest":str(a.output),"sha256":h}; a.output.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n"); a.output.with_suffix(".sha256").write_text(h+"  "+a.output.name+"\n"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
