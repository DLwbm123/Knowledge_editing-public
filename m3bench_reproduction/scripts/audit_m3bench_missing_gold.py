#!/usr/bin/env python3
from __future__ import annotations
import json,hashlib
from pathlib import Path
from m3bench_repro.data.m3bench_base_questions import expand_metadata,sha256_path
def main():
 rows=expand_metadata(); missing=[r for r in rows if not r["gold_answer"].strip()]; target={"dataset":"SLAKE","image_id":"xmlab281","question_id":"xmlab281_3","global_index":4467}; ok=len(missing)==1 and all(missing[0][k]==v for k,v in target.items()); r=missing[0] if missing else {}; from PIL import Image
 image_ok=False
 if r:
  try:
   with Image.open(r["absolute_image_path"]) as im: im.verify(); image_ok=True
  except Exception: pass
 out={"passed":ok and bool(r.get("question"," ").strip()) and image_ok,"missing_records":missing,"target":target,"image_opens":image_ok,"metadata_sha256":{"slake":sha256_path(Path("external/M3Bench/metadata/selected_processed_files/slake_metadata.csv")),"vqarad":sha256_path(Path("external/M3Bench/metadata/selected_processed_files/vqarad_metadata.csv"))},"policy":"no_imputation"};Path("outputs/gates").mkdir(parents=True,exist_ok=True);Path("outputs/gates/missing_gold_audit.json").write_text(json.dumps(out,indent=2)+"\n");Path("outputs/gates/missing_gold_audit.md").write_text(f"# Missing gold audit\n\nPassed: `{out['passed']}`. Exactly one released empty gold: `xmlab281_3`; no replacement used.\n");print(json.dumps(out,indent=2));
 if not out["passed"]:raise SystemExit(1)
if __name__=="__main__":main()
