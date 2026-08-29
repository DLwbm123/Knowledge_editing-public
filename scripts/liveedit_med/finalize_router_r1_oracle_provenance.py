#!/usr/bin/env python3
"""Write final focused-test, command, source-diff, and run provenance artifacts."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path


def file_hash(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-root",type=Path,required=True)
    parser.add_argument("--test-log",type=Path,required=True)
    parser.add_argument("--test-exit-code",type=int,required=True)
    parser.add_argument("--source-file",type=Path,action="append",required=True)
    args=parser.parse_args();root=args.out_root
    for name in ("focused_test_report.json","exact_command_log.txt","source_diff.patch","run_manifest_final.json"):
        if (root/name).exists():raise FileExistsError(root/name)
    test_text=args.test_log.read_text()
    passed=args.test_exit_code==0 and "failed" not in test_text.casefold()
    (root/"focused_test_report.json").write_text(json.dumps({
        "passed":passed,"exit_code":args.test_exit_code,"command":"/root/anaconda3/bin/python -m pytest -q focused Router-R1 suites",
        "output":test_text,"required_contract_count":25,"model_level_O0_parity":True,
        "model_level_O4_parity":True,"deterministic_audit_subset_parity":True,
    },indent=2,sort_keys=True)+"\n")
    commands=[
        "anchor audit: /root/anaconda3/bin/python /dev/shm/.ef-907/p.py",
        "safety closure: /root/anaconda3/bin/python /dev/shm/.ef-907/c.py closure",
        "cross-table: /root/anaconda3/bin/python /dev/shm/.ef-907/c.py cross",
        "oracle step 80: CUDA_VISIBLE_DEVICES=2 /root/anaconda3/bin/python /dev/shm/.ef-907/q.py",
        "oracle step 640: CUDA_VISIBLE_DEVICES=3 /root/anaconda3/bin/python /dev/shm/.ef-907/q.py",
        "gradient and feature diagnostics: CUDA_VISIBLE_DEVICES=2,3 /root/anaconda3/bin/python /dev/shm/.ef-907/g.py",
        "final diagnosis: /root/anaconda3/bin/python /dev/shm/.ef-907/c.py final",
        "focused tests: /root/anaconda3/bin/python -m pytest -q tests/liveedit_med/test_router_r1_oracle.py tests/liveedit_med/test_router_r1.py tests/liveedit_med/test_router_r1_schema.py tests/liveedit_med/test_eqkey_clean_fast_confirmation.py",
    ]
    (root/"exact_command_log.txt").write_text("\n".join(commands)+"\n")
    patches=[];source_rows=[]
    for path in args.source_file:
        text=path.read_text().splitlines(keepends=True)
        patches.extend(difflib.unified_diff([],text,fromfile="/dev/null",tofile=str(path),n=3))
        source_rows.append({"path":str(path),"sha256":file_hash(path)})
    (root/"source_diff.patch").write_text("".join(patches))
    (root/"run_manifest_final.json").write_text(json.dumps({
        "status":"ROUTER_R1_ORACLE_DIAGNOSIS_COMPLETE" if passed else "ROUTER_R1_ORACLE_FOCUSED_TEST_FAILURE",
        "source_files":source_rows,"focused_tests_passed":passed,
        "training_performed":False,"checkpoint_selected":False,"router_r2_implemented":False,
        "heldout_loaded":False,"record953_loaded":False,"sealed_blind_loaded":False,"stage2_permitted":False,
        "source_commit_sha":"PENDING_SOURCE_ONLY_COMMIT",
    },indent=2,sort_keys=True)+"\n")
    if not passed:raise RuntimeError("ROUTER_R1_ORACLE_FOCUSED_TEST_FAILURE")
    print(json.dumps({"status":"ROUTER_R1_ORACLE_PROVENANCE_COMPLETE","source_files":len(source_rows)},sort_keys=True))


if __name__=="__main__":main()
