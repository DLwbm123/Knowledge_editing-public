# Public sanitized copy of the fixed runtime source. Private data paths and bindings are placeholders.
import json,os,sys,time,subprocess
from pathlib import Path
phase=sys.argv[1]
run=Path("<RUN_ROOT>")
base=json.loads(Path("<PRIOR_RUN_ROOT>/private/DISPATCH_24E.json").read_text())
base.update(run=str(run),worker="<WORKER_SCRIPT>",entrypoint="<ENTRYPOINT>",phase=phase,code_commit="8e4aec8940d98670687f50fda5bfee978d8e1fb2",gpu="0",gpu_uuid="GPU-97f3d420-f399-c351-e161-927263614dee",stage24e_plan=str(run/"private/COMPILED_PLAN.json"),common_run="<COMMON_RUN_ROOT>",stage20_run="<STAGE20_RUN_ROOT>",stage21A_run="<STAGE21A_RUN_ROOT>",stage21B_run="<STAGE21B_RUN_ROOT>",python="<RUNTIME_PYTHON>",gpu_deadline_epoch=time.time()+({"R2B":1700,"R2C":7000}[phase]),campaign_epoch=time.time())
# worker needs local run bindings from copied plan/stream; preserve private stream via common
cfg=run/"private"/("DISPATCH_"+phase+".json"); cfg.write_text(json.dumps(base,indent=2)+"\n")

ledger=run/"public/GPU_BUDGET_LEDGER.json"; l=json.loads(ledger.read_text()); start=time.time(); l["sessions"].append({"phase":phase,"started_epoch":start,"status":"RUNNING"}); ledger.write_text(json.dumps(l,indent=2)+"\n")
env=dict(os.environ,JOB_CONFIG=str(cfg),JOB_ARGV="[]")
env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",E_CODE_ROOT="<CODE_ROOT>",R2_N="7" if phase=="R2B" else "19",R2_ARM="B1" if phase=="R2B" else "")
log=run/"private"/(phase+".log")
with log.open("w") as f:
 p=subprocess.Popen([base["python"],base["entrypoint"]],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 l["sessions"][-1]["pid"]=p.pid; ledger.write_text(json.dumps(l,indent=2)+"\n")
 rc=p.wait()
end=time.time(); l=json.loads(ledger.read_text()); sess=l["sessions"][-1]; sess.update(ended_epoch=end,seconds=end-start,exit_code=rc,status="COMPLETE" if rc==0 else "FAILED"); l["used_seconds"]=sum(x.get("seconds",0) for x in l["sessions"]); l["status"]="COMPLETE" if rc==0 else "FAILED"; ledger.write_text(json.dumps(l,indent=2)+"\n"); (run/"public"/("EXIT_"+phase+".json")).write_text(json.dumps({"phase":phase,"exit_code":rc,"seconds":end-start,"run_id":"stage24e-r2-20260920-job525"},indent=2)+"\n"); sys.exit(rc)
