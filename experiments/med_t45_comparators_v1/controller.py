"""Score each completed method/cohort once under the existing CORE contract."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def read(p):return json.loads(Path(p).read_text())


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)


def remote(cfg,code,timeout=120):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000',cfg['remote_python'],'-'],input=code,text=True,capture_output=True,timeout=timeout)
    assert p.returncode==0,p.stderr[-3000:]
    return json.loads(p.stdout)


def main(cfg):
    local=Path(cfg['control']);private=Path(cfg['private_root']);w=Path(cfg['worktree'])
    clock=read(local/'clock.json');signal.alarm(max(1,int(clock['hard_end']-time.time()-30)))
    with (local/'CONTROLLER_ONCE').open('x'):pass
    sys.path.insert(0,str(w/'experiments/med_t5_core_v1'))
    from controller import score
    history=list(cfg['history_roots']);done={};failed={}
    targets=[(m,c) for m in ['BALANCEDIT','GRACE','LORA'] for c in ['T4','T5']]
    while len(done)+len(failed)<len(targets):
        assert time.time()<clock['hard_end']-660
        state=remote(cfg,'from pathlib import Path\nimport json\nr=Path('+repr(cfg['remote_root'])+')\nprint(json.dumps({m:{c:json.loads((r/m/c/"GENERATION_COMPLETE.json").read_text()) if (r/m/c/"GENERATION_COMPLETE.json").exists() else None for c in ["T4","T5"]} | {"failure":json.loads((r/m/"FAILURE.json").read_text()) if (r/m/"FAILURE.json").exists() else None} for m in ["BALANCEDIT","GRACE","LORA"]}))')
        for method,cohort in targets:
            key=method+'_'+cohort
            if key in done or key in failed:continue
            if state[method][cohort]:
                values=remote(cfg,'from pathlib import Path\nimport json\nr=Path('+repr(cfg['remote_root'])+')/'+repr(method)+'/'+repr(cohort)+'\nprint(json.dumps([json.loads(p.read_text()) for p in sorted((r/"outputs").glob("*.json"))]))',300)
                expected=944 if cohort=='T4' else 1752
                assert len(values)==len({v['consumer_id'] for v in values})==expected
                perprivate=private/method/cohort;perprivate.mkdir(parents=True)
                write(perprivate/'OUTPUTS.private.json',values)
                folder=local/method/cohort;folder.parent.mkdir(exist_ok=True)
                report=Path(cfg['report'])/method/cohort;report.mkdir(parents=True,exist_ok=True)
                sub=dict(cfg,private_root=str(perprivate),report=str(report))
                result=score(sub,folder,'method' if cohort=='T4' else 't5_method',values,history)
                write(report/'METHOD.json',dict(method=method,cohort=cohort,method_configuration='../../../PREREG.json',
                    generation=state[method][cohort],precision='FP32 decoder with frozen FP16 multimodal prefill',
                    comparator_training='native only; standard fixed recipe, unlike MDU five-view training',
                    LoRA_activation='always active, not an R0 retrieval' if method=='LORA' else None))
                done[key]=result['status'];history.append(str(folder))
                write(local/'PROGRESS.json',dict(time=time.time(),done=done,failed=failed))
            elif state[method]['failure']:
                failed[key]='GENERATION_FAILED_WITH_PRESERVED_RECEIPT'
                write(private/(key+'_FAILURE.private.json'),state[method]['failure'])
        time.sleep(20)
    write(Path(cfg['report'])/'EXECUTION_AUDIT.json',dict(time=time.time(),clock=clock,scored=done,failed=failed,
        MEND='BLOCKED_NO_VERIFIED_MATCHED_META_EDITOR',main_MDU_results_unchanged=True))
    combined={}
    for cohort in ['T4','T5']:
        original='med_t4_core_v1_20261107' if cohort=='T4' else 'med_t5_core_v1_20261107'
        combined[cohort]={'MDU':read(w/'reports'/original/'RESULTS.json')}
        for method in ['BALANCEDIT','GRACE','LORA']:
            p=Path(cfg['report'])/method/cohort/'RESULTS.json'
            combined[cohort][method]=read(p) if p.exists() else dict(status='INCOMPLETE')
        combined[cohort]['MEND']=dict(status='BLOCKED_NO_VERIFIED_MATCHED_META_EDITOR')
    write(Path(cfg['report'])/'RESULTS.json',dict(status='COMPLETE_AVAILABLE_METHODS_WITH_MEND_BLOCKED',panels=combined))
    sys.path.insert(0,str(w/'experiments/med_t5_data_v1'))
    from job import publish
    publish(cfg)
    write(local/'COMPLETE.json',dict(time=time.time(),scored=done,failed=failed,MEND_pending=True))


if __name__=='__main__':
    cfg=read(os.environ['COMPARE_LOCAL_CONFIG'])
    try:main(cfg)
    except BaseException:
        write(Path(cfg['control'])/'FAILURE.json',dict(time=time.time(),error=traceback.format_exc()));raise
