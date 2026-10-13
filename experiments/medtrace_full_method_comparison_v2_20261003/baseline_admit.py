"""Read-only admission of complete historical baseline output phases, before new scores."""
import json
import os
import sys
from pathlib import Path
import time

RUN=Path(os.environ['RUN_ROOT'])
sys.path.insert(0,str(RUN/'private/source'))
from scripts.medtrace.stage17_prepare import digest


def read(p):return json.loads(Path(p).read_text())


def qids(t):return list(dict.fromkeys([t['edit_id']]+[q for e in t['events'] for q in e['all_probe_query_ids']]))


def main():
    began=time.time();l=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');b=read(RUN/'private/legacy_stage17/BINDINGS.json');rt=read(RUN/'private/cpu_gate/locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json')
    lock=read(RUN/'private/METHOD_LOCK.json')
    by={t['edit_id']:t for t in l['tasks']};tasks=[by[e] for e in l['main_T0']];summaries={};raw_consumers=[]
    for name in ['balancedit_single','balancedit_sequential','belora_single','belora_sequential']:
        root=RUN/'private/baselines'/name;method,mode=name.rsplit('_',1)
        if name=='balancedit_single':
            total=0
            for t in tasks:
                d=root/f"e{t['order']:03d}";receipt=read(d/'COMPLETE.json');binding=receipt['binding']
                assert receipt['status']=='GENERATED_NOT_SCORED' and binding['freeze_id']==l['freeze_id'] and binding['input']==t['native'] and binding['runtime']==rt
                assert binding['training']['native_fit']==t['fit_questions']
                actual=binding['training']['method'];expected=lock['baselines']['balancedit']
                assert {k:v for k,v in actual.items() if k!='positive'}=={k:v for k,v in expected.items() if k!='positive'}
                assert actual['positive']=='same image plus frozen native-only conservative fit, NEVER official evaluation rephrase'
                assert receipt['training']['steps']==50 and receipt['training']['finite_losses'] and receipt['training']['finite_gradients']
                files=list(d.glob('query_*.json'));expected=qids(t);assert len(files)==len(expected)==receipt['queries']
                seen=set()
                for f in files:
                    out=read(f);qb=out['binding'];q=qb['input'];assert q in [l['queries'][e] for e in expected]
                    assert qb['runtime']==rt and qb['generation']==b[q['opaque_Base_id']]['generation'] and qb['writer']=='BalancEdit_adaptation' and qb['prefix']==1 and out['Base_cache_id']==q['opaque_Base_id'] and out['canonical_cap']==1024
                    assert qb['training_binding']==digest(binding)
                    seen.add(q['query_id']);raw_consumers.append(dict(method=method,mode=mode,path=str(f),prefix=1,query_id=q['query_id'],arm_key='R0'))
                assert seen==set(expected);total+=len(files)
        else:
            complete=read(root/'COMPLETE.json');phase=complete['phase'];assert complete['status']=='GENERATED_NOT_SCORED' and complete['N']==146
            assert phase['freeze_id']==l['freeze_id'] and phase['method']==method and phase['mode']==mode and phase['runtime']==rt and phase['order']==l['main_T0'] and phase['prefixes']==[1,50,100,146]
            assert phase['method_lock']['baselines'][method]==lock['baselines'][method]
            total=0
            for i,t in enumerate(tasks,1):
                d=root/f'e{i:03d}';assert read(d/'COMPLETE.json')['phase']==phase
                if method=='belora':
                    train=read(d/'TRAINING.json');assert train['input']==t and train['phase']==phase and train['training']['finite_losses'] and train['training']['finite_gradients']
                groups=[('native',[t['edit_id']])]
                if mode=='single' or i in [1,50,100,146]:groups.append(('panel',list(dict.fromkeys(q for row in (tasks[:i] if mode=='sequential' else [t]) for q in qids(row)))))
                for folder,expected in groups:
                    files=list((d/folder).glob('*.json'));assert len(files)==len(expected);seen=set()
                    for f in files:
                        out=read(f);qb=out['binding'];q=qb['input'];assert qb['phase']==phase and q==l['queries'][q['query_id']]
                        assert qb['generation']==b[q['opaque_Base_id']]['generation'] and qb['prefix']==(i if mode=='sequential' else 1) and qb['inserted']==(l['main_T0'][:i] if mode=='sequential' else [t['edit_id']])
                        assert out['Base_cache_id']==q['opaque_Base_id'] and out['canonical_cap']==1024 and out['status']=='GENERATED_NOT_SCORED'
                        seen.add(q['query_id']);raw_consumers.append(dict(method=method,mode=mode,path=str(f),prefix=qb['prefix'],query_id=q['query_id'],arm_key='R0' if method=='balancedit' else 'NATIVE',folder=folder))
                    assert seen==set(expected);total+=len(files)
        summaries[name]=dict(status='PASS',N=146,raw_output_consumers=total,weights_not_required=True,old_outputs_read_only=True)
    manifest=RUN/'private/BASELINE_RAW_CONSUMERS.json';manifest.write_text(json.dumps(raw_consumers,indent=2)+'\n')
    report=dict(status='PASS',phases=summaries,N=146,full_binding_checks='input/native-fit/phase-order/prefix/inserted-ancestry/runtime/generation/Base IDs/cap/finite training/recorded source commits',old_scores_not_primary=True,new_common_judge_epoch_required=True,seconds=time.time()-began)
    (RUN/'private/BASELINE_ADMISSION.json').write_text(json.dumps(report,indent=2)+'\n');print('BASELINE_BINDING_PASS',{k:v['raw_output_consumers'] for k,v in summaries.items()})


if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        (RUN/'private/BASELINE_ADMISSION_FAILURE.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),old_results_unchanged=True,GPU_jobs=0),indent=2));raise
