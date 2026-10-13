"""Fixed mechanism audit of completed three-point margin counterfactuals."""
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get('RUN_ROOT', '/tmp'))
SOURCE = Path(os.environ.get('SOURCE_ROOT', '/tmp'))
REPLAY = Path(os.environ.get('REPLAY_ROOT', '/tmp'))
TRANSFER = Path(os.environ.get('TRANSFER_ROOT', '/tmp'))


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    path = ROOT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def consumer_id(row):
    return tuple(row[k] for k in ['mode', 'prefix', 'edit', 'task', 'input_id'])


def summarize(rows):
    return dict(occurrences=len(rows),unique_inputs=len({r['input_id'] for r in rows}),
        unique_input_winner_pairs=len({(r['input_id'],r['R0_winner']) for r in rows}),
        tasks=dict(Counter(r['task'] for r in rows)),origins=dict(Counter(r['origin'] for r in rows)),
        outcomes=dict(Counter(r['outcome'] for r in rows)),
        same_source=sum(r['same_source'] is True for r in rows),
        same_image=sum(r['same_image'] is True for r in rows),
        margins=[min(r['q'] for r in rows),max(r['q'] for r in rows)] if rows else None,
        retained_by_lexical=sum(r['protected'] for r in rows))


def selfcheck():
    row=dict(input_id='x',R0_winner='e',task='T1G',origin='unmatched',outcome='missing',same_source=True,same_image=False,q=0.02,protected=False)
    summary=summarize([row,row])
    assert summary['occurrences']==2 and summary['unique_inputs']==1 and summary['outcomes']['missing']==2
    a=dict(mode='sequential',prefix=12,edit='e',task='T1G',input_id='x')
    assert consumer_id(a)!=consumer_id(dict(a,prefix=24))


def main():
    started=time.time();selfcheck()
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    argv=Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert not any(k in argv.lower() for k in ['wangbomin','knowledge_editing','scope'])
    assert read(REPLAY/'RUN_STATUS.json')['status']=='COMPLETE'
    assert read(REPLAY/'public/FINAL_EXECUTION_AUDIT.json')['tau0_reproduction']=='PASS'
    assert read(TRANSFER/'RUN_STATUS.json')['status']=='COMPLETE'
    sys.path.insert(0,str(SOURCE))
    from judge_protocol import read_scores
    scores=read_scores(SOURCE)
    provenance={}
    for row in read(TRANSFER/'private/POINTWISE_PROVENANCE.json'):
        key=tuple(row[k] for k in ['phase','mode','prefix','task','input_id','winner'])
        value={k:row[k] for k in ['origin','same_image','same_source']}
        assert key not in provenance or provenance[key]==value
        provenance[key]=value
    routes={}
    for row in read(REPLAY/'private/COUNTERFACTUAL_ROUTES.json'):
        key=(row['phase'],row['tau'],*row['identity']);assert key not in routes
        routes[key]=row
    aggregate=read(REPLAY/'public/MARGIN_AGGREGATES.json');queue=read(SOURCE/'QUEUE.json')
    metrics=[];out={};denominators={}
    for phase,expected in [('DEV',989),('REG',1058)]:
        txt={consumer_id(row):row for job in queue if job['phase']==phase and job['method']=='TXT'
             for path in (SOURCE/'jobs'/job['id']).rglob('CONSUMERS.json') for row in read(path)}
        assert len(txt)==expected;denominators[phase]=expected
        out[phase]={}
        for tau in [-0.1,0.1]:
            changed=[]
            for identity,b in txt.items():
                c=routes[(phase,tau,*identity)]
                if c['winner']==b['route']['logical_edit_id']:continue
                zero=routes[(phase,0.0,*identity)]
                assert zero['judge_key']==b['judge_key']
                d=b['route_diagnostics'];assert not d['text_guard'] and d['R0_activated']
                winner=b['route']['nearest_logical_edit_id']
                source=provenance[(phase,b['mode'],b['prefix'],b['task'],b['input_id'],winner)]
                before=scores.get(b['judge_key']);after=scores.get(c['judge_key'])
                outcome='missing' if before is None or after is None else 'wrong_correct' if not before and after else 'correct_wrong' if before and not after else 'both_correct' if after else 'both_wrong'
                x=dict(phase=phase,tau=tau,mode=b['mode'],prefix=b['prefix'],input_id=b['input_id'],R0_winner=winner,task=b['task'],q=d['q'],protected=bool(d['text_guard']),outcome=outcome,**source)
                if tau>0:assert 0<=d['q']<tau and c['winner'] is None
                else:assert tau<=d['q']<0 and c['winner']==winner
                changed.append(x);metrics.append(x)
            assert len(changed)==aggregate['by_phase'][phase][str(tau)]['route_changes_vs_executed_tau0']
            damage=[x for x in changed if x['outcome']=='correct_wrong']
            positive_damage=[x for x in damage if x['task'] in ['T0','T1G','T2G']]
            out[phase][str(tau)]=dict(all_changes=summarize(changed),known_damage=summarize(damage),
                known_positive_damage=summarize(positive_damage),positive_damage_by_panel={
                    mode+'/'+str(prefix):summarize([x for x in positive_damage if (x['mode'],x['prefix'])==(mode,prefix)])
                    for mode,prefix in sorted({(x['mode'],x['prefix']) for x in positive_damage})})
    assert len(routes)==3*2047
    write('private/CHANGED_ROUTE_MECHANISMS.json',metrics)
    write('public/MECHANISM_AGGREGATES.json',dict(status='COMPLETE_EXPOSED_DIAGNOSTIC',full_denominators=denominators,
        by_phase=out,no_threshold_selected=True,no_new_verified_scope=True,independent_CONFIRM=False,
        image_hash_difference_not_patient_independence=True,source_group_not_patient_independence=True))
    seconds=time.time()-started;manifest=read(ROOT/'RUN_MANIFEST.json')
    assert seconds<60 and time.time()<manifest['deadline_epoch'] and not (ROOT/'STOP').exists()
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<32*1024**2
    write('public/FINAL_EXECUTION_AUDIT.json',dict(status='COMPLETE',CPU_wall_seconds=seconds,GPU_hours=0,
        new_Judge=0,training_steps=0,new_generation=0,consumers=2047,route_reconstruction='PASS',unique_denominator='PASS'))
    write('PROCESS_RECEIPT.json',dict(pid=os.getpid(),argv=argv,started_epoch=started,CUDA_VISIBLE_DEVICES=''))
    write('RUN_STATUS.json',dict(status='COMPLETE',phase='CLOSED_MECHANISM_AUDIT',epoch=time.time()))
    print(json.dumps({p:{t:x['known_positive_damage'] for t,x in ts.items()} for p,ts in out.items()}))


if __name__=='__main__':
    if '--selfcheck' in sys.argv:
        selfcheck();print('PASS: duplicate-prefix denominator and missing retention')
    else:
        try:main()
        except Exception as error:
            write('FAILURE.json',dict(error=str(error),traceback=traceback.format_exc()))
            write('RUN_STATUS.json',dict(status='FAILED_PRESERVED'))
            raise
