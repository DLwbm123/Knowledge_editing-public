"""Reuse frozen MED/MCE/MDU verdicts; never issue judge calls."""
import json
import os
import random
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

root = Path(os.environ['LOCAL_SCORE_ROOT'])
w = Path(os.environ['WORKTREE'])
deadline = float(os.environ['EXPERIMENT_DEADLINE'])
sys.path[:0] = [str(w/'experiments'/x) for x in ('med_medinit_101_v1','med_medinit_loss_v1','med_common25_loss_ablation_v1','med_fixed_depth_v1')]
import summary

FIELDS = ('core_correct','full_supported','unsupported_expansion','extra_contradiction')
ROLES = ('T0','FIT','T1G','T2G','T1L','T2L')


def read(path):
    return json.loads(Path(path).read_text())


def write(name, obj):
    p = root/name
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False))
    t.chmod(0o600)
    t.replace(p)


def baseline():
    p = Path('/private/tmp/.r.20261104.102prep/BASELINE.private.json')
    base = read(p)
    d = dict(base['decisions'])
    conflicts = set()
    for suffix, expected in (('102',97), ('103',8)):
        folder=Path('/private/tmp/.r.20261104.'+suffix)
        for name in ('WORKER_EXIT.json','GUARD_EXIT.json'):
            if read(folder/name)['code'] != 0: raise ValueError('previous score dependency failed')
        frozen=read(folder/'SUPPLEMENT_INPUT.private.json'); done=read(folder/'COMPLETE.json'); new=read(folder/'DECISIONS.json')
        if len(frozen['payloads']) != expected or done['target_payloads'] != expected or new['unknown'] or set(new['decisions']) != set(frozen['payloads']):
            raise ValueError('supplement coverage mismatch')
        if set(d) & set(new['decisions']): raise ValueError('duplicate verdict')
        for key,v in new['decisions'].items():
            if v.get('opaque_query_id')!=key or any(type(v.get(f)) is not bool for f in FIELDS): raise ValueError('invalid verdict')
            if v['full_supported'] and (not v['core_correct'] or v['unsupported_expansion'] or v['extra_contradiction']): raise ValueError('invalid verdict consistency')
        d.update(new['decisions'])
        conflicts.update(frozen['categories']['conflict_unknown'])
    roles=read(w/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json')
    result=summary.compare(base['consumers'],d,roles,base['routing'])
    expected=read(w/'reports/med_medinit_101_supplement_v1_20261104/RESULTS.json')
    if result['groups'] != expected['groups']: raise ValueError('old supplemented results differ')
    routing=base['routing']['records']
    selected={r['query_id'] for r in routing if r['group']=='original25'}
    if len(selected)!=392: raise ValueError('selected route count differs')
    rows=[r for r in base['consumers'] if r['query_id'] in selected]
    if len(rows)!=392*3: raise ValueError('baseline selected coverage mismatch')
    return rows,d,conflicts,roles


def aggregate(rows, decisions, roles):
    index={(r['arm'],r['query_id']):r['key'] for r in rows}
    if len(index)!=len(rows):raise ValueError('duplicate arm/query')
    arms=('MED','MCE','MDU','BOUNDARY')
    sets=[{q for a,q in index if a==arm} for arm in arms]
    if any(len(s)!=392 or s!=sets[0] for s in sets):raise ValueError('unmatched event sets')
    out={}
    for role in ROLES:
        entries=[(o,q) for o in range(1,151) for q in roles['roles'][str(o)][role] if q in sets[0]]
        out[role]={}
        for field in ('core_correct','full_supported'):
            values={a:[(o,decisions.get(index[a,q],{}).get(field)) for o,q in entries] for a in arms}
            panel={'arms':{a:summary.summarize(v) for a,v in values.items()},'pairs':{}}
            for left,right in (('MED','BOUNDARY'),('MCE','BOUNDARY'),('MDU','BOUNDARY')):
                counts={'both_correct':0,'both_incorrect':0,'gain':0,'loss':0,'unknown':0}
                by_owner={}
                for (owner,a),(_,b) in zip(values[left],values[right]):
                    if a is not None and b is not None: by_owner.setdefault(owner,[]).append(int(b)-int(a))
                    if a is None or b is None: counts['unknown']+=1
                    elif a and b:counts['both_correct']+=1
                    elif not a and not b:counts['both_incorrect']+=1
                    elif b:counts['gain']+=1
                    else:counts['loss']+=1
                counts['edit_bootstrap_95']=None
                if entries and counts['unknown']==0:
                    deltas=[statistics.mean(v) for v in by_owner.values()]
                    rng=random.Random(20261105)
                    samples=sorted(100*statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(10000))
                    counts['edit_bootstrap_95']=[samples[249],samples[9749]]
                panel['pairs'][left+'_to_'+right]=counts
            out[role][field]=panel
    return out


def ssh(script):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pro5000','python3','-'],input=script,text=True,capture_output=True,timeout=180)
    if p.returncode:raise RuntimeError('remote dependency failed: '+p.stderr[-600:])
    if len(p.stdout.encode())>20_000_000:raise ValueError('packet too large')
    return json.loads(p.stdout)


def main():
    rows,decisions,conflicts,roles=baseline()
    chosen=sorted(r['consumer_id'] for r in rows if r['arm']=='MED')
    if len(set(chosen))!=392:raise ValueError('selected consumers differ')
    write('HISTORY_INDEX_FROZEN.json',{'status':'FROZEN','events':392,'new_judge_authorized':False,'historical_conflicts':len(conflicts)})
    if os.environ.get('CHECK_ONLY')=='1':
        write('ADMISSION_CHECK.json',{'status':'BASELINE_REPRODUCED','events':392,'new_judge_attempts':0})
        return
    remote=os.environ['EXPERIMENT_ROOT']; old=os.environ['ADDITIONS_ROOT']; fixed=os.environ['FIXED_ROOT']
    while True:
        if time.time()>=deadline-180:raise TimeoutError('original deadline')
        state=ssh("import json\nfrom pathlib import Path\nr=Path(%r)\np=r/'EVAL_GUARD_EXIT.json'\nprint(json.dumps({'failed':any((r/n).exists() for n in ('TRAIN_STOP.json','EVAL_STOP.json','TRAIN_GUARD_FAILURE.json','EVAL_GUARD_FAILURE.json')),'done':p.is_file(),'code':json.loads(p.read_text())['code'] if p.is_file() else None}))\n"%remote)
        if state['failed'] or state['done'] and state['code']!=0:raise RuntimeError('GPU dependency failed')
        if state['done']:break
        write('DEPENDENCY_WAIT.json',{'time':time.time()});time.sleep(20)
    script="import json,sys\nfrom pathlib import Path\nr=Path(%r);old=Path(%r);sys.path[:0]=[str(r/'tools'),str(Path(%r)/'tools')]\nimport packet\npacket.build(r,old,%r)\nprint((r/'SCORE_PACKET_PRIVATE.json').read_text())\n"%(remote,old,fixed,chosen)
    packet=ssh(script)
    if packet.get('status')!='SCORE_PACKET_FROZEN' or len(packet['consumers'])!=784:raise ValueError('packet not frozen')
    history={(r['arm'],r['consumer_id']):r for r in rows}
    added=[]
    for row in packet['consumers']:
        oldrow=history[('MED',row['consumer_id'])]
        if any(row[k]!=oldrow[k] for k in ('consumer_id','query_id','owner','role','route')):raise ValueError('event identity mismatch')
        if row['arm']=='MED':
            if row['key']!=oldrow['key']:raise ValueError('MED payload mismatch')
        elif row['arm']=='CANDIDATE':
            row=dict(row,arm='BOUNDARY');added.append(row)
        else:raise ValueError('invalid arm')
    if len(added)!=392:raise ValueError('candidate coverage mismatch')
    rows+=added
    main_result=aggregate(rows,decisions,roles)
    conservative=aggregate(rows,{k:v for k,v in decisions.items() if k not in conflicts},roles)
    candidate_keys={r['key'] for r in added}
    result={'status':'PAIRED_DEV_SUMMARY','events':392,'panels':main_result,'conflicts_unresolved':conservative,
            'scoring':{'candidate_unique_payloads':len(candidate_keys),'reused_valid':len(candidate_keys & decisions.keys()),'unknown':len(candidate_keys-decisions.keys()),'new_judge_attempts':0}}
    write('RESULTS.json',result)
    evidence=ssh("import json\nfrom pathlib import Path\nr=Path(%r)\nn=Path(%r)\nfiles=['TRAIN_FROZEN.json','PREDICTIONS_FROZEN.json','TRAIN_GUARD_EXIT.json','EVAL_GUARD_EXIT.json']\nprint(json.dumps({k:json.loads((r/k).read_text()) for k in files}|{'combined':json.loads((n/'COMBINED_EXIT.json').read_text()) if (n/'COMBINED_EXIT.json').exists() else None}))\n"%(remote,os.environ['NEUTRAL_ROOT']))
    if evidence['TRAIN_FROZEN.json'].get('updates')!=4000 or evidence['PREDICTIONS_FROZEN.json'].get('generated')!=392 or any(evidence[k].get('code')!=0 for k in ('TRAIN_GUARD_EXIT.json','EVAL_GUARD_EXIT.json')):
        raise ValueError('actual runtime evidence differs')
    out=w/'reports/med_mdu_boundary_average_v1_20261105'
    with (out/'RESULTS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    lines=['# 边界平均 MDU：固定25专家开发对照','','固定392个实际路由事件，原MED、MCE、MDU及新候选同支持；0新增Judge，未知保留。','']
    for role in ROLES:
        panel=main_result[role]['core_correct']['arms']
        lines.append(role+' CORE：'+'；'.join(a+' '+('NA' if panel[a]['bounds'] is None else f"{panel[a]['bounds'][0]:.4f}–{panel[a]['bounds'][1]:.4f}%（未知{panel[a]['unknown']}）") for a in ('MED','MCE','MDU','BOUNDARY'))+'。')
        lines.append('')
    lines+=['未知赋值界限不是置信区间；这是已曝光开发面板。完整CORE/FULL和配对统计见RESULTS.json，历史冲突敏感性另列；T1L无支持时标NA。','']
    with (out/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(lines))
    with (out/'EXECUTION_AUDIT.json').open('x') as f:json.dump({'training_updates':4000,'actual_generation':392,'fallback':1756,'train_guard_exit':0,'eval_guard_exit':0,'combined_guard':evidence['combined'],'new_judge_attempts':0,'historical_attempt_coverage_complete':False},f,ensure_ascii=False,indent=2)
    write('REPORT_COMPLETE.json',{'status':'REPORT_WRITTEN','events':392})
    write('COMPLETE.json',{'status':'PAIRED_DEV_SUMMARY','events':392,'new_judge_attempts':0})


if __name__=='__main__':
    code=0
    try: main()
    except BaseException:
        code=1;write('DRIVER_FAILURE.private.json',{'traceback':traceback.format_exc()});raise
    finally: write('DRIVER_EXIT.json',{'code':code,'time':time.time()})
