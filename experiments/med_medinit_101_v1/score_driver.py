"""Wait for both frozen banks and reuse only verified historic judgments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

root=Path(os.environ['LOCAL_SCORE_ROOT']);w=Path(os.environ['WORKTREE'])
deadline=float(os.environ['EXPERIMENT_DEADLINE']);remote=os.environ['EXPERIMENT_ROOT']

def write(name,value):
    p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False));t.chmod(0o600);t.replace(p)

def ssh(script,timeout=60):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pro5000','python3','-'],input=script,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError('remote dependency failed: '+p.stderr[-600:])
    if len(p.stdout.encode())>45000000:raise ValueError('bounded packet exceeded')
    return json.loads(p.stdout)

def main():
    sys.path[:0]=[str(w/'experiments'/x) for x in ('med_medinit_101_v1','med_medinit_loss_v1','med_full_d150_v1',
        'med_proposition_generalization_v1','med_fixed_depth_v1','med_source_teacher_v1','med150_source_review_v1')]
    import identities,ledger,score_reuse,summary
    prior=identities.consumers(Path(os.environ['DU_PACKET']))
    history=json.loads(Path(os.environ['HISTORY_INDEX']).read_text())
    if history.get('status')!='AVAILABLE_HISTORY_INDEX_FROZEN' or history.get('history_coverage_complete') is not False:
        raise ValueError('history index missing')
    write('HISTORY_INDEX_FROZEN.json',{'status':'AVAILABLE_HISTORY_INDEX_FROZEN','history_coverage_complete':False,'new_attempts':0})
    probe="""import json\nfrom pathlib import Path\nr=Path(%r); names=('TRAIN_GUARD_FAILURE.json','EVAL_GUARD_FAILURE.json','TRAIN_STOP.json','EVAL_STOP.json');p=r/'EVAL_GUARD_EXIT.json'\nprint(json.dumps({'failed':any((r/n).exists() for n in names),'done':p.is_file(),'code':json.loads(p.read_text())['code'] if p.is_file() else None}))\n"""
    for arm in ('MCE','MDU'):
        while True:
            if time.time()>=deadline-180:raise TimeoutError('original deadline')
            state=ssh(probe % str(Path(remote)/arm))
            if state['failed'] or (state['done'] and state['code']!=0):raise RuntimeError('GPU dependency failed')
            if state['done']:break
            write('DEPENDENCY_WAIT.json',{'arm':arm,'time':time.time()});time.sleep(20)
    routing=ssh("import json\nfrom pathlib import Path\np=Path(%r)/'ROUTING_LOCK.private.json'\nprint(p.read_text())\n" % remote)
    if routing.get('status')!='ROUTING_FROZEN' or len(routing['records'])!=2148:raise ValueError('routing lock missing')
    selected=[r['consumer_id'] for r in routing['records'] if r['group']!='med49']
    fallback=[r['consumer_id'] for r in routing['records'] if r['group']=='med49']
    if len(selected)!=1513 or len(fallback)!=635 or len(set(selected+fallback))!=2148:raise ValueError('routing cohorts differ')
    packets={}
    for arm in ('MCE','MDU'):
        script="""import json,sys\nfrom pathlib import Path\nr=Path(%r);old=Path(%r);sys.path[:0]=[str(r/'tools'),str(Path(%r)/'tools'),str(Path(%r)/'tools')]\nimport packet\npacket.build(r,old,%r)\nprint((r/'SCORE_PACKET_PRIVATE.json').read_text())\n""" % (str(Path(remote)/arm),os.environ['ADDITIONS_ROOT'],os.environ['FULL_D_ROOT'],os.environ['FIXED_ROOT'],selected)
        packets[arm]=ssh(script,timeout=300);write('PACKET_'+arm+'.private.json',packets[arm])
    legacy={k for p in packets.values() for v in p['payloads'].values() for k in v['legacy_keys']}
    semantic={k for p in packets.values() for k in p['payloads']}
    for path in [w/x for x in ('.private/med146_single_sequential_v133_20261027/scoring_recovery1/MERGED_DECISIONS.json',
        '.private/med150_exact_switch_v1_20261028/DECISIONS.json','.private/med150_balancedit_v1_20261028/scoring/DECISIONS.json',
        '.private/med150_memit_extension_v1_20261028/scoring/DECISIONS.json','.private/med150_grace_v1_20261028/scoring/DECISIONS.json')]:
        if path.is_file():
            for key,value in ledger.selected_decisions(path,legacy).items():
                if key in history['valid'] and any(history['valid'][key][f]!=value[f] for f in score_reuse.FIELDS):history['conflicts'].append(key)
                else:history['valid'][key]=value
    for raw in os.environ['DECISION_FILES'].split(os.pathsep):
        path=Path(raw)
        if not path.is_file():raise FileNotFoundError('required decision ledger missing')
        for key,value in ledger.selected_decisions(path,semantic | {prior[(cid,'MED')]['key'] for cid in fallback}).items():
            history['semantic_valid'].setdefault(key,[]).append(value)
    baseline=json.loads(Path(os.environ['PRIOR25_BASELINE']).read_text())
    if len(baseline['consumers'])!=2352:raise ValueError('prior 25 baseline missing')
    for key,value in baseline['decisions'].items():
        if key in semantic:history['semantic_valid'].setdefault(key,[]).append(value)
    decisions={};blocked=set();scoring={};consumers=[];med_keys={}
    for arm,packet in packets.items():
        part=score_reuse.resolve(packet,history);scoring[arm]={k:len(v) for k,v in part.items()};blocked.update(part['conflict_unknown'])
        for key,value in part['known_valid'].items():
            if key in decisions and any(decisions[key][f]!=value[f] for f in score_reuse.FIELDS):blocked.add(key)
            else:decisions[key]=value
        for row in packet['consumers']:
            row=dict(row);old=prior[(row['consumer_id'],'MED')]
            if any(row[k]!=old[k] for k in ('consumer_id','query_id','owner','role','route')):raise ValueError('consumer binding differs')
            if row['arm']=='CANDIDATE':row['arm']=arm;consumers.append(row)
            elif row['arm']=='MED':
                if row['key']!=old['key']:raise ValueError('original semantic binding differs')
                if arm=='MCE':med_keys[row['consumer_id']]=row['key'];consumers.append(row)
                elif med_keys.get(row['consumer_id'])!=row['key']:raise ValueError('two MED payloads differ')
            else:raise ValueError('unexpected arm')
    for cid in fallback:
        med=prior[(cid,'MED')];cand=prior[(cid,'MED_D')]
        if cand['candidate_expert'] or cand['key']!=med['key']:raise ValueError('fallback identity differs')
        key=med['key'];candidates=history['semantic_valid'].get(key,[])
        values={tuple(x.get(f) for f in score_reuse.FIELDS) for x in candidates}
        if len(values)==1 and all(type(x) is bool for x in next(iter(values))):
            decisions[key]=dict(opaque_query_id=key,**dict(zip(score_reuse.FIELDS,next(iter(values)))))
        elif len(values)>1:blocked.add(key)
        for arm in ('MED','MCE','MDU'):
            row=dict(med);row['arm']=arm;consumers.append(row)
    for key in blocked:decisions.pop(key,None)
    roles=json.loads((w/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').read_text())
    result=summary.compare(consumers,decisions,roles,routing);result['scoring']=scoring;result['new_judge_attempts']=0
    old=json.loads((w/'reports/med_medinit_t1g_supplement_v1_20261104/RESULTS.json').read_text())['panels']
    for role in summary.ROLES:
        for field in summary.FIELDS:
            a=result['groups']['original25_route'][role][field];b=old[role][field]
            for arm in summary.ARMS:
                if a['arms'][arm]!=b['arms'][arm]:raise ValueError('previous 25-arm panel mismatch')
            for pair in ('MED_to_MCE','MED_to_MDU','MCE_to_MDU'):
                if any(a['pairs'][pair][k]!=b['pairs'][pair][k] for k in ('gain','loss','unknown')):
                    raise ValueError('previous 25 paired panel mismatch')
    write('RESULTS.json',result)
    out=w/'reports/med_medinit_101_v1_20261104';out.mkdir(parents=True,exist_ok=True)
    with (out/'RESULTS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    lines=['# MED 初始化双臂101+49开发验证','','101个候选专家、49个原MED回退；各臂新增76专家，旧25专家和预测按绑定复用。',
           '2148事件原R0，新增生成1121、旧候选复用392、原MED回退635。评分仅复用既有有效判定，历史未知保留；无新Judge。','']
    for group in ('all','original25_route','new76_route','med49_route'):
        lines += ['## '+group,'']
        for role in summary.ROLES:
            values=result['groups'][group][role]['core_correct']['arms']
            lines.append(role+' CORE：'+'；'.join(a+' '+('NA' if values[a]['bounds'] is None else f"{values[a]['bounds'][0]:.4f}–{values[a]['bounds'][1]:.4f}%（未知{values[a]['unknown']}）") for a in summary.ARMS))
        lines.append('')
    lines += ['完整CORE/FULL、支持、配对增损、编辑自助区间和题目owner分层见RESULTS.json。未知赋值边界不是置信区间。',
              '该面板已曝光，不能当独立确认。T1L无支持按NA；MED初态含历史保护信息，比较方法精度差异需保留。']
    with (out/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    write('REPORT_COMPLETE.json',{'status':result['status'],'events':2148});write('COMPLETE.json',{'status':result['status'],'events':2148,'new_attempts':0})

if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('DRIVER_FAILURE.private.json',{'traceback':traceback.format_exc()});raise
    finally:write('DRIVER_EXIT.json',{'code':code,'time':time.time()})
