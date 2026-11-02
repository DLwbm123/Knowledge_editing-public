"""Reuse only known compatible judgments; preserve all other payloads as UNKNOWN."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

root=Path(os.environ['LOCAL_SCORE_ROOT'])
worktree=Path(os.environ['WORKTREE'])
deadline=float(os.environ['EXPERIMENT_DEADLINE'])
arm=os.environ['ARM']
if arm not in ('D','DU'): raise ValueError('invalid arm')
remote=os.environ['EXPERIMENT_ROOT']
old150=os.environ['ADDITIONS_ROOT']


def write(name,value):
    p=root/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,ensure_ascii=False));t.replace(p)
    p.chmod(0o600)


def ssh(script, timeout=60):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','pro5000','python3','-'],
        input=script,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError('remote dependency or packet failed: '+p.stderr[-600:])
    if len(p.stdout.encode())>45_000_000:raise ValueError('remote packet exceeds bounded size')
    return json.loads(p.stdout)


def main():
    sys.path[:0]=[str(worktree/'experiments/med_full_d150_v1'),str(worktree/'experiments/med_proposition_generalization_v1'),
        str(worktree/'experiments/med_fixed_depth_v1'),str(worktree/'experiments/med_source_teacher_v1'),
        str(worktree/'experiments/med150_source_review_v1')]
    import ledger,partition,report
    src=Path('/private/tmp/.r.20261102.03/HISTORY_INDEX_PRIVATE.json')
    if not src.is_file():raise ValueError('frozen history identity index missing')
    with src.open() as f:history=json.load(f)
    if history.get('status')!='AVAILABLE_HISTORY_INDEX_FROZEN' or history.get('history_coverage_complete') is not False:
        raise ValueError('invalid history identity index')
    if not (worktree/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').is_file():
        raise ValueError('evaluation role identities missing')
    write('HISTORY_INDEX_FROZEN.json',{'status':'AVAILABLE_HISTORY_INDEX_FROZEN','history_coverage_complete':False,
                                      'score_attempts_allowed':0})
    probe="""import json\nfrom pathlib import Path\nr=Path(%r)\nnames=('TRAIN_GUARD_FAILURE.json','EVAL_GUARD_FAILURE.json','TRAIN_STOP.json','EVAL_STOP.json')\np=r/'EVAL_GUARD_EXIT.json'\nprint(json.dumps({'failed':any((r/n).exists() for n in names),'done':p.is_file(),'code':json.loads(p.read_text())['code'] if p.is_file() else None}))\n""" % remote
    while True:
        if time.time()>=deadline-180:raise TimeoutError('original experiment deadline')
        state=ssh(probe)
        if state['failed'] or state['done'] and state['code']!=0:
            raise RuntimeError('GPU dependency failed; no scoring attempted')
        if state['done']:break
        write('DEPENDENCY_WAIT.json',{'status':'WAIT_GPU','time':time.time()})
        time.sleep(20)
    script="""import json,sys\nfrom pathlib import Path\nr=Path(%r);old=Path(%r);fix=Path(%r);full=Path(%r)\nsys.path[:0]=[str(r/'tools'),str(full/'tools'),str(fix/'tools')]\nimport packet\npacket.build(r,old)\nprint((r/'SCORE_PACKET_PRIVATE.json').read_text())\n""" % (remote,old150,os.environ['FIXED_ROOT'],os.environ['FULL_D_ROOT'])
    packet=ssh(script,timeout=180)
    if packet.get('status')!='SCORE_PACKET_FROZEN' or len(packet['consumers'])!=4296:
        raise ValueError('frozen score packet missing')
    write('PACKET_PRIVATE.json',packet)
    legacy={k for v in packet['payloads'].values() for k in v['legacy_keys']}
    semantic=set(packet['payloads'])
    valid=history['valid'];conflicts=set(history['conflicts'])
    legacy_files=[worktree/'.private/med146_single_sequential_v133_20261027/scoring_recovery1/MERGED_DECISIONS.json',
        worktree/'.private/med150_exact_switch_v1_20261028/DECISIONS.json',
        worktree/'.private/med150_balancedit_v1_20261028/scoring/DECISIONS.json',
        worktree/'.private/med150_memit_extension_v1_20261028/scoring/DECISIONS.json',
        worktree/'.private/med150_grace_v1_20261028/scoring/DECISIONS.json']
    used=0
    for path in legacy_files:
        if not path.is_file():continue
        used+=1
        for key,row in ledger.selected_decisions(path,legacy).items():
            if key in valid and any(valid[key][f]!=row[f] for f in partition.FIELDS):conflicts.add(key)
            else:valid[key]=row
    for path in (Path('/private/tmp/.r.20261102.01/DECISIONS.json'),Path('/private/tmp/.r.20261102.02/DECISIONS.json'),
                 Path('/private/tmp/.r.20261102.03/DECISIONS.json'),Path('/private/tmp/.r.20261101.09/DECISIONS.json'),
                 Path('/private/tmp/.r.20261102.05/DECISIONS.json'),Path('/private/tmp/.r.20261102.06/DECISIONS.json'),
                 Path('/private/tmp/.r.20261102.08/DECISIONS.json'),Path('/private/tmp/.r.20261103.01/REAGG_DECISIONS.private.json')):
        if path.is_file():
            for key,row in ledger.selected_decisions(path,semantic).items():
                history['semantic_valid'].setdefault(key,[]).append(row)
    parts=partition.partition(packet,valid,history['attempts'],history['semantic_valid'],history['semantic_attempts'],conflicts)
    unknown=parts['known_unknown']+parts['conflict_unknown']+parts['history_unverified']
    write('DECISIONS.json',{'decisions':parts['known_valid'],'unknown':unknown})
    complete={'status':'COMPLETE_WITH_POSSIBLE_UNKNOWN','events':2148,'unique_payloads':len(packet['payloads']),
              'reused_valid':len(parts['known_valid']),'unknown':len(unknown),'new_attempts':0,
              'history_coverage_complete':False,'legacy_decision_files_scanned':used}
    write('COMPLETE.json',complete)
    roles=json.loads((worktree/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').read_text())
    identity = ssh("import json;from pathlib import Path;print(json.dumps(json.loads((Path(%r)/'ROLE_ADMISSION.json').read_text())['owners']))" % remote)
    if len(identity)!=101 or len(set(identity))!=101 or any(not 1 <= x <= 150 for x in identity):
        raise ValueError('source cohort identity mismatch')
    unresolved=sorted(set(range(1,151))-set(identity))
    if len(unresolved)!=49:
        raise ValueError('unresolved cohort mismatch')
    result=report.run(packet,parts['known_valid'],roles,complete,cohorts={'admitted101':identity,'unresolved49':unresolved})
    out=worktree/'reports/med_operator_bridge_v1_20261103'/'arm_fixed';out.mkdir(parents=True,exist_ok=True)
    result['candidate_experts']=101
    result['fallback_experts']=49
    result_path=out/'RESULTS.json'
    with result_path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    lines=['# 固定算子桥：150专家库开发评估', '',
        f"状态：{result['status']}。候选臂 fixed operator bridge，使用已获准的来源专家，其他专家回退原MED；同原R0评估2148个bank查询。",
        f"评分：{complete['reused_valid']}个唯一载荷复用有效判定，{complete['unknown']}个保留UNKNOWN，新评分0；历史记录不完整。",
        '原146和新增4、T0/FIT/T1G/T2G/T1L/T2L的CORE/FULL、支持、未知与边界见RESULTS.json。未决计分不得视为正确或错误。',
        '原R0 OFF支持为0；原开发集已曝光，未纳入49项来源仍未知；本实验未使用受保护曲率。',
        '完整资源与执行审计以运行回执为准；原MED/M13P和既有负结果保留。','']
    with (out/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(lines))
    write('REPORT_COMPLETE.json',{'status':result['status'],'events':2148})


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('DRIVER_FAILURE.private.json',{'traceback':traceback.format_exc()});raise
    finally:write('DRIVER_EXIT.json',{'code':code,'time':time.time()})
