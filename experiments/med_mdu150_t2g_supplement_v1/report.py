"""Merge only this authorized T2G supplement and publish anonymous aggregates."""
import json
import os
from pathlib import Path
import shutil
import sys

w = Path(os.environ['WORKTREE'])
root = Path(os.environ['LOCAL_SCORE_ROOT'])
sys.path.insert(0, str(w/'experiments/med_mdu150_v1'))
import summary

def read(p):
    return json.loads(Path(p).read_text())

for name in ('GUARD_EXIT.json', 'WORKER_EXIT.json'):
    if read(root/name)['code'] != 0:
        raise ValueError('Scoring process failed: '+name)
frozen = read(root/'SUPPLEMENT_INPUT.private.json')
completed = read(root/'COMPLETE.json')
answer = read(root/'DECISIONS.json')
baseline = read(root/'BASELINE.private.json')
old = read(w/'reports/med_mdu150_v1_20261105/RESULTS.json')
targets = set(frozen['payloads'])
new = answer['decisions']
unknown = set(answer['unknown'])
if (len(targets)!=35 or set(new)&unknown or set(new)|unknown != targets
        or set(new)&set(baseline['decisions']) or completed['new_attempt_items']>35):
    raise ValueError('Authorized score coverage differs')
fields = ('core_correct', 'full_supported', 'unsupported_expansion', 'extra_contradiction')
for k,v in new.items():
    if v.get('opaque_query_id')!=k or any(type(v.get(f)) is not bool for f in fields):
        raise ValueError('Invalid verdict')
    if v['full_supported'] and (not v['core_correct'] or v['unsupported_expansion'] or v['extra_contradiction']):
        raise ValueError('Inconsistent full verdict')
manifest = read(root/'BATCH_MANIFEST.json')
submitted = []
for batch in manifest['batches']:
    operator = root/'batches'/batch['batch_id']/'operator'
    attempt = operator/'ATTEMPT.json'
    if attempt.exists():
        charge = read(attempt)
        if charge['items'] != len(batch['records']) or charge['automatic_retry']:
            raise ValueError('Attempt accounting mismatch')
        submitted += [r['opaque_query_id'] for r in batch['records']]
    evidence = operator/'execution_evidence'/(batch['batch_id']+'.json')
    if evidence.exists():
        e = read(evidence)
        if e['actual_model']!='gpt-6-astra' or e['reasoning_effort']!='medium' or e['tool_event_types']:
            raise ValueError('Scoring contract or isolation changed')
if len(submitted)!=len(set(submitted)) or not set(submitted)<=targets or len(submitted)!=completed['new_attempt_items']:
    raise ValueError('Duplicate or unaccounted score attempt')
decisions = {**baseline['decisions'], **new}
result = summary.compare(baseline['consumers'], decisions, baseline['roles'], baseline['routing'])
conflicts = set()
for suffix in ('102', '103'):
    conflicts.update(read('/private/tmp/.r.20261104.'+suffix+'/SUPPLEMENT_INPUT.private.json')['categories']['conflict_unknown'])
sensitive = summary.compare(baseline['consumers'], {k:v for k,v in decisions.items() if k not in conflicts},
                            baseline['roles'], baseline['routing'])
for group in old['groups']:
    for role in summary.ROLES:
        for field in summary.FIELDS:
            for arm in ('MED', 'MDU101', 'BE'):
                if result['groups'][group][role][field]['arms'][arm] != old['groups'][group][role][field]['arms'][arm]:
                    raise ValueError('An original comparator panel changed')
panel = result['groups']['all']['T2G']['core_correct']['arms']['MDU150']
if panel['events']!=571 or panel['edits']!=150 or panel['unknown']!=len(unknown):
    raise ValueError('T2G denominator or coverage changed')
result.update(status='T2G_COMPLETE' if not unknown else 'T2G_INCOMPLETE',
    scoring=dict(target_events=35, unique_payloads=35, new_attempt_items=len(submitted), new_valid=len(new),
        unknown=len(unknown), automatic_retry=False, history_coverage_complete=False,
        new_core_correct=sum(v['core_correct'] for v in new.values()),
        new_full_supported=sum(v['full_supported'] for v in new.values())),
    history_coverage_complete=False, new_judge_attempts=len(submitted), conflicts_sensitivity=sensitive['groups'])
durable = w/'.private/med_mdu150_t2g_supplement_v1_20261106/scoring'
durable.mkdir(parents=True, exist_ok=False)
for name in ('BASELINE.private.json','SUPPLEMENT_INPUT.private.json','PREPARE_COMPLETE.json','BATCH_MANIFEST.json',
             'DECISIONS.json','COST.json','COMPLETE.json','clock.json','GUARD_EXIT.json','WORKER_EXIT.json'):
    shutil.copy2(root/name, durable/name)
for path in (root/'batches').iterdir():
    source = path/'operator'
    dest = durable/'batches'/path.name/'operator'
    dest.mkdir(parents=True)
    for name in ('ATTEMPT.json','PREFLIGHT.json','FAILURE_NO_RETRY.json'):
        if (source/name).exists():
            shutil.copy2(source/name, dest/name)
    for name in ('execution_evidence','responses'):
        if (source/name).exists():
            shutil.copytree(source/name, dest/name)
out = w/'reports/med_mdu150_t2g_supplement_v1_20261106'
def write(name, obj):
    with (out/name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
write('RESULTS.json', result)
lines = ['# MDU150 T2G 未知评分补全', '',
    f"用户授权的35个原冻结T2G未知载荷：实际新增尝试{len(submitted)}、有效{len(new)}、仍未知{len(unknown)}；无训练或回答生成，原有效评分及原报告不变。",
    f"新判定中CORE正确{result['scoring']['new_core_correct']}/35，FULL支持{result['scoring']['new_full_supported']}/35。", '',
    '| 指标 | MED | MDU101+49回退 | MDU150 | BalancEdit |', '|---|---:|---:|---:|---:|']
for role in summary.ROLES:
    for field,label in (('core_correct','CORE'), ('full_supported','FULL')):
        arms = result['groups']['all'][role][field]['arms']
        def fmt(arm):
            a=arms[arm]; b=a['bounds']
            if b is None:
                return 'NA'
            value=f'{b[0]:.4f}%' if b[0]==b[1] else f'{b[0]:.4f}–{b[1]:.4f}%'
            return value+f"（未知{a['unknown']}；{a['events']}事件/{a['edits']}编辑）"
        lines.append('| '+role+' '+label+' | '+' | '.join(fmt(a) for a in summary.ARMS)+' |')
lines += ['', '编辑等权宏平均；有未知时区间仅为未知赋值边界，不是置信区间。已曝光开发集，不能作独立测试结论。',
    '新增49专家实际路由组仍无合格T1L/T2L支持；全库T2L不能代表该49专家的局部性。T1L仅2事件。BalancEdit运行精度与MED系列不同。',
    '历史尝试账本不完整，35项此前状态不可核实；本次每项最多一次新增尝试，无自动重试，不能排除更早未留存重复。其他角色的残留未知、历史冲突敏感性、分组与逐事件配对增损见RESULTS.json。']
with (out/'REPORT_ZH.md').open('x') as f:
    f.write('\n'.join(lines)+'\n')
clock=read(root/'clock.json')
allocated=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
audit=dict(status=result['status'], original_all_groups_reproduced=True, original_valid_unchanged=True,
    comparator_panels_unchanged=True, frozen_t2g_events=571, frozen_t2g_edits=150, target_unknown_events=35,
    unique_target_payloads=35, new_attempt_items=len(submitted), valid=len(new), unknown=len(unknown),
    model='gpt-6-astra', reasoning='medium', automatic_retry=False, new_training=0, new_generation=0,
    worker_exit=0, guard_exit=0, clock=clock, final_local_run_allocated_bytes=allocated,
    historical_attempt_coverage_complete=False, immutable_model_snapshot=None,
    resource_note='Independent 23h guard, 2GB conservative polling stop, 1GiB single-file cap and 8GiB reserve; not an aggregate filesystem quota.')
write('EXECUTION_AUDIT.json', audit)
(root/'REPORT_COMPLETE.json').write_text(json.dumps(dict(status=result['status'], t2g=panel, new_valid=len(new))))
print(json.dumps(dict(status=result['status'], T2G_CORE=panel, scoring=result['scoring']), ensure_ascii=False))
