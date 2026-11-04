"""Merge frozen T1G decisions and verify the other reported panels are unchanged."""
import json
import os
from pathlib import Path
import sys

w = Path(os.environ['WORKTREE'])
root = Path(os.environ['RECOVERY_ROOT'])
baseline_root = Path(os.environ['BASELINE_ROOT'])
destination = w / 'reports/med_medinit_t1g_supplement_v1_20261104'
sys.path.insert(0, str(w / 'experiments/med_medinit_loss_v1'))
import summary

def load(path):
    return json.loads(path.read_text())

completion = load(root / 'COMPLETE.json')
guard = load(root / 'GUARD_EXIT.json')
worker = load(root / 'WORKER_EXIT.json')
if guard['code'] != 0 or worker['code'] != 0:
    raise ValueError('scoring process failed')
frozen = load(root / 'SUPPLEMENT_INPUT.private.json')
before = load(baseline_root / 'BASELINE.private.json')
response = load(root / 'DECISIONS.json')
targets = set(frozen['payloads'])
new, unknown = response['decisions'], set(response['unknown'])
if len(targets) != 7 or set(new) & unknown or set(new) | unknown != targets:
    raise ValueError('invalid scoring partition')
if completion['new_attempt_items'] > 7 or completion['unknown'] != len(unknown):
    raise ValueError('invalid completion counts')
decisions = before['decisions']
if set(decisions) & targets:
    raise ValueError('accepted original decision would be overwritten')
fields = ('core_correct', 'full_supported', 'unsupported_expansion', 'extra_contradiction')
for key, verdict in new.items():
    if verdict['opaque_query_id'] != key or any(type(verdict.get(x)) is not bool for x in fields):
        raise ValueError('invalid decision')
    if verdict['full_supported'] and (not verdict['core_correct'] or verdict['unsupported_expansion'] or verdict['extra_contradiction']):
        raise ValueError('contradictory decision')
decisions.update(new)
roles = load(w / '.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json')
aggregated = summary.compare(before['consumers'], decisions, roles)
old = load(w / 'reports/med_medinit_t2_recovery_v1_20261104/RESULTS.json')
for role in ('T0', 'FIT', 'T2G', 'T1L', 'T2L'):
    if aggregated['panels'][role] != old['panels'][role]:
        raise ValueError('non-T1G panel changed: ' + role)
aggregated['scoring'] = {'frozen_unique_targets': 7, 'new_valid': len(new), 'unknown': len(unknown),
                         'registered_attempt_items': completion['new_attempt_items']}
aggregated['status'] = 'COMPLETE' if not unknown else 'INCOMPLETE'
with (destination / 'RESULTS.json').open('x') as f:
    json.dump(aggregated, f, ensure_ascii=False, indent=2)
lines = ['# MED 初始状态双臂：T1G 定向补评', '',
         '原392个共同路由开发事件、角色资格、专家预测和已接受判定均保持；独立补评固定 T1G 未知载荷。',
         f"7个历史状态不可核实载荷：新有效{len(new)}、未知{len(unknown)}，不排除更早历史重复。", '',
         '| 指标 | MED | MED 初始 CE | MED 初始 D-U |', '|---|---:|---:|---:|']
for role in ('T1G', 'T2G', 'T2L', 'T0', 'FIT', 'T1L'):
    for field, label in (('core_correct','CORE'), ('full_supported','FULL')):
        panel = aggregated['panels'][role][field]['arms']
        def fmt(arm):
            item = panel[arm]; bounds = item['bounds']
            if bounds is None:
                return 'NA'
            value = f'{bounds[0]:.4f}%' if bounds[0] == bounds[1] else f'{bounds[0]:.4f}–{bounds[1]:.4f}%'
            return value + f"（未知{item['unknown']}）"
        lines.append(f"| {role} {label} | {fmt('MED')} | {fmt('MCE')} | {fmt('MDU')} |")
lines += ['', '数值为编辑宏平均；如有区间，是未知赋值界限而非置信区间。']
for role in ('T1G', 'T2G', 'T2L'):
    panel = aggregated['panels'][role]['core_correct']; n = panel['arms']['MED']
    lines.append(f"{role}：{n['events']}事件、{n['edits']}编辑。")
    for arm in ('MCE', 'MDU'):
        pair = panel['pairs']['MED_to_' + arm]
        lines.append(f"{arm}相对MED CORE：修复{pair['gain']}、损失{pair['loss']}、配对未知{pair['unknown']}。")
lines += ['', 'T0、FIT、T1L、T2G和T2L全部已报告面板与补评前逐项一致；T1L没有支持。',
          '结果属于已曝光开发集；两组权重、完整输出和未来泛化不因聚合值相同而被证明相同。']
with (destination / 'REPORT_ZH.md').open('x') as f:
    f.write('\n'.join(lines) + '\n')
audit = {'status': aggregated['status'], 'frozen_events': 392, 'target_unique_payloads': 7,
         'registered_attempt_items': completion['new_attempt_items'], 'valid_new_decisions': len(new),
         'unknown': len(unknown), 'worker_exit': worker['code'], 'guard_exit': guard['code'],
         'model': 'gpt-6-astra', 'reasoning': 'medium', 'immutable_snapshot': 'unknown',
         'unchanged_roles': ['T0','FIT','T1L','T2G','T2L'], 'training': 0, 'generation': 0,
         'automatic_retry': False, 'deadline_utc': '2026-11-04T11:25:24Z',
         'historical_attempt_coverage_complete': False}
with (destination / 'EXECUTION_AUDIT.json').open('x') as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)
(root / 'REPORT_COMPLETE.json').write_text(json.dumps({'status': aggregated['status'], 'unknown': len(unknown)}))
