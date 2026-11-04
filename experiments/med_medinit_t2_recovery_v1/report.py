"""Merge the authorized decisions into the frozen paired DEV comparison."""
import json
import os
from pathlib import Path
import sys

w = Path(os.environ['WORKTREE'])
root = Path(os.environ['RECOVERY_ROOT'])
baseline_root = Path(os.environ['BASELINE_ROOT'])
destination = w / 'reports/med_medinit_t2_recovery_v1_20261104'
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
result = load(root / 'DECISIONS.json')
targets = set(frozen['payloads'])
new = result['decisions']
unknown = set(result['unknown'])
if len(targets) != 43 or set(new) & unknown or set(new) | unknown != targets:
    raise ValueError('incomplete or overlapping scoring partition')
if completion['new_attempt_items'] > 43 or completion['unknown'] != len(unknown):
    raise ValueError('invalid completion counts')
decisions = before['decisions']
if set(decisions) & targets:
    raise ValueError('accepted original decision would be overwritten')
fields = ('core_correct', 'full_supported', 'unsupported_expansion', 'extra_contradiction')
for key, verdict in new.items():
    if verdict['opaque_query_id'] != key or any(type(verdict.get(x)) is not bool for x in fields):
        raise ValueError('invalid new decision')
    if verdict['full_supported'] and (not verdict['core_correct'] or verdict['unsupported_expansion'] or verdict['extra_contradiction']):
        raise ValueError('contradictory new decision')
decisions.update(new)
roles = load(w / '.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json')
aggregated = summary.compare(before['consumers'], decisions, roles)
aggregated['scoring'] = {'frozen_unique_targets': 43, 'new_valid': len(new), 'unknown': len(unknown),
                         'registered_attempt_items': completion['new_attempt_items'], 'prior_interface_failures': 43}
aggregated['status'] = 'COMPLETE' if not unknown else 'INCOMPLETE'
with (destination / 'RESULTS.json').open('x') as f:
    json.dump(aggregated, f, ensure_ascii=False, indent=2)
lines = ['# MED 初始状态双臂：T2 定向补评', '',
         '固定392个已曝光开发事件，原 MED 与两组各25专家使用相同路由及角色资格；原模型和训练预测未重做。',
         f"用户授权的43个历史状态无法核实载荷：新有效{len(new)}，未知{len(unknown)}；不排除未记录的更早历史重复。", '',
         '| 角色与指标 | MED | MED 初始 CE | MED 初始 D-U |', '|---|---:|---:|---:|']
for role in ('T2G', 'T2L'):
    for field, label in (('core_correct','CORE'), ('full_supported','FULL')):
        panel = aggregated['panels'][role][field]['arms']
        def fmt(arm):
            item = panel[arm]
            bounds = item['bounds']
            if bounds is None:
                return 'NA'
            value = f'{bounds[0]:.4f}%' if bounds[0] == bounds[1] else f'{bounds[0]:.4f}–{bounds[1]:.4f}%'
            return value + f"（未知{item['unknown']}）"
        lines.append(f"| {role} {label} | {fmt('MED')} | {fmt('MCE')} | {fmt('MDU')} |")
    n = aggregated['panels'][role]['core_correct']['arms']['MED']
    lines.append(f"\n{role}：{n['events']}事件、{n['edits']}编辑。")
    for arm in ('MCE', 'MDU'):
        pair = aggregated['panels'][role]['core_correct']['pairs']['MED_to_' + arm]
        lines.append(f"{arm}相对 MED：修复{pair['gain']}、损失{pair['loss']}、配对未知{pair['unknown']}。")
    lines.append('')
lines += ['百分比为编辑宏平均；如有范围，其是对未知项赋值的界限，不是置信区间。',
          'T1L 无支持，其他角色的剩余未知见结果文件；旧 MED 权重含历史保护训练，两新臂未直接读取保护曲率。',
          '首次接口错误在请求前登记43项并失败；本次依用户明确授权恢复，各载荷最多实际请求一次，失败不自动重试。',
          '这些是已曝光开发集结果，不能直接推断独立测试表现。']
with (destination / 'REPORT_ZH.md').open('x') as f:
    f.write('\n'.join(lines) + '\n')
audit = {'status': aggregated['status'], 'frozen_events': 392, 'target_unique_payloads': 43,
         'prior_registered_interface_failures': 43, 'prior_actual_requests': 0,
         'authorized_recovery_attempt_items': completion['new_attempt_items'], 'valid_new_decisions': len(new),
         'unknown': len(unknown), 'worker_exit': worker['code'], 'guard_exit': guard['code'],
         'model': 'gpt-6-astra', 'reasoning': 'medium', 'immutable_snapshot': 'unknown',
         'training': 0, 'generation': 0, 'automatic_retry': False,
         'deadline_utc': '2026-11-04T11:25:24Z', 'historical_attempt_coverage_complete': False}
with (destination / 'EXECUTION_AUDIT.json').open('x') as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)
(root / 'REPORT_COMPLETE.json').write_text(json.dumps({'status': aggregated['status'], 'unknown': len(unknown)}))
