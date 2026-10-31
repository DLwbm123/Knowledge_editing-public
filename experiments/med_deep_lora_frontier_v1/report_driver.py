"""Publishable aggregate of a completed frozen score epoch."""
import json
import os
from pathlib import Path
import report


def main():
    root = Path(os.environ['FIXED_SCORE_ROOT'])
    worktree = Path(os.environ['FIXED_WORKTREE'])
    if json.loads((root/'DRIVER_EXIT.json').read_text())['code'] != 0 or json.loads((root/'GUARD_EXIT.json').read_text())['code'] != 0:
        raise ValueError('Scoring chain did not complete')
    packet = json.loads((root/'PACKET_PRIVATE.json').read_text())
    judgments = json.loads((root/'DECISIONS.json').read_text())
    oldroot = Path('/private/tmp/.q.fxdscore')
    oldpacket = json.loads((oldroot/'PACKET_PRIVATE.json').read_text())
    oldjudgments = json.loads((oldroot/'DECISIONS.json').read_text())
    private = worktree/'.private/med_fixed_depth_v1_20261030'
    baseline = json.loads((private/'BASE_M0_SCORE_BINDINGS.json').read_text())['rows']
    roles = json.loads((private/'EVALUATION_ROLES.json').read_text())
    result = report.evaluate(packet, judgments['decisions'], baseline, roles, oldpacket, oldjudgments['decisions'])
    result['scoring'] = json.loads((root/'COMPLETE.json').read_text())
    destination = worktree/'reports/med_deep_lora_frontier_v1_20261031'
    (destination/'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 深层 LoRA 固定面板结果', '', f"状态：{result['status']}；选中：{result['selected'] or '无'}。", '',
             '| 角色 | MED CORE | 0.25 CORE | 0.50 CORE | 0.75 CORE | 1.00 CORE |',
             '|---|---:|---:|---:|---:|---:|']
    def fmt(value):
        return 'NA/UNKNOWN' if value is None else f'{value:.4f}%'
    for role in report.ROLES:
        panel = result['panels'][role]['core_correct']
        arms = ['MED'] + [f'LORA_{g:.2f}' for g in report.GAMMAS]
        lines.append('| '+role+' | '+' | '.join(fmt(panel[a]['macro_percent']) for a in arms)+' |')
    lines += ['', '所有数值为编辑宏平均；FULL、支持数、未知项、上下界、配对增损与生成质量见 RESULTS.json。',
              '本面板为已曝光开发集，T1L无支持；四个强度均预先固定。历史状态未核实的评分本轮最多尝试一次，可能与不可追溯的历史记录重合。',
              '用户允许默认评分模型按同一口径比较；不可变模型快照未知，不称版本一致。']
    (destination/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    (root/'REPORT_COMPLETE.json').write_text(json.dumps(dict(status=result['status'], selected=result['selected'])))


if __name__ == '__main__':
    main()
