"""Compare frozen operator decisions with the previously completed controls."""
import json
import os
import sys
from pathlib import Path

w = Path(os.environ['WORKTREE'])
out = w/'reports/med_operator_bridge_v1_20261103'
fixed = json.loads((out/'arm_fixed/RESULTS.json').read_text())
trained = json.loads((w/'reports/med_trainable_bridge_v1_20261103/RESULTS.json').read_text())
prior = json.loads((w/'reports/med_d_u101_v1_20261102/RESULTS.json').read_text())
supp = json.loads((w/'reports/med_d_u101_t2g_supplement_v1_20261102/RESULTS.json').read_text())
roots = [Path('/private/tmp/.r.20261103.03'), Path('/private/tmp/.r.20261103.01')]
packets = [json.loads((root/'PACKET_PRIVATE.json').read_text()) for root in roots]
decisions = [json.loads((roots[0]/'DECISIONS.json').read_text())['decisions'],
             json.loads((roots[1]/'REAGG_DECISIONS.private.json').read_text())['decisions']]
roles = json.loads((w/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').read_text())
if roles.get('status') != 'EVALUATION_ROLES_FROZEN' or roles.get('owners') != 150:
    raise ValueError('role lock differs')
maps = []
for packet in packets:
    if packet['status'] != 'SCORE_PACKET_FROZEN' or len(packet['consumers']) != 4296:
        raise ValueError('frozen score packet incomplete')
    rows = {(r['arm'], r['query_id']): r for r in packet['consumers']}
    if len(rows) != 4296:
        raise ValueError('duplicate score consumer')
    maps.append(rows)
if set(maps[0]) != set(maps[1]):
    raise ValueError('comparison queries differ')
for k in maps[0]:
    a, b = maps[0][k], maps[1][k]
    if any(a.get(x) != b.get(x) for x in ('consumer_id','query_id','owner','role','route','candidate_expert')):
        raise ValueError('comparison input identity or route differs')
    if k[0] == 'MED' and a['key'] != b['key']:
        raise ValueError('shared MED score payload differs')

sys.path.insert(0, str(w/'experiments/med_full_d150_v1'))
from report import summarize
comparisons, paired = {}, {}
for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
    comparisons[role], paired[role] = {}, {}
    entries = [(owner,q) for owner in range(1,151) for q in roles['roles'][str(owner)][role]]
    for field in ('core_correct','full_supported'):
        a = fixed['panels']['all'][role][field]
        b = trained['panels']['all'][role][field]
        old = prior['comparison'][role][field]
        if any(x['events'] != a['MED']['events'] or x['support_edits'] != a['MED']['support_edits']
               for x in (b['MED'], old['MED'])):
            raise ValueError('comparison supports differ')
        comparisons[role][field] = {'MED':a['MED'], 'FIXED':a['MED_D'], 'TRAINED':b['MED_D'],
                                    'D':old['D'], 'DU':old['DU'], 'BALANCEDIT':old['BALANCEDIT']}
        if role == 'T2G':
            comparisons[role][field]['DU_USER_SUPPLEMENT'] = supp['panels']['all'][role][field]['MED_D']
        rows_a, rows_b = [], []
        gains = losses = unknown = 0
        for owner, q in entries:
            va = decisions[0].get(maps[0]['MED_D',q]['key'], {}).get(field)
            vb = decisions[1].get(maps[1]['MED_D',q]['key'], {}).get(field)
            rows_a.append((owner,va)); rows_b.append((owner,vb))
            if va is None or vb is None: unknown += 1
            else: gains += va is False and vb is True; losses += va is True and vb is False
        if summarize(rows_a) != a['MED_D'] or summarize(rows_b) != b['MED_D']:
            raise ValueError('paired decisions do not reproduce reported arm summaries')
        paired[role][field] = {'trained_improved':gains,'trained_lost':losses,'unknown_pairs':unknown}

result = dict(fixed)
result.update(same_support_comparison=comparisons, fixed_to_trained=paired,
              cohort='101 fixed mixed experts and 49 original MED fallback',
              pairing_note='2148 input and route bindings match; decisions are event paired',
              score_note='no new Judge; unknown and conflicts retained; D-U user supplement listed separately',
              bounds_note='unknown assignment bounds, not confidence intervals')
with (out/'RESULTS.json').open('x') as f:
    json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')

def fmt(row):
    if row['bounds'] is None: return 'NA'
    if row['macro_percent'] is not None: return f"{row['macro_percent']:.4f}%"
    return f"{row['bounds'][0]:.4f}–{row['bounds'][1]:.4f}%（未知{row['unknown']}）"

lines=['# 固定 0.5 算子桥：开发集同支持对照','',
       f"状态：{fixed['status']}。101 个专家固定混合原 MED 与 D-U 残差，其余 49 个回退 MED；同原 R0 评估 2148 个事件。",'',
       '|角色 / CORE|MED|固定桥|训练桥|原D|D-U|BalancEdit|',
       '|---|---:|---:|---:|---:|---:|---:|']
for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
    row=comparisons[role]['core_correct']
    lines.append('|'+role+'|'+'|'.join(fmt(row[k]) for k in ('MED','FIXED','TRAINED','D','DU','BALANCEDIT'))+'|')
t=fixed['panels']['all']['T2G']['core_correct'];l=fixed['panels']['all']['T2L']['core_correct']
lines += ['',f"T2G 为 571 事件、150 编辑；固定桥相对 MED 已知修复 {t['gained']}、损失 {t['lost']}，未知配对 {t['unknown_pairs']}。T2L 为 54 事件、31 编辑；固定桥已知损失 {l['lost']}，未知配对 {l['unknown_pairs']}。即使未知全部判对，T2L 上界仍低于 MED 超过 5 个百分点，冻结门槛失败。",
          f"固定桥到训练桥的 T2G 已知修复 {paired['T2G']['core_correct']['trained_improved']}、损失 {paired['T2G']['core_correct']['trained_lost']}，未知配对 {paired['T2G']['core_correct']['unknown_pairs']}；T2L 的对应数为 {paired['T2L']['core_correct']['trained_improved']}、{paired['T2L']['core_correct']['trained_lost']}、{paired['T2L']['core_correct']['unknown_pairs']}。未知会限制两种桥之间的结论。",
          f"本轮 {fixed['scoring']['reused_valid']} 个唯一载荷复用有效判定、{fixed['scoring']['unknown']} 个未知，新增评分 0。区间是未知全错至全对的边界，不是置信区间；D-U 补评口径单列于结果文件。",
          '101 个专家的真实层双端点和中点在正式预测前通过；1513 次实际生成、635 次复用。输入、路由和回退与训练桥逐项匹配；不表示输出或权重相同。',
          '这是已曝光 DEV 消融，固定桥不是最终方法；MED 专家带历史保护训练影响。BalancEdit 沿既有裁决口径且运行精度不同。T1L 仅 2 个事件，新增 4 项的 T2L 无支持，OFF 支持为 0。完整 CORE/FULL、原146/新增4和来源分层见 RESULTS.json。']
with (out/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(lines)+'\n')
