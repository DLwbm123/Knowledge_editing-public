"""Summarize fixed-role aggregate comparisons and recorded execution evidence."""
import json,os
from pathlib import Path
w=Path(os.environ['WORKTREE']);out=w/'reports/med_trainable_bridge_v1_20261103';r=json.loads((out/'RESULTS.json').read_text())
old=json.loads((w/'reports/med_d_u101_v1_20261102/RESULTS.json').read_text());du=json.loads((w/'reports/med_d_u101_t2g_supplement_v1_20261102/RESULTS.json').read_text())
roles=('T0','FIT','T1G','T2G','T1L','T2L');fields=('core_correct','full_supported');comparisons={}
for role in roles:
 comparisons[role]={}
 for field in fields:
  x=r['panels']['all'][role][field];other=old['comparison'][role][field]
  if x['MED']['events']!=other['MED']['events'] or x['MED']['support_edits']!=other['MED']['support_edits']:raise ValueError('support mismatch')
  comparisons[role][field]={'MED':x['MED'],'TRAINED_BRIDGE':x['MED_D'],'D':other['D'],'DU':other['DU'],'BALANCEDIT':other['BALANCEDIT']}
  if role=='T2G':comparisons[role][field]['DU_user_supplement']=du['panels']['all'][role][field]['MED_D']
r['same_support_comparison']=comparisons;r['fixed_0_5_bridge']='NOT_RUN';r['alpha_summary']={'count':101,'min':0.38894549012184143,'max':0.6004688143730164,'mean':0.40832234225650826,'median':0.40452298521995544,'below_0_5':99,'above_0_5':2};(out/'RESULTS.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
def fmt(x):
 if x['bounds'] is None:return 'NA'
 if x['macro_percent'] is not None:return f"{x['macro_percent']:.4f}%"
 return f"{x['bounds'][0]:.4f}–{x['bounds'][1]:.4f}%（未知{x['unknown']}）"
lines=['# 可训练算子桥：101项混合库开发对照','',f"状态：{r['status']}。原 R0 下 2148 事件，101 个训练后的混合专家、49 个原 MED 回退；固定 0.5 对照未运行。",'', '|角色 / CORE|MED|训练桥|原D|D-U|BalancEdit|','|---|---:|---:|---:|---:|---:|']
for role in roles:
 c=comparisons[role]['core_correct'];lines.append('|'+role+'|'+'|'.join(fmt(c[k]) for k in ('MED','TRAINED_BRIDGE','D','DU','BALANCEDIT'))+'|')
t=r['panels']['all']['T2G']['core_correct'];l=r['panels']['all']['T2L']['core_correct'];sc=r['scoring']
lines+=['',f"T2G 为571事件/150编辑，训练桥未知{t['MED_D']['unknown']}项，已知相对MED修复{t['gained']}、损失{t['lost']}，未知配对{t['unknown_pairs']}。T2L为54事件/31编辑，训练桥未知{l['MED_D']['unknown']}项，已知损失{l['lost']}、未知配对{l['unknown_pairs']}。区间为未知全错至全对的边界，不是置信区间；不能据上界声称达到目标或保护门槛。",f"本轮无新Judge。原评分复用{sc['original_reused']}个唯一载荷、未知139；补纳既有相同合同决策后复用{sc['reused_now']}个、未知{sc['remaining_unknown']}（已知失败{sc['known_unknown']}、历史冲突{sc['conflict_unknown']}、历史状态不可核实{sc['history_unverified']}）。已判与冲突原证据保留；没有评分重试。",'训练101个混合参数各40步，双端专家和骨干冻结；混合权重均值0.4083、范围0.3889–0.6005，99个小于0.5、2个大于0.5，只是训练状态描述。原MED专家带历史保护训练影响；未重新读取受保护曲率。','固定0.5桥接未运行，不能评价训练比固定混合更好。原D、D-U与BalancEdit为既有同支持汇总，D-U表中为原未补评口径，其T2G用户补评口径82.6667–89.5000%（41未知）另见结果文件。BalancEdit和MED运行精度不同。','T0/FIT为各150编辑、T1G为146编辑578事件，T1L只有1编辑2事件、OFF支持0；新增4的T2L无支持。本面板为已曝光DEV，来源owner分组不等同路由专家；FULL及分层见RESULTS.json。严格5个百分点保护门槛、T2G改善、T1G保护均未能判定，状态INCONCLUSIVE；用户约89/90目标同样没有被证明。']
(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
