"""Combine frozen paired arm summaries and the previously bound comparator."""
import json
import os
from pathlib import Path

w=Path(os.environ['WORKTREE'])
out=w/'reports/med_d_u101_v1_20261102'
source=w/'reports/med_full_d150_supplement_v1_20261102/RESULTS.json'
old=json.loads(source.read_text())
supplement=json.loads((w/'reports/med_d_u101_t2g_supplement_v1_20261102/RESULTS.json').read_text())
if old.get('pairing',{}).get('question_image_prompt_attention_generation_mismatches')!=0:
 raise ValueError('BalancEdit original bank binding missing')
res={arm:json.loads((out/('arm_'+arm)/'RESULTS.json').read_text()) for arm in ('DU','D')}
comparisons={}
for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
 comparisons[role]={}
 for field in ('core_correct','full_supported'):
  d=res['D']['panels']['all'][role][field]
  u=res['DU']['panels']['all'][role][field]
  be=old['balancedit_comparison']['all'][role][field]['BALANCEDIT']
  if d['MED']!=u['MED'] or d['MED']['events']!=be['events'] or d['MED']['support_edits']!=be['support_edits']:
   raise ValueError('comparison support differs')
  comparisons[role][field]={'MED':d['MED'],'D':d['MED_D'],'DU':u['MED_D'],'BALANCEDIT':be}
result={'status_D':res['D']['status'],'status_DU':res['DU']['status'],'events':2148,
        'cohort':'exposed DEV; 101 D-U experts plus 49 original MED; 25 original D plus 125 original MED',
        'pairwise_MED':{arm:res[arm]['panels'] for arm in ('DU','D')},'comparison':comparisons,
        'supplemented_DU_MED':supplement['panels'], 'supplement_scoring':supplement['scoring'],
        'balanc_precision_note':old['precision_note'],
        'unknown_policy':'original arms use no new judge; DU supplement adds 14 valid judgments, 42 reserved pre-call failures remain UNKNOWN',
        'note':'101/49 cohort refers to query owner, not necessarily R0 selected expert; BE aggregate uses previously documented adjudication'}
with (out/'RESULTS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
with (out/'REPORT_ZH.md').open('x') as f:
 f.write('# D-U 101项混合库开发对照\n\n')
 f.write(f"D-U：{res['DU']['status']}；原D：{res['D']['status']}。两个候选分别训练101和25个专家，其余保留原MED；均按原R0对照2148个查询。\n\n")
 for role in ('T2G','T2L','T0','FIT','T1G','T1L'):
  c=comparisons[role]['core_correct']
  vals='；'.join(name+' '+('NA' if c[name]['bounds'] is None else (f"{c[name]['macro_percent']:.4f}%" if c[name]['macro_percent'] is not None else f"{c[name]['bounds'][0]:.4f}–{c[name]['bounds'][1]:.4f}%（未知{c[name]['unknown']}）")) for name in ('MED','D','DU','BALANCEDIT'))
  f.write(f'{role} CORE：{vals}（事件{c["MED"]["events"]}，支持编辑{c["MED"]["support_edits"]}）。\n\n')
 x=supplement['panels']['all']['T2G']['core_correct']
 f.write(f"用户授权的D-U T2G补评口径：D-U {x['MED_D']['bounds'][0]:.4f}–{x['MED_D']['bounds'][1]:.4f}%（未知{x['MED_D']['unknown']}），MED {x['MED']['bounds'][0]:.4f}–{x['MED']['bounds'][1]:.4f}%（未知{x['MED']['unknown']}），已知修复{x['gained']}、损伤{x['lost']}，未知配对{x['unknown_pairs']}；14新评分有效，42项登记后调用前失败保留未知。该补评只用于D-U，原D沿原协议不新评分。\n\n")
 f.write('完整CORE/FULL、分母、未知边界、与原MED配对增损和来源分组见RESULTS.json。BalancEdit采用历史已披露裁决结果及运行精度差异；本实验未知评分未重试。所有结果为已曝光DEV，组别为题目owner而非必然选中专家，不能视为150个D-U专家或独立测试。\n')
