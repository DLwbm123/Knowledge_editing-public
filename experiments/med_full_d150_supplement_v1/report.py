"""Compare the fixed 150-bank MED+D predictions to MED and BalancEdit."""
import json
import importlib.util
import statistics
import sys
from pathlib import Path

W=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(W/'experiments/med_full_d150_v1'),str(W/'experiments/medical_proposition_v133')]
spec=importlib.util.spec_from_file_location('med_full_d150_report',W/'experiments/med_full_d150_v1/report.py')
med_report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(med_report)
from med133_report import interval

ROLES=('T0','FIT','T1G','T2G','T1L','T2L')
FIELDS=('core_correct','full_supported')


def build(score_root,cohorts):
    score_root=Path(score_root)
    done=json.loads((score_root/'COMPLETE.json').read_text())
    if json.loads((score_root/'WORKER_EXIT.json').read_text()).get('code')!=0:
        raise ValueError('scoring worker did not complete')
    lock=json.loads((score_root/'SUPPLEMENT_INPUT.private.json').read_text())
    supplemental=json.loads((score_root/'DECISIONS.json').read_text())
    prior_root=W/'.private/med_full_d150_v1_20261102/scoring'
    packet=json.loads((prior_root/'PACKET_PRIVATE.json').read_text())
    prior=json.loads((prior_root/'DECISIONS.json').read_text())
    roles=json.loads((W/'.private/med_full_d150_v1_20261102/EVALUATION_ROLES_PRIVATE.json').read_text())
    adm=json.loads((score_root/'PAIRING_ADMISSION.json').read_text())
    if adm.get('status')!='PASS' or adm.get('bank_query_events')!=2148 or adm.get('question_image_prompt_attention_generation_mismatches')!=0:
        raise ValueError('comparison input admission missing')
    if set(supplemental['decisions']) & set(prior['decisions']):raise ValueError('old score overwritten')
    target=set(lock['categories']['history_unverified']+lock['categories']['conflict_unknown'])
    if target!=set(prior['unknown']) or set(supplemental['decisions']) | set(supplemental['unknown'])!=target:
        raise ValueError('supplement score coverage differs')
    combined={**prior['decisions'],**supplemental['decisions']}
    conservative={k:v for k,v in combined.items() if k not in set(lock['categories']['conflict_unknown'])}
    a=med_report.run(packet,combined,roles,done,cohorts=cohorts)
    sensitivity=med_report.run(packet,conservative,roles,done,cohorts=cohorts)
    be_root=W/'.private/med150_balancedit_v1_20261028/scoring'
    if not json.loads((be_root/'COMPLETE.json').read_text()).get('generation_and_scoring_complete'):
        raise ValueError('BalancEdit score completion missing')
    if json.loads((be_root/'GUARD_EXIT.json').read_text()).get('code')!=0:
        raise ValueError('BalancEdit score guard failed')
    be_packet=json.loads((be_root/'PACKET_PRIVATE.json').read_text())
    be_dec=json.loads((be_root/'DECISIONS.json').read_text())
    be_rows=[x for x in be_packet['consumers'] if x.get('mode')=='bank' and x.get('prefix')==150]
    be={x['query_id']:x['key'] for x in be_rows}
    if len(be_rows)!=2148 or len(be)!=2148 or be_dec['unknown']:
        raise ValueError('BalancEdit bank coverage or score unknown')
    med_d={x['query_id']:x['key'] for x in packet['consumers'] if x['arm']=='MED_D'}
    if set(be)!=set(med_d) or len(med_d)!=2148:raise ValueError('comparison event identities differ')
    score_be=be_dec['decisions']
    sets={'all':range(1,151),'original':range(1,147),'additional':range(147,151),**cohorts}
    comparison={}
    for name,owners in sets.items():
        comparison[name]={}
        for role in ROLES:
            entries=[(o,q) for o in owners for q in roles['roles'][str(o)][role]]
            comparison[name][role]={}
            for field in FIELDS:
                old=[];new=[];by_owner={};gained=lost=unknown=0
                for owner,q in entries:
                    if q not in be or q not in med_d:raise ValueError('role event absent from comparison')
                    b=score_be.get(be[q],{}).get(field);v=combined.get(med_d[q],{}).get(field)
                    old.append((owner,b));new.append((owner,v));by_owner.setdefault(owner,[]).append((b,v))
                    if b is None or v is None:unknown+=1
                    else:gained+=b is False and v is True;lost+=b is True and v is False
                diffs=[None if any(x is None or y is None for x,y in rows) else statistics.mean(y-x for x,y in rows) for rows in by_owner.values()]
                delta=100*statistics.mean(diffs) if diffs and all(x is not None for x in diffs) else None
                ci=interval(diffs) if name=='all' and role in ('T2G','T2L') else None
                comparison[name][role][field]={'BALANCEDIT':med_report.summarize(old),'MED_D':med_report.summarize(new),
                    'MED_D_minus_BALANCEDIT_pp':delta,'edit_bootstrap95_pp':ci,'gained_vs_balancedit':gained,
                    'lost_vs_balancedit':lost,'unknown_pairs':unknown}
    return {'status':a['status'],'same_support_med':a,'conservative_conflict_sensitivity':sensitivity,
            'balancedit_comparison':comparison,'adjudication':{'history_unverified':48,'conflict_items':5,
                'valid_new':len(supplemental['decisions']),'unknown_remaining':len(supplemental['unknown']),
                'new_attempt_items':done['new_attempt_items'],'original_valid_reused':1376,
                'conflicts_preserved_in_original':True,'all_payloads_not_selected_by_outcome':True},
            'pairing':adm,'cohort':'exposed MED150 dev; 25 MED+D,125 original MED fallback',
            'precision_note':'BalancEdit FP16 backbone plus FP32 writer; MED+D FP32 decoder and FP16 prefill'}


def write_report(result,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'RESULTS.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    def fmt(x):
        if not x['events']:return 'NA'
        return f"{x['macro_percent']:.4f}%" if x['macro_percent'] is not None else f"{x['bounds'][0]:.4f}–{x['bounds'][1]:.4f}%（未知{x['unknown']}）"
    lines=['# MED＋D 150补评与 BalancEdit 同支持比较','',f"状态：{result['status']}。混合150专家库由25个MED＋D和125个原MED组成；同一原R0与2148个bank查询。",'',
           '|角色|事件 / 编辑|原MED CORE|MED＋D CORE|BalancEdit CORE|原MED FULL|MED＋D FULL|BalancEdit FULL|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for role in ROLES:
        c=result['same_support_med']['panels']['all'][role]['core_correct'];f=result['same_support_med']['panels']['all'][role]['full_supported']
        be_c=result['balancedit_comparison']['all'][role]['core_correct']['BALANCEDIT'];be_f=result['balancedit_comparison']['all'][role]['full_supported']['BALANCEDIT']
        lines.append(f"|{role}|{c['MED']['events']} / {c['MED']['support_edits']}|{fmt(c['MED'])}|{fmt(c['MED_D'])}|{fmt(be_c)}|{fmt(f['MED'])}|{fmt(f['MED_D'])}|{fmt(be_f)}|")
    lines+=['','按编辑宏平均计算；T2G/T2L 的原146、新增4、来源匹配/不匹配/身份未知分层、逐项增损及编辑bootstrap区间见 RESULTS.json。',
        '本次针对48个历史状态无法核实的载荷各补评最多一次；5个历史评分冲突的载荷各进行一次盲态裁决。旧冲突原样保留，裁决不表示历史评分一致；保守地将冲突继续视为UNKNOWN的敏感性结果见 RESULTS.json。',
        f"本轮实际提交 {result['adjudication']['new_attempt_items']} 个载荷，剩余 UNKNOWN {result['adjudication']['unknown_remaining']}；原1376个有效判定未重评。历史记录不完整，不能排除无记录的历史重复。",
        'BalancEdit采用FP16骨干和FP32 writer，MED＋D采用FP32 decoder和FP16 prefill；问题、图像、提示词、attention和生成配置已逐项核验，方法精度和资源并非完全相同。',
        '结果来自已曝光开发队列；49项训练来源组身份未解决而回退原MED，弱词面匹配不是关系标注。新增4没有合格T1G/T2L支持，T1L仅2题。','']
    with (dest/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(lines))
