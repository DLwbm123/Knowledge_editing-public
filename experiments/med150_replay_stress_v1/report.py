"""Aggregate the frozen paired forced-expert source CHECK diagnostic."""
import collections
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/medical_proposition_v133'))
from med133_report import interval


def read(p):return json.loads(Path(p).read_text())
def summarize(pairs,field):
    counts=collections.Counter();delta=[];groups=[]
    for a,b in pairs:
        x=a['decision'].get(field);y=b['decision'].get(field)
        if x is None or y is None:counts['unknown']+=1;continue
        counts['known']+=1;counts['base_correct']+=x;counts['MED_correct']+=y
        counts['base_correct_lost']+=x and not y;counts['base_incorrect_repaired']+=not x and y
        delta.append(int(y)-int(x));groups.append(a['source_group'])
    fields=('known','unknown','base_correct','MED_correct','base_correct_lost','base_incorrect_repaired')
    result={k:counts[k] for k in fields}
    result.update(n=len(pairs),base_accuracy_percent=(100*counts['base_correct']/len(pairs) if pairs and not counts['unknown'] else None),
        MED_accuracy_percent=(100*counts['MED_correct']/len(pairs) if pairs and not counts['unknown'] else None),
        delta_pp=(100*statistics.mean(delta) if delta and not counts['unknown'] else None),
        source_group_bootstrap95_delta_pp=interval(delta,groups) if not counts['unknown'] else None,
        base_correct_damage_rate_percent=(100*counts['base_correct_lost']/counts['base_correct'] if counts['base_correct'] and not counts['unknown'] else None))
    return result


def main():
    q=Path('/private/tmp/k.20261028.5');done=read(q/'COMPLETE.json');assert done['generation_and_scoring_complete'] and read(q/'GUARD_EXIT.json')['code']==0
    values=read(q/'PACKET_PRIVATE.json')['consumers'];decisions=read(q/'DECISIONS.json')['decisions'];missing=read(q/'DECISIONS.json')['unknown']
    assert len(values)==186 and done['consumers']==186
    grouped=collections.defaultdict(dict)
    for v in values:
        assert v['arm'] not in grouped[v['query_id']];grouped[v['query_id']][v['arm']]=dict(**v,decision=decisions.get(v['key'],{}))
    assert len(grouped)==93 and all(set(v)=={'BASE','MED_FORCED'} for v in grouped.values())
    pairs=[]
    for v in grouped.values():
        a,b=v['BASE'],v['MED_FORCED'];assert a['source_group']==b['source_group'] and a['answer_kind']==b['answer_kind'] and a['forced_index']==b['forced_index']
        assert 0<=a['forced_index']<32;pairs.append((a,b))
    groups=collections.Counter(a['source_group'] for a,b in pairs);assert len(groups)==31 and set(groups.values())=={3}
    kinds=collections.Counter(a['answer_kind'] for a,b in pairs);assert kinds=={'yes':31,'no':31,'open':31}
    fields=('core_correct','full_supported');panels={}
    for f in fields:
        panels[f]={'all':summarize(pairs,f)}
        for kind in ('yes','no','open'):panels[f][kind]=summarize([p for p in pairs if p[0]['answer_kind']==kind],f)
    core=panels['core_correct']['all'];signal=None if core['unknown'] or core['base_correct']<15 else core['base_correct_damage_rate_percent']>20
    import_code="""from pathlib import Path
import json,collections
r=Path('/data/bmw/Knowledge_editing/outputs/med150-replay-stress-v1-20261028/run')
vs=[json.loads(p.read_text()) for p in (r/'outputs').glob('*.json')];by=collections.defaultdict(dict)
for x in vs:by[x['binding']['input']['query_id']][x['method']]=x
assert len(by)==93 and all(set(v)=={'BASE','MED_FORCED'} for v in by.values())
audit=dict(queries=93,same_inputs=sum(v['BASE']['binding']['judge_input']==v['MED_FORCED']['binding']['judge_input'] for v in by.values()),forced_nonzero_trace=sum(bool(v['MED_FORCED']['generation_trace']) and any(z['active_residual_norm']>0 for z in v['MED_FORCED']['generation_trace']) for v in by.values()),base_no_expert=sum(v['BASE']['effective_expert'] is None for v in by.values()),base_trace_empty=sum(not v['BASE']['generation_trace'] for v in by.values()))
receipts={n:json.loads((r/n).read_text()) for n in ('EVAL_CPU_ADMISSION.json','EVAL_GENERATION_COMPLETE.json','EVAL_WORKER_EXIT.json','EVAL_GUARD_EXIT.json','CLOCK.json')}
print(json.dumps(dict(audit=audit,receipts=receipts)))
"""
    remote=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000','/data/bmw/envs/v0/bin/python','-'],input=import_code,text=True,capture_output=True,timeout=120)
    assert remote.returncode==0,remote.stderr;remote=json.loads(remote.stdout);audit=remote['audit'];receipt=remote['receipts'];assert all(audit[k]==93 for k in ('same_inputs','forced_nonzero_trace','base_no_expert','base_trace_empty'))
    assert receipt['EVAL_CPU_ADMISSION.json']['excluded_reserved_groups']==1 and receipt['EVAL_CPU_ADMISSION.json']['unselected_QA_decoded']==0
    assert receipt['EVAL_GENERATION_COMPLETE.json']['consumers']==186 and receipt['EVAL_WORKER_EXIT.json']['code']==receipt['EVAL_GUARD_EXIT.json']['code']==0
    same=sum(a['key']==b['key'] for a,b in pairs)
    quality={arm:dict(consumers=sum(x['arm']==arm for x in values),EOS=sum(x['arm']==arm and x['EOS'] for x in values),at_cap=sum(x['arm']==arm and x['at_cap'] for x in values)) for arm in ('BASE','MED_FORCED')}
    result=dict(status='COMPLETE' if not missing else 'INCOMPLETE_SCORING',time=time.time(),groups=31,queries=93,reserved_groups_excluded=1,unselected_QA_decoded=0,
        scoring=done,panels=panels,diagnostic_signal=signal,signal_rule='BASE CORE support >=15 and MED CORE Base-correct damage rate >20%; false or UNKNOWN is not clinical evidence',
        paired_exact_same_payload=same,input_and_forced_activation_audit=audit,quality=quality,forced_activation=True,natural_routing=False,independent_confirmation=False,
        clinical_truth='UNKNOWN',patient_independence='UNKNOWN',generation_counts=receipt['EVAL_GENERATION_COMPLETE.json']['counts'],experiment_elapsed_seconds=receipt['EVAL_GENERATION_COMPLETE.json']['time']-receipt['CLOCK.json']['original_start'])
    dst=ROOT/'reports/med150_replay_stress_v1_20261028';(dst/'RESULTS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# MED 强制激活来源保护压力诊断','',f"状态：{result['status']}；31个来源组、93题，原Base和强制MED各生成93条。原CHECK32中与V95保留角色重叠的一组已在问题内容解码前排除，不补位。",'',
        '|来源支持|Base正确/总数|MED正确/总数|Base原正确后损伤|Base原错误后修复|UNKNOWN|','|---|---:|---:|---:|---:|---:|']
    for f in fields:
        for typ in ('all','yes','no','open'):
            m=panels[f][typ];lines.append(f"|{f} / {typ}|{m['base_correct']}/{m['n']}|{m['MED_correct']}/{m['n']}|{m['base_correct_lost']}/{m['base_correct']}|{m['base_incorrect_repaired']}|{m['unknown']}|")
    lines+=['',f"预登记诊断信号：{signal}。规则要求Base CORE正确支持至少15题且MED使其中超过20%变错；此信号仅用于判断是否值得设计MED＋来源回放研究，并非安全结论或已授权训练。",
        f"CORE总体MED−Base差值 {core['delta_pp']} pp；按来源组bootstrap 95%区间 {core['source_group_bootstrap95_delta_pp']} pp。FULL、分类型、分母与缺失见RESULTS.json。相同源问题、图像/token/runtime绑定93/93，实际MED强制钩子非零93/93；Base未挂专家93/93。",
        f"186个消费者对应{done['unique_payloads']}个唯一评分载荷，其中同一问题的Base/MED完全相同载荷{same}对；新评分尝试{done['new_attempt_items']}、旧同合同复用{done['reused']}、UNKNOWN {done['unknown']}。",'',
        '固定CHECK为历史已曝光开发数据，患者/研究独立与医学参考真实性未知；强制激活不是自然R0路由或实际伤害频率。不能把来源一致性评分写成临床正确、独立确认或整体优越。任何后续MED＋来源回放须另行冻结方法、同支持对照与资源，仅使用历史FIT角色，不使用本CHECK或保留组训练。',
        '本次没有训练、修改MED专家、重建M13P或扩展来源。原始QA、图像、完整回答和逐例身份不公开。','']
    (dst/'REPORT_ZH.md').write_text('\n'.join(lines));print(json.dumps(dict(status=result['status'],CORE=panels['core_correct']['all'],FULL=panels['full_supported']['all'],diagnostic_signal=signal,scoring=done)))

if __name__=='__main__':main()
