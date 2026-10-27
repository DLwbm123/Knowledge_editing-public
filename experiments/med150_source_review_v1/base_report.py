"""Publish aggregate results of the fixed source-only Base feasibility audit."""
import collections
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def read(p):return json.loads(Path(p).read_text())
def main():
    q=Path('/private/tmp/y.20261028.4');done=read(q/'COMPLETE.json');guard=read(q/'GUARD_EXIT.json')
    assert done['generation_and_scoring_complete'] and done['consumers']==37 and guard['code']==0
    f=read(q/'SOURCE_BASE_FEASIBILITY_PRIVATE.json');group=f['groups'];assert len(group)==8
    records=read(q/'PACKET_PRIVATE.json')['consumers'];judgments=read(q/'DECISIONS.json');d=judgments['decisions'];unknown=judgments['unknown']
    assert len(records)==37 and done['unknown']==len(unknown)
    target=[]
    for x in group:
        target.append(dict(group_index=x['group_index'],candidate_questions=x['candidate_questions'],status=x['status'],protection_questions=len((x.get('selected') or {}).get('protect_qids',[]))))
    count=collections.Counter(x['status'] for x in group)
    metrics=dict(core_correct=sum(d.get(x['key'],{}).get('core_correct') is True for x in records),core_incorrect=sum(d.get(x['key'],{}).get('core_correct') is False for x in records),full_supported=sum(d.get(x['key'],{}).get('full_supported') is True for x in records),unknown_consumers=sum(x['key'] not in d for x in records),EOS=sum(x['EOS'] for x in records),at_cap=sum(x['at_cap'] for x in records))
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000','/data/bmw/envs/v0/bin/python','-'],input="""from pathlib import Path
import json
r=Path('/data/bmw/Knowledge_editing/outputs/med150-source-base-v1-20261028/run')
names=('BASE_GENERATION_COMPLETE.json','BASE_WORKER_EXIT.json','BASE_GUARD_EXIT.json','CLOCK.json')
print(json.dumps({n:json.loads((r/n).read_text()) for n in names}))
""",text=True,capture_output=True,timeout=30);assert r.returncode==0,r.stderr
    remote=json.loads(r.stdout);generation=remote['BASE_GENERATION_COMPLETE.json'];assert generation['consumers']==37 and remote['BASE_WORKER_EXIT.json']['code']==remote['BASE_GUARD_EXIT.json']['code']==0
    result=dict(status='COMPLETE' if not unknown else 'INCOMPLETE_SCORING',time=time.time(),candidate_groups=8,candidate_queries=37,group_status={k:count[k] for k in ('ANNOTATION_ASSUMED_BASE_POTENTIAL','NO_KNOWN_BASE_PAIR','UNKNOWN_SCORING')},per_group=target,
        scoring=done,base_quality=metrics,generation_counts=generation['counts'],same_image_binding_per_group=generation['same_image_binding_per_group'],
        experiment_elapsed_seconds=generation['time']-remote['CLOCK.json']['original_start'],initial_scoring_engineering_failure=True,initial_attempted_items=0,
        qualified_clinical_groups=None,clinical_source_status='UNKNOWN',image_semantics='UNVERIFIED',patient_independence='UNKNOWN',method_trainings=0,
        comparison='One Base only; no MED method evaluation',selection='First Base-core-wrong target with at least one different original content_type and Base-FULL-supported candidate; all judgments in group required',
        limitation='Content-type difference is a conservative structural proxy, not confirmed distinct medical predicates; source annotation is not clinical truth.')
    dest=ROOT/'reports/med150_source_review_v1_20261028';(dest/'BASE_FEASIBILITY_RESULTS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# 固定八组来源假设下的 Base 可行性','',f"状态：{result['status']}。在原始8组、37条来源标注候选中，{count['ANNOTATION_ASSUMED_BASE_POTENTIAL']}组具有预先定义的潜在Base目标与保护组合；{count['NO_KNOWN_BASE_PAIR']}组无组合，{count['UNKNOWN_SCORING']}组评分未知。",'',
        f"37个Base回答中，CORE正确{metrics['core_correct']}、错误{metrics['core_incorrect']}、未知{metrics['unknown_consumers']}；FULL支持{metrics['full_supported']}。这是参考标注的一致性判断，不能等同图像医学真实性。",'',
        '|固定组（匿名序号）|候选问题|结构性潜在组合状态|潜在保护问题数|','|---:|---:|---|---:|']
    for g in target:
        label={'ANNOTATION_ASSUMED_BASE_POTENTIAL':'有潜在组合，待来源核实','NO_KNOWN_BASE_PAIR':'无满足规则组合','UNKNOWN_SCORING':'UNKNOWN'}[g['status']]
        lines.append(f"|{g['group_index']}|{g['candidate_questions']}|{label}|{g['protection_questions']}|")
    lines+=['','目标按原qid顺序选择Base CORE错误且有不同content_type的Base FULL支持项；保留所有8组，不以方法表现或难度补组。不同content_type仅是保守结构代理，仍需独立检查是否为不同谓词。图像仅通过文件解码和同组输入绑定，未进行临床图像语义审查，患者独立性未知；本轮没有开展临床合格审定，因此正式临床合格组数为UNKNOWN，不代表已证明没有可用病例。','',
        f"37次Base生成及保护器正常退出；Astra medium唯一载荷新尝试{done['new_attempt_items']}次、严格旧判断复用{done['reused']}次、UNKNOWN {done['unknown']}。初始评分程序误从来源元数据读取尚不存在的图像摘要，未启动任何Judge即失败；修正为使用实际Base运行输入绑定后，在原时钟内仅继续未尝试评分，旧失败记录保留。没有重生成、方法训练或对照方法优化。",
        '后续若要正式执行方法比较，必须另行冻结数据角色与模型合同，补充来源图像/不同谓词依据和必要的医疗专业审查；不得将本次可行性当作独立确认、临床保护或已证明主线效果。','']
    (dest/'BASE_FEASIBILITY_ZH.md').write_text('\n'.join(lines));print(json.dumps(dict(status=result['status'],groups=result['group_status'],base_quality=metrics,scoring=done)))

if __name__=='__main__':main()
