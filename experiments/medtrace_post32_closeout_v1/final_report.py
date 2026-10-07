"""Matched structural results and exact incomplete scope, no candidate expansion."""
import os,sys
from pathlib import Path
RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import core
common=core.common;report=core.tool('report')

def main():
    records,scores=report.load();ts=[t for t in common.read(RUN/'private/QUEUES.json')['tasks'] if t['cohort']=='P2'][:8];panels=[];coeff={};groups={}
    for mode in ['single_R0','bank_R0']:
        for arm in (['TT88_W0','LORA_W0'] if mode=='single_R0' else ['FROZEN_W0','LORA_W0']):
            for task in ['T0','T1G','T2G']:
                m,c,g=report.panel(records,scores,'P2',arm,0,mode,task,ts,1 if mode=='single_R0' else 8);panels.append(m);coeff[(mode,arm,task)]=c;groups.update(g)
    paired=[]
    for mode in ['single_R0','bank_R0']:
        for task in ['T0','T1G','T2G']:
            aa=coeff[(mode,'LORA_W0',task)];bb=coeff[(mode,'TT88_W0' if mode=='single_R0' else 'FROZEN_W0',task)];cc={e:report.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa.keys()&bb.keys()};bounds=report.score_bounds(report.combine([{k:v/len(cc) for k,v in c.items()} for c in cc.values()]),scores);paired.append(dict(comparison='LoRA-TT88',mode=mode,task=task,delta_pp=[v*100 for v in bounds],**report.bootstrap(cc,scores,groups)))
    ledger=common.read(RUN/'RESOURCE_LEDGER.json');assert all(x.get('ended_epoch') for x in ledger['gpu_sessions']);assert ledger['gpu_seconds_used']<=86400 and ledger['Judge_attempts']<=6000
    common.write(RUN/'public/FINAL_COMPARISON.json',dict(panels=panels,paired=paired,scale=8,retrospective=True,TT_parameters=7168,LoRA_parameters=18432,normalized_LoRA_rank=1,parameter_matched=False,not_standard_unmodified_LoRA_recipe=True,extra_U_selected_version=False,benchmark146_complete=False,independent_confirmation=False,GPU_hours=ledger['gpu_seconds_used']/3600,Judge_attempts=ledger['Judge_attempts'],owned_artifact_peak_bytes=ledger['weights_observed_peak_bytes']))
    text='# PR32后收尾：已执行部分与缺口\n\n四条件固定R0的核心40条新增续训与真实8库/24混合库结果见CORE_RESULTS.json/CSV。TT88与匹配输入RMS归一化的LoRA rank1结构对照见FINAL_COMPARISON.json；实际参数7168与18432，非等参数、非标准未经修改的LoRA训练配方。原146回顾性，当前仅8个匹配编辑；146仍差138个匹配编辑及prefix50/100/146，不能拼接历史CP/rank或全MULTI24。完整系统BalancEdit/BELoRA复用尚未通过本轮完整身份审核，因此未报告新的系统比较。独立确认未完成；保护参考资格不足，医学保持指标NA，续训不晋级，版本锁W0+R0。公开前仍需人工科学/资产审阅和GitHub交付。\n\n'
    text+='|模式|臂|任务|宏平均%|正确|观测|缺失|\n|---|---|---|---:|---:|---:|---:|\n'
    for m in panels:text+='|'+ '|'.join(str(m[k]) for k in ['mode','arm','task','macro','known_correct','observations','missing'])+'|\n'
    (RUN/'public/REPORT_ZH.md').write_text(text)
if __name__=='__main__':main()
