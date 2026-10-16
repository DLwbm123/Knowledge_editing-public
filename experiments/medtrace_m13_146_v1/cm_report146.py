"""Matched C/M13 DEV result; missing, absent controls and small support stay explicit."""
import json
import os
from pathlib import Path
import sqlite3
import time
import cm_eval146 as protocol
import selected_report146 as report


def gate(result):
    label='bank_146';comparison=result['paired_native_FIT160_contrasts'][label]
    terms={role:comparison[role]['M13_preserves_each_C_owner'] for role in ('FIT','T2G','T2L')}
    terms['damage']=comparison['M13_damage_no_worse_than_C_per_owner']
    return dict(terms=terms,passed=result['complete_no_missing'] and all(v is True for v in terms.values()),
        scope='Fixed matched C/M13 development compatibility only; not original M1/M3 joint success')


def main():
    import astra_queue as a
    q,run=a.q,a.RUN;queue=run/'private'/protocol.QUEUE
    protocol.admit(run)
    from cm_queue146 import recovery,completion_root,all_workers_done
    assert (queue/'ALL_WORKERS_COMPLETE.json').exists() and all_workers_done(completion_root(queue,recovery(run)),4 if (run/'private/CM_PARALLEL_SCORING1.json').exists() else 1)
    with sqlite3.connect('file:'+str(queue/'queue.sqlite')+'?mode=ro',uri=True) as db:
        assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
        records={};scores={}
        for path,binding,status,score in db.execute('SELECT c.path,c.output_binding,p.status,p.correct FROM consumer c JOIN payload p ON p.key=c.payload_key'):
            data=q.read(Path(path));assert q.digest(data)==binding
            key=tuple(data[k] for k in ('method','mode','prefix','owner'))+(data['binding']['input']['query_id'],)
            assert key not in records;records[key]=data;scores[key]=score if status=='FORMAT_VALID' else None
    result,owners=report.calculate(q.read(run/'private/ORIGINAL146_TASKS.json'),q.read(run/'private/ORIGINAL_BASE_MASK146.json'),records,scores,methods=protocol.METHODS)
    result['matched_C_M13_compatibility']=gate(result)
    ledger=q.read(run/'RESOURCE_LEDGER.json');prior=q.read(run/'private/INHERITED_COST.json')
    assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    result['cost']=dict(new_GPU_seconds=ledger['gpu_seconds_used']-prior['gpu_seconds_used'],new_Judge=ledger['Judge_attempts']-prior['Judge_attempts'])
    result['outputs']=dict(consumers=17081,reused=706,new_generations=16375,new_training_updates=0)
    result['Fisher_status']='PAUSED_PARTIAL_NOT_EVALUATED'
    q.write(run/'private/CM_RESULTS_WITH_OWNER_AUDIT.json',dict(result,owner_panels=owners))
    public=json.loads(json.dumps(result))
    for method in protocol.METHODS:
        public['panels'][method]['bank_1']={'status':'PRIVATE_SMALL_GROUP_AUDIT_ONLY'}
        public['protection'][method+'_bank_1']={'status':'PRIVATE_SMALL_GROUP_AUDIT_ONLY'}
    public['paired_native_FIT160_contrasts']['bank_1']={'status':'PRIVATE_SMALL_GROUP_AUDIT_ONLY'}
    q.write(run/'public/CM_RESULTS.json',public)
    lines=['# 原146 C与M13冻结开发评估','','完整无缺失：'+str(result['complete_no_missing']),
        '匹配C/M13兼容门槛：'+str(result['matched_C_M13_compatibility']['passed']),
        '','| 方法 | FIT宏平均 | T2G宏平均 | T2L宏平均 | 新损伤 / fresh Base正确 |','|---|---:|---:|---:|---:|']
    for method in protocol.METHODS:
        p=result['panels'][method]['bank_146'];d=result['protection'][method+'_bank_146']
        lines.append(f"| {method} | {p['FIT']['macro']} | {p['T2G']['macro']} | {p['T2L']['macro']} | {d['new_damage']} / {d['fresh_Base_correct']} |")
    lines+=['','这是冻结原146、原图+FIT160、原R0路由的DEV比较，完整single与prefix及逐编辑保留门槛见JSON。',
        'Fisher仍暂停，原三臂完整验证未完成；同配方M1/M3控制缺失，不据此宣称原联合成功。',
        '没有新增训练、换种子或修改评估问法。复用身份一致的706输出，其余16375输出仅尝试一次；历史失败与缺失保留。',
        '无独立确认或医学安全结论；保护支持分母及缺失必须随指标解释。']
    (run/'public/CM_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    q.write(run/'private/CM_REPORT_COMPLETE1.json',dict(status='COMPLETE',complete_no_missing=result['complete_no_missing'],epoch=time.time()))


if __name__=='__main__':main()
