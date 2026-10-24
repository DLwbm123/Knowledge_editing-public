"""Fixed comparisons and owner-level protection gates for the old curriculum."""
import json
import sqlite3
import time
from pathlib import Path
import curriculum_protocol as f
import selected_report146 as report

def calculate(*args,**kwargs):
    result,audit=report.calculate(*args,**kwargs)
    left,right=kwargs['methods']
    for role in ('T1G','T1L'):
        result['paired_native_FIT160_contrasts']['single_1'][role]={right+'_preserves_each_'+left+'_owner':report.reporting.preserved(audit['role_panels'][left+'_single_1'][role],audit['role_panels'][right+'_single_1'][role])}
    return result,audit

def main():
    import astra_queue as a
    q,run=a.q,a.RUN;root=run/'private'/f.QUEUE;assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    records={};scores={}
    with sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True) as db:
        assert not db.execute("select 1 from payload where status in ('PENDING','RESERVED')").fetchone()
        for path,binding,status,score in db.execute('select c.path,c.output_binding,p.status,p.correct from consumer c join payload p on p.key=c.payload_key'):
            d=q.read(Path(path));assert q.digest(d)==binding
            k=tuple(d[x] for x in ('method','mode','prefix','owner'))+(d['binding']['input']['query_id'],)
            assert k not in records;records[k]=d;scores[k]=score if status=='FORMAT_VALID' else None
    out=dict(scope='Original146 single DEV, three fixed 540-step curricula; no independent confirmation',contrasts={},historical=q.read(run/'private/HISTORICAL_COMPARISON.json'));owners={}
    for left,right in [('P160','M13P'),('C','M1'),('M1','M13P'),('C','M13P'),('C160','C'),('M1160','M1')]:
        subset={k:v for k,v in records.items() if k[0] in (left,right,'BASE')}
        result,audit=calculate(q.read(run/'private/ORIGINAL146_TASKS.json'),q.read(run/'private/ORIGINAL_BASE_MASK146.json'),subset,{k:scores[k] for k in subset},methods=(left,right),single_only=True)
        key=left+'_to_'+right;out['contrasts'][key]=result;owners[key]=audit
    primary=out['contrasts']['P160_to_M13P'];out['primary_gate']=f.gate(primary)
    q.write(run/'public/CURRICULUM_RESULTS.json',out);q.write(run/'private/CURRICULUM_OWNER_AUDIT.json',owners)
    lines=['# 旧课程三臂完整结果','','联合门槛：'+json.dumps(out['primary_gate'],ensure_ascii=False),'','| 配方 | FIT | T1G | T1L | T2G | T2L | 新损伤/Base正确 |','|---|---:|---:|---:|---:|---:|---:|']
    for method,label,pair in [('P160','M13P_160','P160_to_M13P'),('C','C540','C_to_M13P'),('M1','M1_540','M1_to_M13P'),('M13P','M13P_540','P160_to_M13P')]:
        d=out['contrasts'][pair];p=d['panels'][method]['single_1'];damage=d['protection'][method+'_single_1']
        lines.append('| '+label+' | '+' | '.join(str(p[k]['macro']) for k in ('FIT','T1G','T1L','T2G','T2L'))+' | '+str(damage['new_damage'])+'/'+str(damage['fresh_Base_correct'])+' |')
    lines+=['','历史参照及当前160步参照：'+json.dumps(out['historical'],ensure_ascii=False),'','完整配对差、bootstrap区间及支持数见CURRICULUM_RESULTS.json；逐编辑审计私有保留。T1L仅2条/1编辑，T2L54条/31编辑；不以总平均掩盖保护退化。旧TT88实现差异未消除，不把历史对比解释为单因素因果效应。']
    (run/'public/CURRICULUM_REPORT_ZH.md').write_text('\n'.join(lines)+'\n');q.write(run/'private/CURRICULUM_REPORT_COMPLETE.json',dict(status='COMPLETE',gate=out['primary_gate'],epoch=time.time()))
if __name__=='__main__':main()
