"""All predeclared single-expert contrasts; M3 remains the primary candidate."""
import json
import os
from pathlib import Path
import sqlite3
import time
import factorial_protocol as f
import selected_report146 as report

def main():
    import astra_queue as a
    q,run=a.q,a.RUN;root=run/'private'/f.QUEUE
    assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    records={};scores={}
    with sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True) as db:
        assert not db.execute("select 1 from payload where status in ('PENDING','RESERVED')").fetchone()
        for path,binding,status,score in db.execute('select c.path,c.output_binding,p.status,p.correct from consumer c join payload p on p.key=c.payload_key'):
            d=q.read(Path(path));assert q.digest(d)==binding
            k=tuple(d[x] for x in ('method','mode','prefix','owner'))+(d['binding']['input']['query_id'],)
            assert k not in records;records[k]=d;scores[k]=score if status=='FORMAT_VALID' else None
    out={'scope':'Original146 single-expert DEV factorial; not independent confirmation','contrasts':{}}
    owner_audits={}
    for left,right in [('C','M13'),('C','M1'),('C','M3'),('M13','M3'),('M1','M13'),('M3','M13')]:
        subset={k:v for k,v in records.items() if k[0] in (left,right,'BASE')}
        result,owners=report.calculate(q.read(run/'private/ORIGINAL146_TASKS.json'),q.read(run/'private/ORIGINAL_BASE_MASK146.json'),subset,{k:scores[k] for k in subset},methods=(left,right),single_only=True)
        key=left+'_to_'+right;out['contrasts'][key]=result;owner_audits[key]=owners
    primary=out['contrasts']['M13_to_M3'];terms=primary['paired_native_FIT160_contrasts']['single_1']
    out['primary_M3_gate']=dict(complete_no_missing=primary['complete_no_missing'],FIT=terms['FIT']['M3_preserves_each_M13_owner'],T2L=terms['T2L']['M3_preserves_each_M13_owner'],T2G_improves=terms['T2G']['M3_minus_M13_macro_pp']>0,damage=terms['M3_damage_no_worse_than_M13_per_owner'])
    out['primary_M3_gate']['passed']=all(v is True for v in out['primary_M3_gate'].values())
    original=q.read(Path(f.admit(run)['evaluation_parent'])/'public/CM_RESULTS.json')
    for m in ('C','M13'):
        assert out['contrasts']['C_to_M13']['panels'][m]['single_1']==original['panels'][m]['single_1']
    panels=out['contrasts'];score=lambda pair,m:panels[pair]['panels'][m]['single_1']['T2G']['macro']
    C=score('C_to_M1','C');M1=score('C_to_M1','M1');M3=score('C_to_M3','M3');M13=score('C_to_M13','M13')
    out['T2G_factorial_pp']=dict(canonical_without_penalty=M1-C,penalty_on_raw=M3-C,penalty_on_canonical=M13-M1,canonical_with_penalty=M13-M3,interaction=(M13-M1)-(M3-C))
    q.write(run/'public/FACTORIAL_RESULTS.json',out);q.write(run/'private/FACTORIAL_OWNER_AUDIT.json',owner_audits)
    q.write(run/'private/FACTORIAL_REPORT_COMPLETE1.json',dict(status='COMPLETE',primary_gate=out['primary_M3_gate'],epoch=time.time()))

if __name__=='__main__':main()
