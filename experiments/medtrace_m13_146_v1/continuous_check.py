"""Check conditional boundaries and reproduce an existing actual-bank report."""
import copy
import os
from pathlib import Path
import sqlite3
import time
import continuous_protocol as f
import continuous_report as report

def main():
    import common as c
    run=Path(os.environ['RUN_ROOT']);single=Path(f.read(run/'private/CONTINUOUS_ADMISSION.json')['single_run'])
    baseline=Path(f.read(single/'private/CURRICULUM_ADMISSION.json')['evaluation_parent'])
    old=f.read(baseline/'public/BALANCE_RESULTS.json')
    # Test only the new admission predicate, using known protected results and synthetic boundary values.
    stub=dict(primary_gate=dict.fromkeys(('complete_no_missing','FIT','T1G','T1L','T2L','damage'),True),contrasts={'P160_to_M13P':{'panels':{'M13P':{'single_1':{'T2G':{'macro':80.}}}}}})
    assert not f.gate(stub)
    stub['contrasts']['P160_to_M13P']['panels']['M13P']['single_1']['T2G']['macro']=80.0001;assert f.gate(stub)
    for key in stub['primary_gate']:
        changed=copy.deepcopy(stub);changed['primary_gate'][key]=False;assert not f.gate(changed)
    plan=f.read(run/'private/EVAL_PLAN146.json');assert len(plan)==9587
    assert len([v for v in plan if v['mode']=='insertion'])==146
    assert {v['prefix'] for v in plan if v['mode']=='bank'}=={1,50,100,146}
    assert [len([v for i,v in enumerate(f.read(run/'private/CONTINUOUS_NEW_PLAN.json')) if i%2==p]) for p in (0,1)]==[2269,2269]
    cm=Path(os.environ['CONTINUOUS_CHECK_PARENT']);records={};scores={}
    with sqlite3.connect('file:'+str(cm/'private/judge_cm146_astra_medium/queue.sqlite')+'?mode=ro',uri=True) as db:
        for path,status,score in db.execute('select c.path,p.status,p.correct from consumer c join payload p on p.key=c.payload_key'):
            d=f.read(Path(path))
            if d['method'] not in ('M13','BASE'):continue
            method='M13P' if d['method']=='M13' else 'BASE';k=(method,d['mode'],d['prefix'],d['owner'],d['binding']['input']['query_id'])
            assert k not in records;records[k]=d;scores[k]=score if status=='FORMAT_VALID' else None
    result,owners=report.calculate(f.read(run/'private/ORIGINAL146_TASKS.json'),f.read(run/'private/ORIGINAL_BASE_MASK146.json'),records,scores)
    prior=f.read(cm/'public/CM_RESULTS.json')
    for label in ('single_1','bank_50','bank_100','bank_146'):
        assert result['panels'][label]==prior['panels']['M13'][label]
        for key in ('fresh_Base_correct','new_damage','missing'):assert result['protection'][label][key]==prior['protection']['M13_'+label][key]
    assert result['insertion']['observations']==146
    receipt=dict(status='PASS',strict_above_80_boundary=True,all_protection_terms_required=True,consumers=9587,new_consumers=4538,partitions=[2269,2269],existing_bank_panels_and_damage_reproduced=True,insertion_supported=True,new_model_calls=0,epoch=time.time())
    c.write(run/'private/CONTINUOUS_CPU_CHECK.json',receipt);print(receipt)
if __name__=='__main__':main()
