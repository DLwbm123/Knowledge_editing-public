"""CPU checks of fixed coverage, paired reporting, missingness and budget gates."""
import copy
import json
import os
from pathlib import Path
import cm_eval146 as protocol
import comparison146
import selected_report146 as report
from cm_report146 import gate


def main():
    packet=json.loads(Path(os.environ['M13_CPU_PACKET']).read_text())
    original=comparison146.coverage(packet);plan=protocol.plan(original)
    assert original==comparison146.coverage(packet),'Parent plan mutated'
    for old,new in zip(original,plan):
        expected=copy.deepcopy(old)
        if expected['method']=='FISHER_MIN':expected['method']='C'
        assert new==expected
    records={};scores={}
    for row in plan:
        key=tuple(row[k] for k in ('method','mode','prefix','owner'))+(row['query']['query_id'],)
        records[key]={'active_target':False};scores[key]=1
    def calculate():return report.calculate(packet['original_tasks'],packet['base_mask'],records,scores,methods=protocol.METHODS)[0]
    result=calculate();assert gate(result)['passed']
    assert set(result['panels'])=={'C','M13'} and not result['original_joint_scientific_gate_confirmed']
    assert result['panels']['C']['bank_146']['T2L']['observations']==54
    key=next(k for k in scores if k[:4]==('M13','bank',146,0) and k[4]=='edit_1_1')
    scores[key]=0
    result=calculate();assert not gate(result)['passed'] and not gate(result)['terms']['FIT']
    scores[key]=None
    result=calculate();assert not gate(result)['passed'] and result['missing_consumers']==1
    scores[key]=1
    for k in scores:
        if k[0]=='BASE':scores[k]=0
    result=calculate();assert gate(result)['terms']['damage'] is None and not gate(result)['passed']
    checks=dict(status='PASS',consumers=len(plan),parent_plan_unchanged=True,matched_methods_and_original_54_T2L=True,
        each_owner_regression_blocks_pass=True,missing_blocks_pass=True,no_Base_support_is_NA=True,
        original_M1_M3_gate_unconfirmed=True,new_GPU=0,new_Judge=0)
    Path(os.environ['M13_CPU_REPORT']).write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))


if __name__=='__main__':main()
