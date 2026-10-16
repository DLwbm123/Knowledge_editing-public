"""Full17081 synthetic report checks; no models, scoring, or real outputs."""
import json
import os
from pathlib import Path
import comparison146 as contract
import selected_report146 as report
import scorer_start


def main():
    packet=json.loads(Path(os.environ['M13_CPU_PACKET']).read_text());plan=contract.coverage(packet)
    records={};scores={}
    for row in plan:
        key=tuple(row[k] for k in ('method','mode','prefix','owner'))+(row['query']['query_id'],)
        records[key]={'active_target':False};scores[key]=1
    result,_=report.calculate(packet['original_tasks'],packet['base_mask'],records,scores)
    assert result['complete_no_missing'] and not result['original_joint_scientific_gate_confirmed']
    assert result['original_M1_fit_T2G_preservation146'] is None and result['original_M3_damage_preservation146'] is None
    for method in contract.METHODS:
        assert result['panels'][method]['bank_146']['T2L']['observations']==54
        assert result['panels'][method]['bank_146']['T2L']['edit_units']==31
    def fit(method,owner):return next(k for k in scores if k[:4]==(method,'single',1,owner) and k[4]==f'edit_{owner}_1')
    scores[fit('M13',1)]=0;scores[fit('FISHER_MIN',2)]=0
    result,_=report.calculate(packet['original_tasks'],packet['base_mask'],records,scores)
    assert result['panels']['M13']['single_1']['FIT']['macro']==result['panels']['FISHER_MIN']['single_1']['FIT']['macro']
    assert not result['paired_native_FIT160_contrasts']['single_1']['FIT']['FISHER_MIN_preserves_each_M13_owner']
    scores[fit('FISHER_MIN',2)]=None
    result,_=report.calculate(packet['original_tasks'],packet['base_mask'],records,scores)
    assert result['missing_consumers']==1 and not result['complete_no_missing']
    assert result['panels']['FISHER_MIN']['single_1']['FIT']['macro'] is None
    assert not result['paired_native_FIT160_contrasts']['single_1']['FIT']['FISHER_MIN_preserves_each_M13_owner']
    for key in scores:
        if key[0]=='BASE':scores[key]=0
    result,_=report.calculate(packet['original_tasks'],packet['base_mask'],records,scores)
    assert all(v['fresh_Base_correct']==0 and v['correct_retention_rate'] is None for v in result['protection'].values())
    assert all(v['FISHER_MIN_damage_no_worse_than_M13_per_owner'] is None for v in result['paired_native_FIT160_contrasts'].values())
    cfg=scorer_start.configs(Path('/fixture'),'fixture','fixture','fixture',queue_file='selected_queue146.py',
        result_folder='judge_native146_two_methods_astra_medium',epoch='MEDTRACE_NATIVE146_TWO_METHODS_20261015_V1')
    assert len(cfg)==4 and len({v['SCORER_STATE'] for v in cfg})==4
    assert all(v['JUDGE_QUEUE_FILE']=='selected_queue146.py' and v['JUDGE_EFFORT']=='medium'
        and v['JUDGE_EPOCH']=='MEDTRACE_NATIVE146_TWO_METHODS_20261015_V1' for v in cfg)
    checks=dict(status='PASS',synthetic_consumers=17081,original_T2L_31_edits_54_probes=True,
        equal_averages_do_not_hide_owner_regression=True,missing_not_valid_complete=True,
        absent_M1_M3_controls_not_scientific_success=True,zero_Base_support_is_NA=True,
        four_selected_scorer_configs_and_epoch=True,new_GPU=0,new_VLM=0,new_Judge=0,execution_admitted=False)
    Path(os.environ['M13_CPU_REPORT']).write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))


if __name__=='__main__':main()
