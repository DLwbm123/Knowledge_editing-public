"""New reserve trajectory with immutable RAW_QP controls and explicit missingness."""
import argparse,json
from pathlib import Path

def summarize(private):
    read=lambda p:json.loads(p.read_text());tasks=[];exits=[]
    for i in (0,1):
        d=private/'remote'/str(i);end=read(d/'EXIT.json');v=read(d/'STUDY_COMPLETE.json');a=read(d/'ENGINEERING_ADMISSION.json');old=read(private/'RAW_REFERENCE'/str(i)/'PROJECT_COMPLETE.json')
        assert end['code']==0 and v['save_load_exact'] and a['status']=='PASS' and v['inner']==5e-5 and v['outer']==1e-4
        assert len(v['curve'])==v['accepted_steps'] and v['remaining_NA']==31-v['accepted_steps']
        assert all(x['max_support_regression']<=1e-4 and x['actual_operator_norm']>0 for x in v['curve'])
        assert len(v['curve'])==31 or not v['trajectory_qualified']
        c=end['counts'];assert c['forwards']<=15500 and c['backwards']<=1800 and c['candidate_checks']<=248 and c['updates']==len(v['curve']) and c['reused_gradient_sequences']==55 and c['generations']==0
        n=min(len(old['curve']),len(v['curve']));common=None if n==0 else dict(steps=n,reserve_source_gain=v['curve'][0]['source_loss_before']-v['curve'][n-1]['source_loss_after'],RAW_source_gain=old['curve'][0]['source_loss_before']-old['curve'][n-1]['source_loss_after'])
        tasks.append(dict(edit=i+1,status=v['status'],accepted_steps=v['accepted_steps'],remaining_NA=v['remaining_NA'],source_gain=v['source_gain'],reference_NONE31_gain=v['reference_NONE31_gain'],trajectory_qualified=v['trajectory_qualified'],curve=v['curve'],RAW_accepted_steps=old['accepted_steps'],RAW_remaining_NA=old['remaining_steps'],common_prefix=common));exits.append(end)
    counts={k:sum(x['counts'].get(k,0) for x in exits) for k in set().union(*(x['counts'] for x in exits))}
    return dict(status='COMPLETE_RESERVE_CONSUMERS',tasks=tasks,counts=counts,gpu_process_hours=sum(x['seconds'] for x in exits)/3600,both_trajectory_qualified=all(x['trajectory_qualified'] for x in tasks),semantic_effectiveness_tested=False,independent_confirmation=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(summarize(a.private),indent=2)+'\n')
