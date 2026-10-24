"""All accepted/rejected trajectory prefixes and missingness, no semantic claims."""
import argparse,json
from pathlib import Path

def summarize(private):
    read=lambda p:json.loads(p.read_text());rows=[];exits=[]
    for i in (0,1):
        root=private/'remote'/str(i);exit=read(root/'EXIT.json');admission=read(root/'REUSE_ADMISSION.json');result=read(root/'TRAJECTORY_COMPLETE.json');a=read(root/'PROJECT_COMPLETE.json');b=read(root/'NORM_COMPLETE.json')
        assert exit['code']==0 and admission['status']=='PASS' and a['save_load_exact'] and b['save_load_exact']
        assert len(a['curve'])==len(b['curve'])==result['accepted_steps'] and result['remaining_steps']==31-result['accepted_steps']
        assert a['status'] in ('COMPLETE','NO_FEASIBLE_LEARNING_STEP') and all(x['max_support_regression']<=1e-4 and x['actual_norm']>0 for x in a['curve'])
        assert all(abs(x['actual_norm']-y['actual_norm'])<=1e-10+.002*x['actual_norm'] for x,y in zip(a['curve'],b['curve']))
        assert result['accepted_steps']==31 or not result['trajectory_qualified']
        rows.append(dict(result,project_curve=a['curve'],control_curve=b['curve'],reuse_admission=admission));exits.append(exit)
    counts={k:sum(v['counts'].get(k,0) for v in exits) for k in set().union(*(v['counts'] for v in exits))}
    assert counts['forwards']<=36000 and counts['backwards']<=3600 and counts['candidate_checks']<=600 and counts['updates']<=120 and counts['logical_transitions']<=124 and counts['reused_transitions']==4 and counts['generations']==0
    return dict(status='COMPLETE_PREFIX_QUALIFICATION',tasks=rows,counts=counts,gpu_process_hours=sum(x['seconds'] for x in exits)/3600,both_tasks_trajectory_qualified=all(x['trajectory_qualified'] for x in rows),semantic_effectiveness_tested=False,independent_confirmation=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(summarize(a.private),indent=2)+'\n')
