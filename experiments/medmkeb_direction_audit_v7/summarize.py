"""Sanitized full-candidate audit aggregation; qualification is not effectiveness."""
import argparse,json
from pathlib import Path

def summarize(private):
    read=lambda p:json.loads(p.read_text())
    tasks=[];receipts=[]
    for i in (0,1):
        part=private/'remote'/str(i);r=read(part/'AUDIT_COMPLETE.json');e=read(part/'EXIT.json');admission=read(part/'ENGINEERING_ADMISSION.json')
        assert e['code']==0 and r['status']=='COMPLETE' and admission['status']=='PASS'
        rows=r['candidates'];assert len(rows)==24 and {(v['kind'],v['scale']) for v in rows}=={(k,2.**-j) for k in ('SOURCE','PROJECT','NORM_SOURCE') for j in range(8)}
        pair=[]
        for j in range(8):
            scale=2.**-j;a=next(v for v in rows if v['kind']=='PROJECT' and v['scale']==scale);b=next(v for v in rows if v['kind']=='NORM_SOURCE' and v['scale']==scale)
            pair.append(dict(scale=scale,project_actual_feasible=a['actual_feasible'],source_control_actual_feasible=b['actual_feasible'],project_gain=a['source_gain'],source_control_gain=b['source_gain'],gain_difference=a['source_gain']-b['source_gain'],norm_difference=a['actual_operator_norm']-b['actual_operator_norm']))
        tasks.append(dict(task=r['task'],engineering='PASS',support_initial_max_difference=admission['support_initial_difference'],zero_constraint_state_identity=admission['zero_constraint_state_identity'],QP=r['qp'],candidate_qualified=r['candidate_qualified'],qualified_scales=r['qualified_scales'],full_source_gain=r['full_source_gain'],candidates=rows,matched_norm_pairs=pair));receipts.append(e)
    counts={k:sum(v['counts'].get(k,0) for v in receipts) for k in set().union(*(v['counts'] for v in receipts))}
    assert counts['forwards']<=3000 and counts['backwards']<=120 and counts['candidate_checks']==48 and counts['updates']==counts['generations']==counts['operator_backwards']==0
    return dict(status='COMPLETE_ENGINEERING_AUDIT',tasks=tasks,counts=counts,gpu_process_hours=sum(v['seconds'] for v in receipts)/3600,semantic_protection_effectiveness_tested=False,independent_confirmation=False,metric='Euclidean fixed canonical parameter frame, not operator invariant',prior_V6_unchanged=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(summarize(a.private),indent=2)+'\n')
