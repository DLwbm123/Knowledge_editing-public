"""Complete factorial consumers and full cost accounting; raw metrics stay private."""
import json,argparse
from pathlib import Path

def summarize(private):
    read=lambda p:json.loads(p.read_text());tasks=[];exits=[]
    for i in (0,1):
        p=private/'remote'/str(i);end=read(p/'EXIT.json');audit=read(p/'AUDIT_COMPLETE.json');recon=read(p/'RECONSTRUCTION_COMPLETE.json')
        assert end['code']==0 and recon['status']=='PASS' and audit['save_load_exact']
        rows=audit['candidates'];assert len(rows)==32 and len({(x['path'],x['grad_enabled'],x['scale']) for x in rows})==32
        assert all(x['norm_exact'] for x in recon['parity']) and len(recon['parity'])==19
        assert end['counts']['updates']==19 and end['counts']['reused_transitions']==1
        assert end['counts']['forwards']<=3500 and end['counts']['backwards']<=1200 and end['counts']['candidate_checks']<=40 and end['counts']['generations']==0
        grid={}
        for path in ('LEGACY','DIRECT'):
            for mode in (False,True):
                selected=[x for x in rows if x['path']==path and x['grad_enabled']==mode];grid[path+'_'+str(mode)]=dict(qualified_scales=[x['scale'] for x in selected if x['qualified']],max_regression=max(x['max_support_regression'] for x in selected),min_regression=min(x['max_support_regression'] for x in selected))
        tasks.append(dict(edit=i+1,baseline=audit['baseline'],factorial=grid,candidates=rows,repeat_cache_pass=audit['mechanical_repeat_cache_pass'],fixed_candidate_qualified=audit['fixed_candidate_qualified'],reconstruction_updates=19,endpoint_gradient_vs_old_no_grad=recon['endpoint_gradient_vs_old_no_grad']));exits.append(end)
    counts={k:sum(x['counts'].get(k,0) for x in exits) for k in set().union(*(x['counts'] for x in exits))}
    return dict(status='COMPLETE_FORENSIC_CONSUMERS',tasks=tasks,counts=counts,gpu_process_hours=sum(x['seconds'] for x in exits)/3600,both_fixed_candidates_qualified=all(x['fixed_candidate_qualified'] for x in tasks),semantic_effectiveness_tested=False,independent_confirmation=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.write_text(json.dumps(summarize(a.private),indent=2)+'\n')
