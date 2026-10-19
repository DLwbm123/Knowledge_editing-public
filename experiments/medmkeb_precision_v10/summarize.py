"""Complete two-precision/two-readout diagnostics; no raw logits or medical data."""
import json,argparse
from pathlib import Path

def summarize(private):
    read=lambda p:json.loads(p.read_text());rows=[];exits=[]
    for i in (0,1):
        p=private/'remote'/str(i);e=read(p/'EXIT.json');a=read(p/'AUDIT_COMPLETE.json');assert e['code']==0 and a['original_FP32_reproduction'] and a['loaded_state_exact'] and a['capture_max_error']<=1e-6
        assert len(a['summaries'])==32 and len({(x['precision'],x['readout'],x['scale']) for x in a['summaries']})==32
        c=e['counts'];assert c['forwards']<=1200 and c['backwards']==c['updates']==c['generations']==0 and c['candidate_checks']==20
        rows.append(dict(edit=i+1,capture_max_error=a['capture_max_error'],summaries=a['summaries']));exits.append(e)
    counts={k:sum(x['counts'].get(k,0) for x in exits) for k in set().union(*(x['counts'] for x in exits))}
    return dict(status='COMPLETE_PRECISION_CONSUMERS',tasks=rows,counts=counts,gpu_process_hours=sum(x['seconds'] for x in exits)/3600,primary_criterion_unchanged=True,semantic_effectiveness_tested=False,full_Transformer_FP64=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.write_text(json.dumps(summarize(a.private),indent=2)+'\n')
