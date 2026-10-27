"""Registered CPU-only candidate log diagnostic; never imports model code."""
import json,argparse
from pathlib import Path

def run(root):
    read=lambda p:json.loads(p.read_text());anchor=read(root/'REFERENCE_POOL.json')['nll'];rows=[]
    for i in (0,1):
        for p in sorted((root/'remote'/str(i)/'candidates').glob('step*.json')):
            v=read(p);x=v['private_metrics']['support_nll'];assert len(x)==54
            ce=[x[j]-anchor[j] for j in range(0,54,2)];margin=[anchor[j+1]-anchor[j]-(x[j+1]-x[j]) for j in range(0,54,2)]
            rows.append(dict(edit=i+1,**v['summary'],max_CE=max(ce),max_margin=max(margin),CE_violations=sum(z>1e-4 for z in ce),margin_violations=sum(z>1e-4 for z in margin)))
    assert len(rows)<=600
    last=sorted([x for x in rows if x['edit']==2 and x['step']==21],key=lambda x:-x['scale']);assert len(last)==8
    baseline=read(root/'remote/1/PROJECT_COMPLETE.json')['curve'][-1]['max_support_regression']
    return dict(protocol_commit='27b8b8e',candidate_rows=len(rows),all_violating_candidates=sum(x['max_support_regression']>1e-4 for x in rows),positive_source_gain_all=all(x['source_gain']>0 for x in rows),stopping_step=last,previous_accepted_max_regression=baseline,shrinking_is_monotone=all(y['max_support_regression']<=x['max_support_regression'] for x,y in zip(last,last[1:])),smallest_scale_increase_over_accepted=last[-1]['max_support_regression']-baseline,new_model_calls=0,new_updates=0,causal_mechanism_identified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(run(a.private),indent=2)+'\n')
