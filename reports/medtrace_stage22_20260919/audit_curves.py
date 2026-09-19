"""Read-only CPU audit; does not load a model or regenerate any old outcome."""
import json,statistics,sys
from pathlib import Path
from collections import defaultdict,Counter
import torch


def summary(values):
    return dict(n=len(values),mean=statistics.mean(values) if values else None,min=min(values) if values else None,max=max(values) if values else None)


def audit(old,current):
    results=[]
    paths=[('FACT',p) for p in sorted((old/'private/edits').glob('e*/C_FACT/latest.pt'))]
    paths += [('NO_H',p) for root in (old/'private/ablation/private/edits',current/'private/edits') for p in sorted(root.glob('e*/C_NO_H/latest.pt'))]
    seen=set()
    for arm,path in paths:
        state=torch.load(path,map_location='cpu',weights_only=True);position=int(path.parent.parent.name[1:]);identity=(arm,position)
        assert identity not in seen and state['step']==320;seen.add(identity)
        task=state['binding']['task'];curve=state['curve'];assert [x['step'] for x in curve]==list(range(1,321))
        terms=defaultdict(lambda:defaultdict(list));exposure=Counter()
        for row in curve:
            for name,term in row['terms'].items():
                for field in ('unweighted','weighted','weighted_gradient_norm','tokens'):terms[name][field].append(term[field])
            if row.get('extra_role')=='H_fit':exposure[task['H_fit'][row['extra_index']]['source_group']]+=1
        results.append(dict(arm=arm,position=position,source=task['native']['source_group'],writer_layer=state['binding'].get('layer_id'),terms={name:{key:summary(v) for key,v in fields.items()} for name,fields in terms.items()},preclip_total=summary([row['gradient_norm'] for row in curve]),H_fit_QA=len(task['H_fit']),H_sources=len({r['source_group'] for r in task['H_fit']}),H_source_exposures=dict(exposure),H_nonzero_steps=sum(row['terms'].get('extra',{}).get('weighted_gradient_norm',0)>0 for row in curve),gradient_cosines=None,cosine_status='Not observed in historical records; norms cannot establish direction conflict',teacher_binding_count=len(state['binding']['U_teacher_bindings']),training_binding_code=state['binding']['code']))
    assert Counter(r['arm'] for r in results)=={'FACT':45,'NO_H':45}
    # Remove source identities from public aggregate; per-source private evidence remains available.
    groups=defaultdict(list)
    for row in results:groups[row['arm']].append(row)
    public={}
    for arm,rows in groups.items():
        public[arm]=dict(edits=len(rows),native_sources=len({r['source'] for r in rows}),H_source_counts=dict(Counter(r['H_sources'] for r in rows)),H_nonzero_steps=summary([r['H_nonzero_steps'] for r in rows]),preclip_total_edit_means=summary([r['preclip_total']['mean'] for r in rows]),terms={name:{field:summary([r['terms'][name][field]['mean'] for r in rows if name in r['terms']]) for field in ('unweighted','weighted','weighted_gradient_norm','tokens')} for name in sorted({k for r in rows for k in r['terms']})})
    return dict(private_edit_source_rows=results,public_summary=public,training_rerun=False,gradient_cosines='NA: not observed',new_GPU_seconds=0,new_judgments=0)


if __name__=='__main__':
    result=audit(Path('/root/rivermind-data/job-520/run'),Path('/root/rivermind-data/job-521/run'))
    print(json.dumps(result))
