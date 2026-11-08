"""Describe longitudinal tensor changes from existing captures; zero generation."""
from collections import defaultdict
import json
import os
from pathlib import Path
import runpy
import torch


def main():
    cfg=json.loads(Path(os.environ['DIAG_CONFIG']).read_text());r=Path(cfg['root'])
    compare=runpy.run_path(str(r/'tools/analyze.py'))['compare']
    panel=json.loads((r/'PANEL.private.json').read_text())['queries'];groups=defaultdict(dict)
    for i,q in enumerate(panel):groups[q['edit_id']][q['role']]=(i,q)
    assert len(groups)==16 and all(set(g)=={'T0','T5'} for g in groups.values())
    allrows={}
    for arm in ['A','B']:
        rows=[]
        for g in groups.values():
            i,q=g['T0'];j,_=g['T5']
            before=torch.load(r/arm/'tensors'/f'{i:02d}.pt',map_location='cpu',weights_only=True)
            after=torch.load(r/arm/'tensors'/f'{j:02d}.pt',map_location='cpu',weights_only=True)
            a=json.loads((r/arm/'answers'/f'{i:02d}.json').read_text());b=json.loads((r/arm/'answers'/f'{j:02d}.json').read_text())
            c={name:compare(before[name],after[name]) for name in ['pixels','vision','projector','fused','first_logits']}
            pa=before['first_logits'].float().softmax(-1);pb=after['first_logits'].float().softmax(-1)
            rows.append(dict(code=q['presence_code'],measurements=c,answer_equal=a['answer']==b['answer'],tokens_equal=a['tokens']==b['tokens'],
                first_probability_KL=float((pa*(pa.clamp_min(1e-30).log()-pb.clamp_min(1e-30).log())).sum())))
        allrows[arm]=rows
    (r/'PAIRED.private.json').write_text(json.dumps(allrows,indent=2)+'\n')
    def aggregate(rows):
        return dict(n=len(rows),same_answer=sum(x['answer_equal'] for x in rows),same_tokens=sum(x['tokens_equal'] for x in rows),
            tensors={name:dict(different=sum(not x['measurements'][name]['exact'] for x in rows),
                min_relative_l2=min(x['measurements'][name]['relative_l2'] for x in rows),
                max_relative_l2=max(x['measurements'][name]['relative_l2'] for x in rows)) for name in ['pixels','vision','projector','fused','first_logits']},
            first_probability_KL_min=min(x['first_probability_KL'] for x in rows),first_probability_KL_max=max(x['first_probability_KL'] for x in rows))
    result=dict(status='PAIRED_ENGINEERING_DESCRIPTION',new_generations=0,new_Judge=0,
        arms={arm:dict(all=aggregate(rows),by_presence_code={code:aggregate([x for x in rows if x['code']==code]) for code in ['11','01','10']}) for arm,rows in allrows.items()},
        limitations=['Different tensors/logits show computational dependence, not correct clinical grounding.',
            'Includes known source CT-comparison concern; no label-accuracy or qualification changes.',
            'A first-input initialization ordering deviation is recorded separately; B is an independent native process.'])
    (r/'PAIRED_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
