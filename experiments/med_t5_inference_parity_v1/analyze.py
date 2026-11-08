"""CPU comparison of private artifacts; publish only cohort-level diagnostics."""
from collections import Counter
import json
import os
from pathlib import Path
import re
import time
import torch


def read(p):return json.loads(Path(p).read_text())


def compare(a,b):
    if a is None or b is None:return dict(both_missing=a is None and b is None,exact=a is None and b is None)
    if a.shape!=b.shape:return dict(exact=False,shape_equal=False,A=list(a.shape),B=list(b.shape))
    x,y=a.double(),b.double();finite=torch.isfinite(x)&torch.isfinite(y)
    d=(x[finite]-y[finite]);xx=x[finite]
    return dict(shape_equal=True,same_dtype=a.dtype==b.dtype,exact=torch.equal(a,b),
        nonfinite_pattern_equal=bool(((torch.isfinite(x)==torch.isfinite(y)) & (torch.isfinite(x)|(x==y))).all()),
        max_abs=float(d.abs().max()) if len(d) else 0,
        relative_l2=float(d.norm()/xx.norm().clamp_min(1e-30)))


def prefix(answer):
    m=re.match(r'\s*(yes|no)\b',answer,re.I)
    return m.group(1).lower() if m else 'other'


def main():
    cfg=read(os.environ['DIAG_CONFIG']);r=Path(cfg['root']);panel=read(r/'PANEL.private.json')['queries'];records=[]
    for i,q in enumerate(panel):
        a=read(r/'A/answers'/f'{i:02d}.json');b=read(r/'B/answers'/f'{i:02d}.json')
        ta=torch.load(r/'A/tensors'/f'{i:02d}.pt',map_location='cpu',weights_only=True)
        tb=torch.load(r/'B/tensors'/f'{i:02d}.pt',map_location='cpu',weights_only=True)
        measurements={name:compare(ta.get(name),tb.get(name)) for name in sorted(set(ta)|set(tb))}
        pa=ta['first_logits'].float().softmax(-1);pb=tb['first_logits'].float().softmax(-1)
        measurements['first_logits'].update(argmax_equal=bool(torch.equal(pa.argmax(-1),pb.argmax(-1))),
            KL_A_to_B=float((pa*(pa.clamp_min(1e-30).log()-pb.clamp_min(1e-30).log())).sum()))
        records.append(dict(index=i,role=q['role'],presence_code=q['presence_code'],token_equal=a['tokens']==b['tokens'],
            answer_equal=a['answer']==b['answer'],A_matches_historical=a['tokens']==q['historical_tokens'],
            A_prefix=prefix(a['answer']),B_prefix=prefix(b['answer']),A_EOS=a['EOS'],B_EOS=b['EOS'],
            A_at_cap=a['at_cap'],B_at_cap=b['at_cap'],prompt_equal=a['prompt']==b['prompt'],measurements=measurements))
    (r/'COMPARISON.private.json').write_text(json.dumps(records,indent=2)+'\n')
    def summarize(rows):
        out=dict(n=len(rows),tokens_equal=sum(x['token_equal'] for x in rows),answers_equal=sum(x['answer_equal'] for x in rows),
            A_historical_token_matches=sum(x['A_matches_historical'] for x in rows),prompts_equal=sum(x['prompt_equal'] for x in rows),
            A_prefix=dict(Counter(x['A_prefix'] for x in rows)),B_prefix=dict(Counter(x['B_prefix'] for x in rows)),
            A_EOS=sum(x['A_EOS'] for x in rows),B_EOS=sum(x['B_EOS'] for x in rows),A_at_cap=sum(x['A_at_cap'] for x in rows),B_at_cap=sum(x['B_at_cap'] for x in rows))
        out['tensor_comparisons']={name:dict(exact=sum(x['measurements'][name].get('exact',False) for x in rows),
            both_missing=sum(x['measurements'][name].get('both_missing',False) for x in rows),
            maximum_abs_difference=max((x['measurements'][name].get('max_abs',0) for x in rows),default=None),
            maximum_relative_l2=max((x['measurements'][name].get('relative_l2',0) for x in rows),default=None)) for name in records[0]['measurements']}
        out['first_logit_argmax_equal']=sum(x['measurements']['first_logits']['argmax_equal'] for x in rows)
        out['first_logit_KL_max']=max((x['measurements']['first_logits']['KL_A_to_B'] for x in rows),default=None)
        return out
    result=dict(status='COMPLETE_ENGINEERING_DIAGNOSTIC_NOT_CORE_EVALUATION',time=time.time(),
        full_generations=64,updates=0,Judge_calls=0,eligibility_changed=False,labels_changed=False,
        all=summarize(records),by_role={role:summarize([x for x in records if x['role']==role]) for role in ['T0','T5']},
        by_presence_code={code:summarize([x for x in records if x['presence_code']==code]) for code in ['11','01','10']},
        clean_base={arm:read(r/arm/'CLEAN_BASE.private.json') for arm in ['A','B']},
        generation_parameter_dtypes={arm:read(r/arm/'answers/00.json')['parameter_dtypes'] for arm in ['A','B']},
        limitations=['A/B differ in loader/entry and precision; output differences do not isolate either factor.',
            'Cross-precision logits are descriptive; no posthoc tolerance or higher-accuracy implementation selection.',
            'Includes one known source CT-comparison concern for engineering only; no clinical correctness claim.',
            'Deterministic per-image preprocessing seed shared; actual pixel tensors compared.'])
    (r/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    (r/'COMPLETE.json').write_text(json.dumps(dict(status=result['status'],time=time.time()))+'\n')


if __name__=='__main__':main()
