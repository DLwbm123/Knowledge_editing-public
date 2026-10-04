"""Read-only generation audit. Text matching is never an Astra verdict."""
import json
import os
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def normalized(text):
    return ' '.join(text.lower().split())


def summarize(root):
    c=read(root/'CONFIG.private.json')
    report=dict(run_id=c['run_id'],N=146,freeze_id=c['freeze_id'],result=read(root/'RESULT.json'),
                launch=read(root/'LAUNCH.json'),statistics=read(root/'statistics/STATUS.json'),arms={},
                metrics_status='PENDING_ASTRA_HIGH',
                exact_match_definition='lowercase and whitespace normalization only; not semantic correctness')
    base={v['query_id']:v for v in (json.loads(line) for line in (root/'statistics/BASE_OUTPUTS.private.jsonl').read_text().splitlines())}
    report['Base_native_text_exact']=sum(normalized(base[e]['raw_answer'])==normalized(c['queries'][e]['reference']) for e in c['main_T0'])
    for arm in c['arms']:
        a=dict(layers=c['arms'][arm]['layers'],role=c['arms'][arm]['role'],status=read(root/arm/'STATUS.json'),phases={})
        for phase in ('single','sequential'):
            d=root/arm/phase
            receipts=[read(p) for p in sorted(d.glob('*.RECEIPT.json'))]
            outputs=[read(p) for p in sorted(d.glob('*.OUTPUT.private.json'))]
            if len(receipts)!=146 or [o['edit_id'] for o in outputs]!=c['main_T0']:
                raise ValueError('Incomplete or reordered cohort')
            for index,output in enumerate(outputs,1):
                if phase=='single':expected=set(c['tasks'][index-1]['query_ids'])
                elif index in c['prefixes']:expected={q for t in c['tasks'][:index] for q in t['query_ids']}
                else:expected={output['edit_id']}
                if set(output['outputs'])!=expected:raise ValueError('Output panel coverage mismatch')
            generated=[v for o in outputs for v in o['outputs'].values()]
            entry=dict(completed=len(receipts),completion=read(d/'COMPLETE.json'),output_files=len(outputs),
                generation_rows=len(generated),query_counts=[len(o['outputs']) for o in outputs],
                reload_checked=sum(v['clean_reload'] is not None for v in receipts),
                reload_pass=sum(v['clean_reload'] is not None and v['clean_reload']['logit_max_error']==0
                    and v['clean_reload']['generation_identical'] and not v['clean_reload']['hooks']
                    and not v['clean_reload']['algorithm_loaded'] for v in receipts),
                case_seconds=sum(v['seconds'] for v in receipts),
                empty_answers=sum(not v['raw_answer'].strip() for v in generated),
                hit_token_cap=sum(v['hit_token_cap'] for v in generated),
                tokens=sum(len(v['raw_token_ids']) for v in generated),
                native_insertion_text_exact=sum(normalized(o['outputs'][o['edit_id']]['raw_answer'])==
                    normalized(c['queries'][o['edit_id']]['reference']) for o in outputs))
            if phase=='sequential':
                entry['prefixes']={}
                for n in c['prefixes']:
                    out=outputs[n-1]['outputs'];ids=c['main_T0'][:n]
                    entry['prefixes'][str(n)]=dict(unique_queries=len(out),native_count=n,
                        native_text_exact=sum(normalized(out[e]['raw_answer'])==normalized(
                            c['active_targets'][str(n)].get(e,c['queries'][e]['reference'])) for e in ids))
            deletions=[read(p) for p in d.glob('*.DELETION.json')]
            if (d/'DELETION.json').exists():deletions.append(read(d/'DELETION.json'))
            entry['cleanup_receipts']=len(deletions);entry['current_pt_files']=len(list(d.glob('*.pt')))
            a['phases'][phase]=entry
        report['arms'][arm]=a
    report['projection_deletion']=read(root/'statistics/DELETION.json')
    report['remaining_projection_pt']=len(list((root/'statistics').glob('basis-*.private.pt')))
    return report


if __name__=='__main__':
    print(json.dumps(summarize(Path(os.environ['CAMPAIGN_DIR'])),indent=2,allow_nan=False))
