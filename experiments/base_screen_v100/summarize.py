"""Recompute frozen admission from complete private, full-text adjudications."""
import collections
import json
from pathlib import Path
import sys

ROLES = ['target','brief_control','negative_direct','negative_reverse','positive_direct','positive_reverse']


def summarize(rows, review):
    byid={r['qid']:r for r in review}
    assert len(byid)==len(rows)==90 and set(byid)=={r['qid'] for r in rows}
    groups=collections.defaultdict(dict)
    for row in rows:
        v=byid[row['qid']]
        assert v['core'] in ['CORRECT','WRONG','UNKNOWN']
        assert v['role']==row['role']
        groups[row['source_group']][row['role']]=dict(row,verdict=v)
    def okay(r):return r['verdict']['core']=='CORRECT' and not r['verdict']['unsupported_medical_addition']
    ids={k:[] for k in ['persistent','negative','positive','discriminative','operational','strong']}
    for group,g in groups.items():
        assert set(g)==set(ROLES)
        err=all(g[r]['verdict']['core']=='WRONG' for r in ['target','brief_control'])
        neg=all(okay(g[r]) for r in ['negative_direct','negative_reverse'])
        pos=all(okay(g[r]) for r in ['positive_direct','positive_reverse'])
        for key,flag in dict(persistent=err,negative=neg,positive=pos,discriminative=neg and pos,operational=err and neg,strong=err and neg and pos).items():
            if flag:ids[key].append(group)
    coverage=dict(target_answers=sorted({groups[g]['target']['reference'] for g in ids['strong']}),
        target_findings=sorted({groups[g]['target']['target_label'] for g in ids['strong']}),
        protection_findings=sorted({groups[g]['target']['protection_label'] for g in ids['strong']}))
    core={r:sum(g[r]['verdict']['core']=='CORRECT' for g in groups.values()) for r in ROLES}
    balanced={f:(core['negative_'+f]+core['positive_'+f])/30 for f in ['direct','reverse']}
    categories=sorted({groups[g]['target']['protection_label'] for g in ids['discriminative']})
    result=dict(core_correct=core,unknown={r:sum(g[r]['verdict']['core']=='UNKNOWN' for g in groups.values()) for r in ROLES},
        fully_supported={r:sum(okay(g[r]) for g in groups.values()) for r in ROLES},
        counts={k:len(v) for k,v in ids.items()},strong_coverage=coverage,balanced_accuracy=balanced,
        discriminative_categories=categories,discrimination_signal=min(balanced.values())>=.75 and len(categories)>=2,
        migration_admitted=len(ids['strong'])>=8 and all(len(v)>=2 for v in coverage.values()))
    return result,ids


def main(private, previous, destination):
    private,previous,destination=map(Path,[private,previous,destination])
    rows=json.loads((private/'SOURCE_ROWS.json').read_text())
    results={};qualified={}
    for key in ['old','Q','L']:
        path=previous/'FULL_TEXT_REVIEW.json' if key=='old' else private/(key+'_REVIEW.json')
        review=json.loads(path.read_text())
        if key!='old':
            outputs={p.stem:json.loads(p.read_text()) for p in (private/'results'/key/'outputs').glob('*.json')}
            assert len(outputs)==90
            for r in review:
                o=outputs[str(r['qid'])]
                assert o['raw_answer']==r['answer']
                assert o['query_id']==next(x['query_id'] for x in rows if x['qid']==r['qid'])
                if o['at_cap'] or not o['EOS']:assert r['core']=='UNKNOWN'
            completion=json.loads((private/'results'/key/'COMPLETE.json').read_text())
            assert completion['counts']['updates']==completion['counts']['backwards']==0
        results[key],qualified[key]=summarize(rows,review)
        if key!='old':
            core_review=[dict(r,unsupported_medical_addition=False) for r in review]
            core_result,_=summarize(rows,core_review)
            results[key]['descriptive_core_only_discriminative_pairs']=core_result['counts']['discriminative']
            results[key]['descriptive_core_only_strong_pairs']=core_result['counts']['strong']
            results[key]['execution']={k:v for k,v in completion.items() if k!='time'}
    admitted=[k for k in ['Q','L'] if results[k]['migration_admitted']]
    admitted.sort(key=lambda k:(-results[k]['counts']['strong'],-min(results[k]['balanced_accuracy'].values()),k!='Q'))
    out=dict(status='MIGRATION_CANDIDATE_ADMITTED' if admitted else 'NO_MIGRATION_ADMISSION',selected=admitted[0] if admitted else None,
        pairs=15,patients=30,questions_per_model=90,old_outputs_reused=90,new_generations=180,models=results,method_training=False)
    out.update(source_candidates=18,source_excluded=3)
    completion=json.loads((private/'results/COMPLETE.json').read_text())
    all_outputs=[json.loads(p.read_text()) for key in ['Q','L'] for p in (private/'results'/key/'outputs').glob('*.json')]
    assert len(all_outputs)==180 and all(x['vision_calls']==1 for x in all_outputs)
    out['execution']=dict(status=completion['status'],gpu_process_seconds=completion['gpu_process_seconds'],
        generations=len(all_outputs),forwards=sum(x['forwards'] for x in all_outputs),vision_calls=sum(x['vision_calls'] for x in all_outputs),
        backwards=0,updates=0,all_EOS=all(x['EOS'] for x in all_outputs),truncated=sum(x['at_cap'] for x in all_outputs),
        retries=0,visual_tokens_min=min(x['visual_tokens'] for x in all_outputs),visual_tokens_max=max(x['visual_tokens'] for x in all_outputs),
        model_loading_memory_MiB={key:int(json.loads((private/'results'/key/'ENVIRONMENT.json').read_text())['gpu_processes'][0].split(',')[-1].strip().split()[0]) for key in ['Q','L']},
        memory_note='Single post-loading reading, not full runtime peak')
    (private/'QUALIFIED_GROUPS_PRIVATE.json').write_text(json.dumps(qualified,indent=2)+'\n')
    destination.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main(*sys.argv[1:])
