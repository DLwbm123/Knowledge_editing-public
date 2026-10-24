"""Recompute public aggregate admission results from private fixed-source verdicts."""
import collections
import json
from pathlib import Path

def summarize(private):
    read=lambda name:json.loads((private/name).read_text())
    manifest=read('CANDIDATES_FROZEN.json');initial=read('CANDIDATES_BEFORE_VISUAL_REVIEW.json');review=read('BASE_VERDICTS.json')['rows']
    assert len(review)==len({v['query_id'] for v in review})==len(manifest['rows'])
    assert {v['query_id'] for v in review}=={v['query_id'] for v in manifest['rows']}
    assert len(initial['candidates'])==len(manifest['candidates'])+len(manifest['source_review_excluded'])
    labels={v['query_id']:v['label'] for v in review};assert set(labels.values())<={'CORRECT','INCORRECT','UNCERTAIN','CONTRADICTORY'}
    by_split={split:dict(collections.Counter(v['label'] for v in review if v['split']==split)) for split in ('train','eval')}
    articles={c['article'] for c in manifest['candidates'] if c['split']=='eval' and labels[c['row']['query_id']]=='CORRECT'}
    admission=read('DATA_ADMISSION.json');assert len(articles)==admission['eval_correct']
    audit=read('remote/FINAL_PROCESS_AUDIT.json');counts={k:sum(w['exit']['counts'][k] for w in audit['workers']) for k in audit['workers'][0]['exit']['counts']}
    assert counts['generations']==len(review) and counts['updates']==counts['backwards']==counts['operator_backwards']==0
    outputs=[read('remote/'+str(i%2)+'/outputs/BASE/'+r['query_id']+'.json') for i,r in enumerate(manifest['rows'])]
    assert audit['own_gpu_processes_absent'] and all(w['exit']['code']==0 and not w['matching_process_alive'] for w in audit['workers'])
    return dict(status=admission['status'],candidate_caps=dict(train=32,eval=128),initial_candidates=dict(train=initial['train_count'],eval=initial['eval_count']),source_excluded_before_base=len(manifest['source_review_excluded']),qualified_candidates=dict(train=manifest['train_count'],eval=manifest['eval_count']),base_review=by_split,base_correct_independent_eval_articles=len(articles),required_eval_correct_articles=4,counts=counts,EOS=sum(v['EOS'] for v in outputs),at_cap=sum(v['at_cap'] for v in outputs),worker_process_hours=audit['worker_process_hours'],workers_exited=True,premodel_failure_attempts=2,premodel_failure_model_calls=0,six_arm_started=False,new_fisher_started=False,current_task_reference_started=False,balancedit_original_started=False,balancedit_equal_data_started=False,external_Judge=0,independent_clinical_verification=False,official_benchmark_score=False,old_results_preserved=True,downstream_implementation_completed=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--private',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();result=summarize(a.private);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
