"""Aggregate every frozen eight-condition consumer after method-hidden semantic review."""
import json,collections
from pathlib import Path
ARMS=['OFF_NONE','OFF_CE','OFF_SEM','ON_NONE','ON_CE','ON_SEM','BALANCEDIT','BALANCEDIT_CE']
LABELS={'CORRECT','INCORRECT','UNCERTAIN','CONTRADICTORY'}
def summarize(private):
    read=lambda p:json.loads(Path(p).read_text())
    packet=read(private/'BLIND_PACKET.json');verdicts=read(private/'BLIND_VERDICTS.json')['rows'];labels={v['id']:v['label'] for v in verdicts}
    assert len(labels)==len(verdicts)==len(packet) and set(labels)=={v['id'] for v in packet} and set(labels.values())<=LABELS
    mapping=read(private/'BLIND_MAPPING.json');admission=read(private/'DATA_ADMISSION.json');manifest=read(private/'CANDIDATES_FROZEN.json');tasks=admission['tasks'];evaluation=[c['row']['query_id'] for c in manifest['candidates'] if c['split']=='eval']
    base={v['query_id']:labels[v['id']] for v in mapping if v['arm']=='BASE'}
    original_correct=[q for q in evaluation if base[q]=='CORRECT'];assert len(original_correct)>=4
    results={}
    for arm in ARMS:
        entries=[v for v in mapping if v['arm']==arm];by_task={}
        for task in tasks:
            values={v['query_id']:labels[v['id']] for v in entries if v['task']==task['key']}
            assert set(values)=={v['query_id'] for v in read(private/'EVAL_ROWS.json')}|{task['rephrase']['query_id']}
            source=task['source']['query_id'];cross=source.removesuffix('_native')+'_cross'
            by_task[task['key']]=dict(source_correct=values[source]=='CORRECT',own_cross_correct=values[cross]=='CORRECT',own_rephrase_correct=values[task['rephrase']['query_id']]=='CORRECT',heldout_four_category=dict(collections.Counter(values[q] for q in evaluation)),base_correct_denominator=len(original_correct),new_damage=sum(values[q]!='CORRECT' for q in original_correct),damage_to_wrong_or_contradictory=sum(values[q] in ('INCORRECT','CONTRADICTORY') for q in original_correct),damage_to_uncertain=sum(values[q]=='UNCERTAIN' for q in original_correct),all_originally_correct_retained=all(values[q]=='CORRECT' for q in original_correct))
        results[arm]=dict(tasks=by_task,source_correct=sum(v['source_correct'] for v in by_task.values()),cross_correct=sum(v['own_cross_correct'] for v in by_task.values()),rephrase_correct=sum(v['own_rephrase_correct'] for v in by_task.values()),base_correct_denominator=sum(v['base_correct_denominator'] for v in by_task.values()),new_damage=sum(v['new_damage'] for v in by_task.values()))
    receipts=[read(private/'remote'/str(i)/'EXIT.json') for i in (0,1)];assert all(v['code']==0 for v in receipts)
    counts={k:sum(v['counts'].get(k,0) for v in receipts) for k in set().union(*(set(v['counts']) for v in receipts))};assert counts['updates']==2120 and counts['generations']==sum(len([v for v in mapping if v['arm']==a]) for a in ARMS)==512
    assert all(read(private/'remote'/str(i)/'STUDY_COMPLETE.json')['status']=='COMPLETE' for i in (0,1))
    assert counts['backwards']<=20000 and counts['operator_backwards']<=1100 and counts['forwards']<=800126
    comparisons={}
    for prefix in ('OFF','ON'):
        sem_entries={(v['task'],v['query_id']):labels[v['id']] for v in mapping if v['arm']==prefix+'_SEM'}
        for other in ('NONE','CE'):
            control={(v['task'],v['query_id']):labels[v['id']] for v in mapping if v['arm']==prefix+'_'+other}
            assert control.keys()==sem_entries.keys()
            damage_keys=[k for k in control if k[1] in original_correct]
            recovered=sum(control[k]!='CORRECT' and sem_entries[k]=='CORRECT' for k in damage_keys)
            extra=sum(control[k]=='CORRECT' and sem_entries[k]!='CORRECT' for k in damage_keys)
            same_source=results[prefix+'_SEM']['source_correct']>=results[prefix+'_'+other]['source_correct']
            same_transfer=all(results[prefix+'_SEM'][k]>=results[prefix+'_'+other][k] for k in ('cross_correct','rephrase_correct'))
            identifiable=results[prefix+'_'+other]['new_damage']>0
            comparisons[prefix+'_SEM_vs_'+other]=dict(control_has_damage=identifiable,recovered_checks=recovered,additional_damage_checks=extra,source_not_worse=same_source,transfer_not_worse=same_transfer,development_protection_criterion_met=identifiable and recovered>extra and extra==0 and same_source and same_transfer)
    return dict(status='COMPLETE',paired_comparisons=comparisons,base_correct_eval_questions=len(original_correct),heldout_questions=len(evaluation),panels=results,counts=counts,unique_hidden_reviews=len(packet),formal_tasks=2,arms=ARMS,independent_clinical_confirmation=False,official_counterfactual_benchmark=False,extra_support_shared=True,method_win_not_automatically_inferred=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(summarize(a.private),indent=2)+'\n')
