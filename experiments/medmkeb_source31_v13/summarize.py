"""Complete source/transfer and full Base-correct denominator after hidden review."""
import json,collections
from pathlib import Path
LABELS={'CORRECT','INCORRECT','UNCERTAIN','CONTRADICTORY'}

def summarize(private):
    read=lambda p:json.loads(p.read_text());packet=read(private/'BLIND_PACKET.json');verdict=read(private/'BLIND_VERDICTS.json')['rows'];labels={v['id']:v['label'] for v in verdict};assert len(labels)==len(verdict)==len(packet) and set(labels)=={x['id'] for x in packet} and set(labels.values())<=LABELS
    mapping=read(private/'BLIND_MAPPING.json');assert len(mapping)==64;base=read(private/'BASE_LABELS_FROZEN.json');evals=[c['row']['query_id'] for c in read(private/'CANDIDATES_FROZEN.json')['candidates'] if c['split']=='eval'];assert len(evals)==27;correct=[q for q in evals if base[q]=='CORRECT'];assert len(correct)==11
    results=[];exits=[]
    for i,t in enumerate(read(private/'DATA_ADMISSION.json')['tasks']):
        root=private/'remote'/str(i);assert read(root/'TRAIN_COMPLETE.json')['updates']==31 and read(root/'TRAIN_COMPLETE.json')['zero_constraint_path_exact'];e=read(root/'EXIT.json');assert e['code']==0 and read(root/'STUDY_COMPLETE.json')['answers']==32;v={x['query_id']:labels[x['id']] for x in mapping if x['task']==t['key']};assert len(v)==32;source=t['source']['query_id'];cross=source.removesuffix('_native')+'_cross'
        results.append(dict(edit=i+1,source_correct=v[source]=='CORRECT',cross_correct=v[cross]=='CORRECT',rephrase_correct=v[t['rephrase']['query_id']]=='CORRECT',heldout_four_category=dict(collections.Counter(v[q] for q in evals)),base_correct_denominator=len(correct),damage=sum(v[q]!='CORRECT' for q in correct),damage_to_uncertain=sum(v[q]=='UNCERTAIN' for q in correct),damage_to_wrong_or_contradiction=sum(v[q] in {'INCORRECT','CONTRADICTORY'} for q in correct)));exits.append(e)
    failed=[]
    counts={k:sum(x['counts'].get(k,0) for x in exits) for k in set().union(*(x['counts'] for x in exits))};assert counts['generations']==64 and counts['forwards']<=66200 and counts['updates']==62 and counts['backwards']==60 and counts['reused_gradient_sequences']==2
    previous=private.parent/'medmkeb_answer_v12_20261019';sem_labels={x['id']:x['label'] for x in read(previous/'BLIND_VERDICTS.json')['rows']};sem={(x['task'],x['query_id']):sem_labels[x['id']] for x in read(previous/'BLIND_MAPPING.json')};control={(x['task'],x['query_id']):labels[x['id']] for x in mapping};assert sem.keys()==control.keys()
    keys=[k for k in control if k[1] in correct];recovered=sum(control[k]!='CORRECT' and sem[k]=='CORRECT' for k in keys);extra=sum(control[k]=='CORRECT' and sem[k]!='CORRECT' for k in keys);sem_summary=read(previous/'RESULTS_FROZEN.json')
    return dict(status='COMPLETE_SOURCE31_PAIRED_CONSUMERS',SOURCE_tasks=results,SEM_tasks=sem_summary['tasks'],SOURCE_source_correct=sum(x['source_correct'] for x in results),SEM_source_correct=sem_summary['source_correct'],SOURCE_cross_correct=sum(x['cross_correct'] for x in results),SOURCE_rephrase_correct=sum(x['rephrase_correct'] for x in results),SOURCE_damage=sum(x['damage'] for x in results),SEM_damage=sem_summary['new_damage'],base_correct_question_count=11,paired_checks=22,recovered_by_SEM=recovered,extra_damage_by_SEM=extra,control_has_damage=any(x['damage'] for x in results),both_source_gates_failed=not all(x['source_correct'] for x in results) and not sem_summary['source_capability_gate'],protection_benefit_identified=any(x['damage'] for x in results) and recovered>extra and extra==0 and sem_summary['source_correct']==2,EOS=sum(x['EOS'] for x in mapping),at_cap=sum(x['at_cap'] for x in mapping),counts=counts,gpu_process_hours=sum(x['seconds'] for x in exits)/3600,independent_confirmation=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(summarize(a.private),indent=2)+'\n')
