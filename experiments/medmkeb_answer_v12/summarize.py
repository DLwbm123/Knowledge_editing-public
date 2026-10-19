"""Complete source/transfer and full Base-correct denominator after hidden review."""
import json,collections
from pathlib import Path
LABELS={'CORRECT','INCORRECT','UNCERTAIN','CONTRADICTORY'}

def summarize(private):
    read=lambda p:json.loads(p.read_text());packet=read(private/'BLIND_PACKET.json');verdict=read(private/'BLIND_VERDICTS.json')['rows'];labels={v['id']:v['label'] for v in verdict};assert len(labels)==len(verdict)==len(packet) and set(labels)=={x['id'] for x in packet} and set(labels.values())<=LABELS
    mapping=read(private/'BLIND_MAPPING.json');assert len(mapping)==64;base=read(private/'BASE_LABELS_FROZEN.json');evals=[c['row']['query_id'] for c in read(private/'CANDIDATES_FROZEN.json')['candidates'] if c['split']=='eval'];assert len(evals)==27;correct=[q for q in evals if base[q]=='CORRECT'];assert len(correct)==11
    results=[];exits=[]
    for i,t in enumerate(read(private/'DATA_ADMISSION.json')['tasks']):
        root=private/'remote'/str(i);e=read(root/'EXIT.json');assert e['code']==0 and read(root/'STUDY_COMPLETE.json')['answers']==32;v={x['query_id']:labels[x['id']] for x in mapping if x['task']==t['key']};assert len(v)==32;source=t['source']['query_id'];cross=source.removesuffix('_native')+'_cross'
        results.append(dict(edit=i+1,source_correct=v[source]=='CORRECT',cross_correct=v[cross]=='CORRECT',rephrase_correct=v[t['rephrase']['query_id']]=='CORRECT',heldout_four_category=dict(collections.Counter(v[q] for q in evals)),base_correct_denominator=len(correct),damage=sum(v[q]!='CORRECT' for q in correct),damage_to_uncertain=sum(v[q]=='UNCERTAIN' for q in correct),damage_to_wrong_or_contradiction=sum(v[q] in {'INCORRECT','CONTRADICTORY'} for q in correct)));exits.append(e)
    counts={k:sum(x['counts'].get(k,0) for x in exits) for k in set().union(*(x['counts'] for x in exits))};assert counts['generations']==64 and counts['forwards']<=66000 and counts['updates']==counts['backwards']==0
    return dict(status='COMPLETE_ANSWER_CONSUMERS',tasks=results,source_correct=sum(x['source_correct'] for x in results),cross_correct=sum(x['cross_correct'] for x in results),rephrase_correct=sum(x['rephrase_correct'] for x in results),source_capability_gate=all(x['source_correct'] for x in results),base_correct_question_count=11,paired_checks=22,new_damage=sum(x['damage'] for x in results),EOS=sum(x['EOS'] for x in mapping),at_cap=sum(x['at_cap'] for x in mapping),counts=counts,gpu_process_hours=sum(x['seconds'] for x in exits)/3600,independent_confirmation=False,matched_31step_comparator_available=False,protection_benefit_identified=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(summarize(a.private),indent=2)+'\n')
