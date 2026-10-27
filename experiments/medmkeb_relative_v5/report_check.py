"""Synthetic full-denominator check of completion aggregation, no model calls."""
import json,tempfile
from pathlib import Path
from summarize import summarize,ARMS

def run():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        def write(name,value):
            path=p/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        evals=[dict(query_id='V'+str(i).zfill(2)) for i in range(1,28)]
        edits=[dict(query_id=q) for q in ('C02_native','C02_cross','C03_native','C03_cross')]
        tasks=[dict(key='E01',source=edits[0],rephrase=dict(query_id='E01_rephrase')),dict(key='E02',source=edits[2],rephrase=dict(query_id='E02_rephrase'))]
        write('CANDIDATES_FROZEN.json',dict(candidates=[dict(split='eval',row=r) for r in evals],rows=evals))
        write('DATA_ADMISSION.json',dict(tasks=tasks));write('EVAL_ROWS.json',evals+edits)
        mapping=[];packet=[];verdicts=[]
        def append(arm,task,q,label):
            rid='R'+str(len(packet)+1);packet.append(dict(id=rid));verdicts.append(dict(id=rid,label=label));mapping.append(dict(arm=arm,task=task,query_id=q,id=rid))
        for row in evals+edits+[t['rephrase'] for t in tasks]:append('BASE',None,row['query_id'],'CORRECT')
        for arm in ARMS:
            for t in tasks:
                for row in evals+edits+[t['rephrase']]:append(arm,t['key'],row['query_id'],'INCORRECT' if arm=='ON_NONE' and row['query_id']=='V01' else 'CORRECT')
        write('BLIND_PACKET.json',packet);write('BLIND_VERDICTS.json',dict(rows=verdicts));write('BLIND_MAPPING.json',mapping)
        for i in (0,1):
            counts=dict(updates=1060,generations=256,backwards=4000,operator_backwards=0 if i==0 else 962,forwards=10000)
            if i==0:counts['algebraic_TT_backwards']=1
            write(f'remote/{i}/EXIT.json',dict(code=0,counts=counts));write(f'remote/{i}/STUDY_COMPLETE.json',dict(status='COMPLETE'))
        result=summarize(p);assert result['counts']['generations']==512 and result['counts']['algebraic_TT_backwards']==1
        assert result['paired_comparisons']['ON_SEM_vs_NONE']['development_protection_criterion_met']
        assert not result['paired_comparisons']['OFF_SEM_vs_NONE']['development_protection_criterion_met']
        assert result['paired_comparisons']['ON_SEM_vs_NONE']['recovered_checks']==2
        verdicts.pop();write('BLIND_VERDICTS.json',dict(rows=verdicts))
        try:summarize(p)
        except AssertionError:pass
        else:raise AssertionError('Missing consumer verdict admitted')
    return dict(status='PASS',all512_method_consumers=True,counter_union=True,paired_damage=True,zero_control_damage_not_gain=True,missing_verdict_rejected=True,new_model_calls=0)
if __name__=='__main__':print(json.dumps(run()))
