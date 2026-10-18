"""Validate complete fixed-panel outputs and aggregate prewritten blind semantic verdicts."""
import json,sys
from pathlib import Path
from collections import Counter

def summarize(root):
    def read(path):return json.loads((root/path).read_text())
    packet={v['id']:v for v in read('BLIND_UNIQUE_PACKET.json')}
    verdicts={v['id']:v['verdict'] for v in read('BLIND_UNIQUE_VERDICTS.json')}
    mapping=read('BLIND_MAPPING.json');assert len(mapping)==288 and set(packet)==set(verdicts)
    assert all(v in ('CORRECT','INCORRECT','UNCERTAIN','CONTRADICTORY') for v in verdicts.values())
    rows={};counts=Counter();eos=cap=0;partitions=[]
    for part in ('0','1'):
        complete=read(part+'/COMPLETE.json');assert complete['status']=='MODEL_WORK_COMPLETE_HOLDING_RESIDENCY'
        assert complete['counts']['generations']==144
        counts.update(complete['counts']);partitions.append({v['query_id'] for v in read(part+'/PANEL.json')})
    assert len(partitions[0])==len(partitions[1])==48 and not partitions[0]&partitions[1]
    for item,m in mapping.items():
        output=read(m['partition']+'/outputs/'+m['arm']+'/'+m['query']+'.json');p=packet[m['review_id']]
        assert (p['question'],p['reference'],p['answer'])==(output['binding']['question'],output['binding']['reference'],output['raw_answer'])
        key=(m['arm'],m['query']);assert key not in rows
        rows[key]=dict(output,verdict=verdicts[m['review_id']]);eos+=int(output['EOS']);cap+=int(output['at_cap'])
    arms=('BASE','CONTROL','PROTECTED');queries=sorted(partitions[0]|partitions[1]);assert len(rows)==288
    for q in queries:
        b=rows['BASE',q]['binding']
        for arm in arms[1:]:
            assert all(rows[arm,q]['binding'][k]==b[k] for k in ('question','reference','image_sha256','prompt_ids','runtime','generation'))
    base_correct=[q for q in queries if rows['BASE',q]['verdict']=='CORRECT']
    result=dict(status='NO_INCREMENTAL_SEMANTIC_PROTECTION_BENEFIT_OBSERVED_ON_FIXED96',queries=96,outputs=288,unique_blind_semantic_records=len(packet),semantic={arm:{v:sum(rows[arm,q]['verdict']==v for q in queries) for v in ('CORRECT','INCORRECT','UNCERTAIN','CONTRADICTORY')} for arm in arms},base_correct_denominator=len(base_correct),new_damage={arm:sum(rows[arm,q]['verdict']!='CORRECT' for q in base_correct) for arm in arms[1:]},repairs={arm:sum(rows['BASE',q]['verdict']!='CORRECT' and rows[arm,q]['verdict']=='CORRECT' for q in queries) for arm in arms[1:]},control_protected_same_verdict=sum(rows['CONTROL',q]['verdict']==rows['PROTECTED',q]['verdict'] for q in queries),control_protected_token_identical=sum(rows['CONTROL',q]['raw_token_ids']==rows['PROTECTED',q]['raw_token_ids'] for q in queries),counts=dict(counts),eos_terminated=eos,at_cap=cap,external_judge_calls=0,model_work_seconds=sum(read(p+'/COMPLETE.json')['model_work_seconds'] for p in ('0','1')),processes_status='RESIDENT_UNTIL_AUTHORIZED_DEADLINE',natural_routing=False,training_changes=False,independent_confirmation=False)
    assert counts['updates']==counts['backwards']==counts['operator_backwards']==0
    assert counts['generations']==eos==288 and cap==0
    return result

if __name__=='__main__':
    result=summarize(Path(sys.argv[1]));Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
