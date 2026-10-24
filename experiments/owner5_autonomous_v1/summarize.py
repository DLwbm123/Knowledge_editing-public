"""Aggregate the completed two-phase fixed diagnostic after blinded adjudication."""
import json,sys
from pathlib import Path

def summarize(root):
    def read(p):return json.loads((root/p).read_text())
    verdicts=read('BLIND_VERDICTS.json');mapping=read('BLIND_MAPPING.json')
    assert len(verdicts)==len(mapping)==15 and len({v['id'] for v in verdicts})==15
    for phase,expected in (('reserve',9),('trajectory',6)):
        assert read(phase+'/EXIT.json')['code']==0
        assert read(phase+'/COMPLETE.json')['counts']['generations']==expected
    assert read('trajectory/PARITY.json')==dict(endpoint_tensors_exact=True,training_curves_exact=True,arms=2)
    by={}
    for v in verdicts:
        assert v['verdict'] in ('CORRECT','INCORRECT','UNCERTAIN','CONTRADICTORY')
        m=mapping[v['id']];by[(m['phase'],m['arm'],m['query'])]=v['verdict']
    panel={arm:[by['reserve',arm,'reserve_'+str(i)] for i in (1,2,3)] for arm in ('BASE','CONTROL','PROTECTED')}
    base=[i for i,v in enumerate(panel['BASE']) if v=='CORRECT']
    counters=[read(p+'/COMPLETE.json')['counts'] for p in ('reserve','trajectory')]
    totals={k:sum(c[k] for c in counters) for k in counters[0]}
    assert totals==dict(updates=320,backwards=984,operator_backwards=320,generations=15,forwards=3210)
    files=[p for phase in ('reserve','trajectory') for p in (root/phase/'outputs').glob('*/*.json')]
    outputs=[json.loads(p.read_text()) for p in files];assert len(outputs)==15
    assert all(v['EOS'] and not v['at_cap'] for v in outputs)
    tokens=lambda phase,arm,query:read(phase+'/outputs/'+arm+'/'+query+'.json')['raw_token_ids']
    chain=read('CHAIN_COMPLETE.json');assert chain['status']=='COMPLETE'
    return dict(status='NO_INCREMENTAL_SEMANTIC_PROTECTION_EVIDENCE',reserve=dict(total_per_arm=3,correct={a:v.count('CORRECT') for a,v in panel.items()},base_correct_denominator=len(base),new_damage={a:sum(panel[a][i]!='CORRECT' for i in base) if base else None for a in ('CONTROL','PROTECTED')},damage_rate_defined=bool(base),all_three_arm_token_identical=sum(tokens('reserve','BASE',f'reserve_{i}')==tokens('reserve','CONTROL',f'reserve_{i}')==tokens('reserve','PROTECTED',f'reserve_{i}') for i in (1,2,3))),trajectory=dict(nodes=[40,80,120],correct={a:sum(by['trajectory',a,'node_'+str(n)]=='CORRECT' for n in (40,80,120)) for a in ('CONTROL','PROTECTED')},paired_token_identical=sum(tokens('trajectory','CONTROL','node_'+str(n))==tokens('trajectory','PROTECTED','node_'+str(n)) for n in (40,80,120)),endpoint_tensors_exact=True,training_curves_exact=True),counts=totals,eos_terminated=15,at_cap=0,model_worker_process_seconds=sum(v['seconds'] for v in chain['phases']),scope='one owner; source-grounded development diagnosis; no independent clinical adjudication',external_judge_calls=0)

if __name__=='__main__':
    root=Path(sys.argv[1]);result=summarize(root)
    if len(sys.argv)>2:Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
