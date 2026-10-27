"""Freeze the 150-expert development evaluation plan and preserve its library."""
import json
import os
from pathlib import Path
import time
import torch
import med133_evaluate

ROOT=Path(os.environ['EXPERIMENT_ROOT']);OLD=Path(os.environ['RAW_MED_ROOT'])

def write(name,value):
    p=ROOT/name;assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')

def main():
    clock=json.loads((ROOT/'CLOCK.json').read_text());assert time.time()<clock['hard_end']-620
    assert json.loads((ROOT/'TRAIN_COMPLETE.json').read_text())['actual_new_scientific_updates']==1280
    for n in ('TRAIN_WORKER_EXIT.json','TRAIN_GUARD_EXIT.json','BASE_WORKER_EXIT.json','BASE_GUARD_EXIT.json'):assert json.loads((ROOT/n).read_text())['code']==0
    f=json.loads((OLD/'INPUTS_FROZEN.json').read_text());extra=json.loads((ROOT/'ADDITIONAL_INPUTS.json').read_text());p=torch.load(OLD/'retained/MED146.pt',map_location='cpu',weights_only=True)
    assert len(p['tasks'])==len(p['R0_entries'])==len(p['states'])==146
    assert all(med133_evaluate.state_id(state)==p['state_hashes'][key] for key,state in p['states'].items())
    for e in extra['tasks']:
        t=e['task'];owner=t['order'];assert owner==len(p['tasks'])+1 and t['edit_id'] not in p['states']
        state=torch.load(ROOT/'candidates'/f'MED_{owner}.pt',map_location='cpu',weights_only=True)
        router=torch.load(ROOT/'routers'/f'{owner}.pt',map_location='cpu',weights_only=True)
        assert len(router['entries'])==1 and router['entries'][0]['logical_edit_id']==t['edit_id']
        p['tasks'].append(t);p['R0_entries'].append(router['entries'][0]);p['states'][t['edit_id']]=state;p['state_hashes'][t['edit_id']]=med133_evaluate.state_id(state)
        p['source_rows'][str(owner)]=e['source_rows'];p['calibration'][str(owner)]=json.loads((ROOT/'calibration'/f'{owner}.json').read_text())
    p['extension']=dict(kind='DEVELOPMENT_146_PLUS_4',independent_confirmation=False,source='MED150 frozen four',t1g_semantic_status='UNKNOWN')
    assert len(p['tasks'])==len(p['R0_entries'])==len(p['states'])==150
    dest=ROOT/'retained/MED150.pt';dest.parent.mkdir(exist_ok=True);assert not dest.exists();torch.save(p,dest.with_suffix('.tmp'));dest.with_suffix('.tmp').replace(dest)
    check=torch.load(dest,map_location='cpu',weights_only=True);assert check['state_hashes']==p['state_hashes'] and len(check['states'])==150
    outputs=[json.loads(x.read_text()) for x in sorted((ROOT/'base_outputs').glob('*.json'))];assert len(outputs)==55 and len({x['query']['query_id'] for x in outputs})==55
    new=[dict(method='MED',mode='single',prefix=1,owner=x['owner'],query=x['query']) for x in outputs]
    bank=[dict(method='MED',mode='bank',prefix=150,owner=0,query=x['query']) for x in f['plan'] if x['mode']=='bank' and x['prefix']==146]
    assert len(bank)==2093 and len({x['query']['query_id'] for x in bank})==2093
    assert not ({x['query']['query_id'] for x in new}&{x['query']['query_id'] for x in bank})
    bank.extend(dict(method='MED',mode='bank',prefix=150,owner=0,query=x['query']) for x in outputs)
    insert=[dict(method='MED',mode='insertion',prefix=e['task']['order'],owner=e['task']['order'],query=next(x['query'] for x in outputs if x['owner']==e['task']['order'] and x['query']['audit_role']=='T0')) for e in extra['tasks']]
    plan=dict(plan=new+insert+bank,cohort='development; no independent confirmation',counts=dict(single=55,insertion=4,bank150=2148),t1g_semantics='UNKNOWN')
    assert len(plan['plan'])==2207
    write('EVAL_PLAN_FROZEN.json',plan)
    write('LIBRARY150_FROZEN.json',dict(time=time.time(),experts=150,old146_state_ids_preserved=True,save_load=True,bytes=dest.stat().st_size,new_four=4,independent_backup_pending=True))
    print(json.dumps(dict(status='PASS',experts=150,plan=2207,library_bytes=dest.stat().st_size)))

if __name__=='__main__':main()
