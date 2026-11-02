"""Check 2148 paired input bindings without reading references or answers."""
import collections
import json
import os
from pathlib import Path
import signal
import sys

root=Path(os.environ['EXPERIMENT_ROOT'])
balanced=Path(os.environ['BALANCEDIT_ROOT'])
full_d=Path(os.environ['FULL_D_ROOT'])
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('pairing audit deadline')))
signal.alarm(120)
sys.path.insert(0,str(full_d/'tools'))
import prior_binding
import stream_extract as stream

original=json.loads((root/'BANK_INPUTS_PRIVATE.json').read_text())
expected={x['query']['query_id']:x for x in original['records']}
if len(expected)!=2148:raise ValueError('original bank identities missing')
counts=collections.Counter();seen=set()
allowed={'mode':True,'prefix':True,'binding':{'input':{'query_id':True,'question':True,'image_sha256':True},
         'judge_input':{'question':True,'image_sha256':True,'prompt_ids':True,'attention_mask':True,'generation':True}}}
for file in (balanced/'outputs').glob('*.json'):
    with file.open('rb') as f:row=prior_binding.select(f,stream.nonspace(f),allowed)
    if row.get('mode')!='bank' or row.get('prefix')!=150:continue
    query=row['binding']['input'];binding=row['binding']['judge_input'];qid=query['query_id']
    if qid in seen or qid not in expected:raise ValueError('comparison consumer identity differs')
    seen.add(qid);prior=expected[qid];counts['events']+=1
    for key in ('question','image_sha256'):
        counts['input_mismatch_'+key]+=query.get(key)!=prior['query'].get(key)
    for key in ('question','image_sha256','prompt_ids','attention_mask','generation'):
        counts['binding_mismatch_'+key]+=binding.get(key)!=prior['judge_input'].get(key)
if len(seen)!=2148:raise ValueError('incomplete comparison event coverage')
print(json.dumps({'status':'PASS' if all(v==0 for k,v in counts.items() if k!='events') else 'MISMATCH',
                  'counts':dict(counts),'reference_or_answer_decoded':0}))
