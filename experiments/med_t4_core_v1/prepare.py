"""Freeze the role-screened T4 Base packet before any new CORE judgments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    cfg = json.loads(Path(os.environ['T4_CONFIG']).read_text())
    worktree, root = Path(cfg['worktree']), Path(cfg['root'])
    sys.path.insert(0, cfg['common_source'])
    from med133_score_contract import PROTOCOL, PROMPT, digest
    contract = json.loads(Path(cfg['accepted_contract']).read_text())
    assert all(contract[k] == v for k, v in dict(protocol=PROTOCOL, prompt=PROMPT,
        model='gpt-6-astra', reasoning_effort='medium').items())
    role = json.loads(Path(cfg['role_filter']).read_text())
    events = [e for e in role['events'] if e['task'] in ('T4G', 'T4L') and not e['role_exclusion_reasons']]
    assert sum(e['task'] == 'T4G' for e in events) == 79
    assert sum(e['task'] == 'T4L' for e in events) == 58
    ids = sorted({q for e in events for q in [e['edit_query_id']] + e['all_probe_query_ids']})
    assert len(ids) == 211
    def write(name, value):
        with (root / name).open('x') as f:
            json.dump(value, f, ensure_ascii=False)
    write('EVENTS_FROZEN.private.json', dict(events=events, query_ids=ids,
        selection_uses_new_scores=False, source_role_filter=cfg['role_filter']))
    primitives = (worktree / 'experiments/med150_source_review_v1/stream_extract.py').read_text().split('def selfcheck():')[0]
    projection = (worktree / 'experiments/med_t4_core_v1/projection.py').read_text().replace('import stream_extract as s', '')
    command = 'import types\ns=types.ModuleType("stream_extract")\nexec('+repr(primitives)+',s.__dict__)\n'+projection+'\nIDS='+repr(ids)+'\nPATHS='+repr(cfg['remote_paths'])+'\n'+r'''
from pathlib import Path
legacy=Path(PATHS['legacy'])
index=read(legacy/'BINDINGS.json',{'*':{'query_id':True}})
keys=[k for k,v in index.items() if v['query_id'] in IDS]
fields={k:True for k in ('query_id','question','reference','image_sha256','image_path','prompt_ids','attention_mask','runtime','generation','writer','prefix')}
fields['output']={k:True for k in ('model_answer_raw','raw_generated_token_ids','prompt_token_ids','image_sha256','generated_token_count','error','runtime')}
bindings=read(legacy/'BINDINGS.json',{k:fields for k in keys})
assert len(bindings)==len(IDS) and {b['query_id'] for b in bindings.values()}==set(IDS)
parent=Path(PATHS['parent'])
with (parent/'private/EVAL_BINDINGS.json').open('rb') as f:
 assert s.nonspace(f)==b'{';assert s.nonspace(f)==b'"';s.string(f,True);assert s.nonspace(f)==b':'
 current=project(f,s.nonspace(f),{'runtime':True,'generation':True})
for b in bindings.values():
 o=b['output'];assert b['writer']=='Base' and b['prefix']==0 and o['error'] is None and o['runtime']=='official'
 assert b['runtime']==current['runtime'] and b['generation']==current['generation']
 assert b['image_sha256']==o['image_sha256'] and b['prompt_ids']==o['prompt_token_ids']
 assert 0<len(o['raw_generated_token_ids'])==o['generated_token_count']<=1024
 assert b['attention_mask']==[1]*len(b['prompt_ids'])
print(json.dumps({'bindings':bindings,'configuration':current,'runtime_and_generation_match':True}))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', cfg['ssh_host'], cfg['remote_python'], '-'],
        input=command, text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    material = json.loads(result.stdout)
    write('BASE_BINDINGS.private.json', material)
    accepted = json.loads(Path(cfg['accepted_decisions']).read_text())['decisions']
    payloads, consumers, votes, failures = {}, [], {}, set()
    fields = ('core_correct', 'full_supported', 'unsupported_expansion', 'extra_contradiction')
    def semantic(row):
        return digest(dict(protocol=PROTOCOL, model='gpt-6-astra', reasoning='medium',
            question=row['question'], gold_answer=row['gold_answer'], raw_base_answer=row['raw_base_answer']))
    for b in material['bindings'].values():
        answer = dict(raw_answer=b['output']['model_answer_raw'], raw_token_ids=b['output']['raw_generated_token_ids'])
        legacy = digest(dict(query_id=b['query_id'], question=b['question'], reference=b['reference'],
            image_sha256=b['image_sha256'], prompt_ids=b['prompt_ids'], answer=answer))
        row = dict(question=b['question'], gold_answer=b['reference'], raw_base_answer=answer['raw_answer'])
        key = semantic(row)
        payloads[key] = dict(row, opaque_query_id=key)
        consumers.append(dict(query_id=b['query_id'], key=key, legacy_key=legacy))
        if legacy in accepted:
            votes.setdefault(key, set()).add(tuple(accepted[legacy][f] for f in fields))
    # Reuse only exact judge-visible payloads under the unchanged scoring contract.
    for prior in cfg['prior_score_roots']:
        prior = Path(prior)
        packet_path, decisions_path = prior/'PACKET_PRIVATE.json', prior/'DECISIONS.json'
        if not packet_path.exists() or not decisions_path.exists():
            continue
        packet, decisions = json.loads(packet_path.read_text()), json.loads(decisions_path.read_text())
        for oldkey, raw in packet['payloads'].items():
            row = raw.get('record', raw)
            key = semantic(row)
            if key not in payloads:
                continue
            if oldkey in decisions['decisions']:
                votes.setdefault(key, set()).add(tuple(decisions['decisions'][oldkey][f] for f in fields))
            if oldkey in decisions.get('unknown', []):
                failures.add(key)
    known, unknown, pending = {}, [], []
    for key in sorted(payloads):
        values = votes.get(key, set())
        if len(values) > 1:
            unknown.append(key)
        elif values:
            known[key] = dict(opaque_query_id=key, **dict(zip(fields, next(iter(values)))))
        elif key in failures:
            unknown.append(key)
        else:
            pending.append(key)
    write('PACKET_PRIVATE.json', dict(payloads=payloads, consumers=consumers,
        known_valid=known, inherited_unknown=unknown, pending=pending, contract=contract))
    write('PREPARATION_COMPLETE.json', dict(time=time.time(), queries=len(ids), events=len(events),
        unique_payloads=len(payloads), known_valid=len(known), inherited_unknown=len(unknown),
        new_attempt_items=len(pending), runtime_and_generation_match=True, new_GPU_calls=0))
    print(json.dumps(json.loads((root/'PREPARATION_COMPLETE.json').read_text())))


if __name__ == '__main__':
    main()
