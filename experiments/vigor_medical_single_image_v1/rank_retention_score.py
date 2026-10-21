"""Once-only blind retention scoring, reusing exact Base answers only."""
import json
import os
import random
import time
import uuid
from pathlib import Path


def prepare(rows):
    records=[];mapping={};keys={};scores={};cached=[]
    for row in rows:
        alias=row['alias']
        if row['status']!='COMPLETE':continue
        if row['base_correct'] is not True:raise ValueError('Unqualified Base')
        if row['raw_base_answer']==row['base_answer']:
            scores[alias]=True;cached.append(alias);continue
        key=(row['question'],row['gold_answer'],row['raw_base_answer'])
        if key not in keys:
            opaque=uuid.uuid4().hex;keys[key]=opaque;mapping[opaque]=[]
            records.append(dict(opaque_query_id=opaque,question=key[0],gold_answer=key[1],raw_base_answer=key[2]))
        mapping[keys[key]].append(alias)
    random.SystemRandom().shuffle(records)
    return records,mapping,scores,cached


def main():
    root=Path(os.environ['RUN_ROOT'])
    execution=json.loads((root/'private/EXECUTION_BINDING.json').read_text())
    terminal=json.loads((root/'public/NUMERIC_TERMINAL.json').read_text())
    closure=json.loads((root/'closure.json').read_text());guard=execution['manifest']['guard']
    if not (closure['verified_exit'] and closure['gpu_absent'] and closure['pid']==guard['target_pid'] and closure['birth']==guard['target_birth'] and terminal['status']=='COMPLETE' and terminal['restored']):
        raise PermissionError('Need current restored terminal and process exit')
    c=execution['config'];aliases=c['panel_aliases']
    if aliases!=[f'L{i:03}' for i in range(1,7)] or len(terminal['retention'])!=6:raise ValueError('Frozen six query denominator required')
    if time.time()+600>=execution['manifest']['start_epoch']+c['whole_seconds']:raise TimeoutError('Original overall clock insufficient')
    with (root/'private/SCORE_RESERVED.json').open('x') as f:json.dump(dict(epoch=time.time(),attempts_per_payload=1,maximum_unique=6),f)
    rows=[json.loads((root/f'private/{a}_GENERATION.json').read_text()) for a in aliases]
    records,mapping,scores,cached=prepare(rows)
    packet=root/'private/SCORE_PACKETS.json';packet.write_text(json.dumps(dict(records=records)))
    (root/'private/SCORE_MAPPING.json').write_text(json.dumps(dict(mapping=mapping,base_cached=cached)))
    summary=dict(status='CACHE_ONLY',payloads=0,elapsed_seconds=0,decisions=[],tool_event_types=[])
    if records:
        config=dict(scorer_source=os.environ['SCORER_SOURCE'],private_root=str(root/'private/scoring'),packets=str(packet),payload_max=len(records),attempts=1,cli=os.environ['SCORER_CLI'])
        config_file=root/'private/SCORE_CONFIG.json';config_file.write_text(json.dumps(config));os.environ['CONFIG_FILE']=str(config_file)
        from .factF3_judge import main as judge
        judge()
        summary=json.loads((root/'private/scoring/SUMMARY.json').read_text())
        seen=set()
        for decision in summary['decisions']:
            key=decision['opaque_query_id']
            if key not in mapping or key in seen or type(decision['is_correct']) is not bool:raise ValueError('Invalid/duplicate scoring decision')
            seen.add(key)
            for alias in mapping[key]:scores[alias]=decision['is_correct']
    grades={a:scores.get(a) for a in aliases}
    damaged=sum(x is False for x in grades.values());missing=sum(x is None for x in grades.values())
    result=dict(status=summary['status'],scores=grades,Base_correct=6,post_correct=sum(x is True for x in grades.values()),damaged=damaged,missing=missing,damage_interval=[damaged,damaged+missing],retention_qualified=damaged==missing==0,base_exact_cached=cached,new_unique_payloads=len(records),attempts_per_payload=1,retries=0,judge_seconds=summary['elapsed_seconds'],tool_event_types=summary['tool_event_types'],isolation_checks=summary.get('isolation_checks',{}),near_boundary_support=0,scope='fixed development unrelated panel; no independent generalization claim')
    with (root/'public/SCORES.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
