"""Prefix-only ownership and exactly matched replay sampling, without student outputs."""
from collections import Counter
from dataclasses import asdict
from itertools import combinations
import random
import torch
from freshstart.runtime import ROOT, read, write
from freshstart.pipeline import input_key
from scripts.medtrace.stage17_prepare import digest
from m3bench_repro.editors.routing import MemoryRouter


def signature(rows):
    return (len(rows),len({r['row']['source_group'] for r in rows}),
            len({input_key(r['row']) for r in rows}),tuple(sorted(Counter(r['token_bin'] for r in rows).items())))


def matched_uniform(aligned, pool, rng):
    if not aligned:
        return []
    target = signature(aligned)
    chosen, eligible = None, 0
    # Exact uniform reservoir over feasible subsets; never substitute an approximate match.
    for n, subset in enumerate(combinations(pool,len(aligned))):
        if n >= 200000:
            raise RuntimeError('Exact matched replay enumeration capacity exceeded')
        if signature(subset)==target:
            eligible += 1
            if rng.randrange(eligible)==0:
                chosen = list(subset)
    if chosen is None:
        raise ValueError('No exactly matched random support')
    return chosen


def compile_plan(runtime, tasks, seed):
    bases = read(ROOT/'private/preparation/BASE_OUTPUTS.json')
    router_state = torch.load(ROOT/'private/preparation/router.pt',weights_only=True)
    keys = torch.load(ROOT/'private/preparation/query_keys.pt',weights_only=True)
    supports, references = {}, []
    for prefix, task in enumerate(tasks,1):
        for row in [task['native']]+task['H_fit']+task['U_fit']:
            base = bases[digest([input_key(row),row['reference']])]
            score = read(ROOT/'private/judge/scores'/f'{base["judge_key"]}.json')
            if score['key']!=base['judge_key'] or type(score['is_correct']) is not bool:
                raise ValueError('Unbound Base verdict')
            expected=dict(opaque_query_id=base['judge_key'],question=row['question'],gold_answer=row['reference'],raw_base_answer=base['output']['raw_answer'])
            if score['payload_binding']!=digest(expected):raise ValueError('Base verdict payload mismatch')
            references.append(dict(input=input_key(row),arrival=prefix,correct=score['is_correct']))
        for role in ['H','U']:
            for row in task[role+'_fit']:
                sid = digest([role,row])
                if sid not in supports:
                    tokens=len(runtime.adapter.tokenizer.encode(row['reference'],add_special_tokens=False))+1
                    supports[sid]=dict(id=sid,role=role,row=row,arrival=prefix,
                        token_bin=0 if tokens<=4 else 1 if tokens<=8 else 2 if tokens<=16 else 3)
    pending, previous, slots, states = [], {}, [], {}
    router=MemoryRouter('euclidean'); rng=random.Random(seed)
    entries={e['logical_edit_id']:e for e in router_state['entries']}
    for prefix, task in enumerate(tasks,1):
        entry=entries[task['canonical_edit_id']]
        router.add(entry['logical_edit_id'],entry['key'],entry['radius'])
        active=[]; state={}
        for sid,support in supports.items():
            if support['arrival']>prefix:continue
            masked=support['role']=='U' and any(r['input']==input_key(support['row']) and r['arrival']<=prefix and not r['correct'] for r in references)
            route=asdict(router.route(keys[input_key(support['row'])]))
            owner=route['logical_edit_id'] if route['activated'] else None
            state[sid]=dict(state='masked' if masked else 'active',owner=owner,route=route)
            if not masked:
                active.append(support)
                if owner and previous.get(sid)!=owner and owner not in pending:pending.append(owner)
            previous[sid]=owner
        current=entry['logical_edit_id']
        selected=[current]+[p for p in pending if p!=current][:1]
        for ordinal,expert in enumerate(selected,1):
            aligned={role:[r for r in active if r['role']==role and state[r['id']]['owner']==expert] for role in ['H','U']}
            random_rows={role:matched_uniform(aligned[role],[r for r in active if r['role']==role],rng) for role in ['H','U']}
            slot=dict(prefix=prefix,expert=expert,ordinal=ordinal,
                FA={k:[r['id'] for r in v] for k,v in aligned.items()},FR={k:[r['id'] for r in v] for k,v in random_rows.items()},
                steps=20 if any(aligned.values()) else 0)
            slots.append(slot)
            if expert in pending:pending.remove(expert)
        states[str(prefix)]=state
    plan=dict(seed=seed,N=len(tasks),supports=supports,roles=states,slots=slots,
        reference_policy='available legal references override fresh Base U teacher; semantic source Judge',
        matching='uniform exact feasible subsets; same role counts/source counts/distinct inputs/token bins',
        student_outputs_used=False)
    plan['plan_hash']=digest(plan)
    for slot in slots:slot['slot_key']=f"{plan['plan_hash']}/{slot['prefix']}/{slot['expert']}/{slot['ordinal']}"
    if len({s['slot_key'] for s in slots})!=len(slots):raise ValueError('Duplicate maintenance slot')
    return plan


def self_check():
    rows=[dict(row=dict(source_group=str(i),image_sha256=str(i),question=str(i)),token_bin=0) for i in range(4)]
    assert signature(matched_uniform(rows[:2],rows,random.Random(1)))==signature(rows[:2])
    assert matched_uniform([],rows,random.Random(1))==[]


if __name__=='__main__':self_check()
