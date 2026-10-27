"""Bind existing independent-expert predictions to the frozen growing R0 bank.

Routing is computed from the unchanged Base before attaching a selected expert.
Consequently a recorded prediction from that same expert, query and runtime is
exactly reusable at every prefix selecting it. This is functional composition of
executed components, not a claim that a new full-bank GPU run occurred.
Stateful matrix editors are explicitly excluded from this reuse rule.
"""
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(os.environ['V131_ROOT'])
OUT = ROOT / 'M13P_160'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def consumer_id(row):
    return json.dumps([row[k] for k in ('method','mode','prefix','owner')] +
                     [row['query']['query_id']], separators=(',', ':'))


def name(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    PARENT = Path(os.environ['PARENT_RUN'])
    ROUTING = Path(os.environ['V131_ROUTING_PARENT'])
    assert not (OUT / 'REUSE_FROZEN.json').exists()
    assert read(PARENT / 'private/ROUTER_BINDINGS146.json') == read(ROUTING / 'private/ROUTER_BINDINGS146.json')
    packet = read(PARENT / 'private/M13_SOURCE_PACKET.json')
    tasks = packet['original_tasks']
    byid = {t['edit_id']: t['order'] for t in tasks}
    assert len(byid) == 146
    singles, bases, routes = {}, {}, {}
    for path in (PARENT / 'private/outputs/M13P_single_p1').rglob('*.json'):
        value = read(path)
        key = (value['owner'], value['binding']['input']['query_id'])
        assert key not in singles
        singles[key] = (path, value)
    for path in (PARENT / 'private/outputs/BASE_base_p0').rglob('*.json'):
        value = read(path)
        key = value['binding']['input']['query_id']
        assert key not in bases
        bases[key] = (path, value)
    folders = list((ROUTING / 'private/outputs').glob('M13_bank_p*'))
    folders += list((ROUTING / 'private/outputs').glob('M13_insertion_p*'))
    for folder in folders:
        for path in folder.rglob('*.json'):
            value = read(path)
            key = (value['mode'], value['prefix'], value['owner'], value['binding']['input']['query_id'])
            assert key not in routes and value['original_router'] is True
            routes[key] = (path, value)
    assert (len(singles),len(bases),len(routes)) == (2956,2093,4538)
    plan = [row for row in read(ROOT / 'EVAL_PLAN.json') if row['method'] == 'M13P_160']
    assert len(plan) == 7494
    requests = {}
    counts = collections.Counter()
    provenance = []
    for row in plan:
        qid = row['query']['query_id']
        if row['mode'] == 'single':
            source_path, source = singles[(row['owner'], qid)]
            route_path, route = source_path, source
            kind = 'EXISTING_SINGLE'
        else:
            route_path, route = routes[(row['mode'],row['prefix'],row['owner'],qid)]
            selected = route['effective_expert']
            assert selected is None or byid[selected] <= row['prefix']
            found = bases.get(qid) if selected is None else singles.get((byid[selected],qid))
            if found is None:
                assert selected is not None
                key = (byid[selected], qid)
                request = requests.setdefault(key, dict(owner=key[0], selected=selected,
                    query=row['query'], judge_binding=route['binding']['judge_input'], consumers=[]))
                assert request['judge_binding'] == route['binding']['judge_input']
                request['consumers'].append(dict(row=row, route_path=str(route_path)))
                counts['missing_consumers'] += 1
                continue
            source_path, source = found
            assert source['binding']['judge_input'] == route['binding']['judge_input']
            if selected is not None:
                assert source['effective_expert'] == selected
            kind = 'EXISTING_BASE' if selected is None else 'EXISTING_SELECTED_EXPERT'
        assert source['binding']['judge_input']['question'] == row['query']['question']
        assert source['binding']['judge_input']['reference'] == row['query']['reference']
        value = copy.deepcopy(source)
        value.update(method='M13P_160', mode=row['mode'], prefix=row['prefix'], owner=row['owner'],
            consumer_id=consumer_id(row), binding=route['binding'], route=route['route'],
            effective_expert=route['effective_expert'],
            prediction_reused=True, new_GPU_generation=False,
            provenance=dict(kind=kind, prediction=str(source_path), actual_routing=str(route_path),
                routing_execution=route['execution'], full_judge_binding_equal=True,
                no_C_or_M13_prediction_substitution=True),
            functional_bank_composition=row['mode'] != 'single')
        write(OUT / 'outputs' / (name(consumer_id(row)) + '.json'), value)
        counts[kind] += 1
        provenance.append(dict(consumer_id=consumer_id(row), prediction=str(source_path),
            actual_route=str(route_path), kind=kind))
    requests = list(requests.values())
    assert len(requests) == 287 and len({r['owner'] for r in requests}) == 94
    assert counts == dict(EXISTING_SINGLE=2956, EXISTING_SELECTED_EXPERT=4101,
                         EXISTING_BASE=1, missing_consumers=436)
    write(OUT / 'MISSING_REQUESTS.json', dict(requests=requests,
        missing_unique_predictions=287, affected_experts=94, missing_consumers=436,
        excluded='No completed single generation, BalancEdit, AlphaEdit or GRACE is rerun'))
    write(OUT / 'REUSE_PROVENANCE.json', provenance)
    # The actual growing membership is part of the replay contract, even though
    # expert outputs and routing are reused rather than recomputed on a GPU.
    write(OUT / 'BANK_MEMBERSHIP.json', [dict(prefix=i,
        logical_ids=[t['edit_id'] for t in tasks[:i]]) for i in range(1,147)])
    write(OUT / 'REUSE_FROZEN.json', dict(status='FROZEN_REUSE_WITH_EXPLICIT_GAPS', counts=counts,
        exact_input_runtime_generation_bindings=True, actual_R0_routing_reused=True,
        independent_codebook_only=True, new_full_bank_GPU_run=False,
        target_consumers=7494, time=time.time()))


if __name__ == '__main__':
    main()
