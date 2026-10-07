"""Reproduce frozen panels from compact bindings; retain all missing score bounds."""
from collections import Counter, defaultdict
import importlib.util
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time

RUN = Path(os.environ['RUN_ROOT'])
sys.path.insert(0, str(RUN/'private/tools'))
import common
spec = importlib.util.spec_from_file_location('original_report', RUN/'private/tools/report.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def bootstrap(coeff, scores, groups):
    # Linear bounds can be aggregated before resampling when missing-key signs agree.
    signs = defaultdict(set)
    for c in coeff.values():
        for k, v in c.items():
            if scores.get(k) is None and v:
                signs[k].add(v > 0)
    if any(len(s) > 1 for s in signs.values()):
        return r.bootstrap(coeff, scores, groups)
    if not coeff:
        return dict(edit_ci=None, source_ci=None)
    units = {e: r.score_bounds(c, scores) for e, c in coeff.items()}
    clusters = defaultdict(list)
    for e in coeff:
        clusters[groups[e]].append(e)
    def sample(packages):
        packs = [(sum(units[e][0] for e in p), sum(units[e][1] for e in p), len(p)) for p in packages]
        rng = random.Random(20260912)
        low, high = [], []
        for _ in range(10000):
            draw = rng.choices(packs, k=len(packs))
            n = sum(x[2] for x in draw)
            low.append(sum(x[0] for x in draw)/n)
            high.append(sum(x[1] for x in draw)/n)
        return [sorted(low)[249]*100, sorted(high)[9749]*100]
    return dict(edit_ci=sample([[e] for e in coeff]), source_ci=sample(list(clusters.values())),
        source_groups=len(clusters), interpretation='Development uncertainty only; missing scores bounded, not dropped')


def selfcheck():
    c = {'a': {'x': .5, 'z': .5}, 'b': {'y': 1}, 'c': {'x': .25, 'z': .75}}
    s = {'x': 1, 'y': 0, 'z': None}
    g = {'a': 'one', 'b': 'one', 'c': 'two'}
    fast, original = bootstrap(c, s, g), r.bootstrap(c, s, g)
    for k in ('edit_ci', 'source_ci'):
        assert all(abs(a-b) < 1e-10 for a, b in zip(fast[k], original[k]))


def main():
    selfcheck()
    root = RUN/'private/judge_astra_medium'
    assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    def connect(path):
        db = sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        return db
    old = connect(RUN/'private/judge_common/queue.sqlite')
    new = connect(root/'queue.sqlite')
    original = {x['key']: x for x in old.execute('SELECT * FROM payload')}
    paired = {x['old_key']: x for x in new.execute('SELECT * FROM paired')}
    updated = {x['key']: x for x in new.execute('SELECT * FROM payload')}
    assert len(original) == len(paired) == len(updated) == 2137
    qwen = {k: v['correct'] for k, v in original.items()}
    astra = {k: updated[v['new_key']]['correct'] for k, v in paired.items()}
    assert all(x['status'] in ('FORMAT_VALID', 'MISSING') for x in updated.values())
    for k, pair in paired.items():
        a, b = json.loads(original[k]['binding']), json.loads(updated[pair['new_key']]['binding'])
        assert b.pop('judge')['reasoning_effort'] == 'medium'
        a.pop('judge')
        assert a == b and pair['qwen_correct'] == qwen[k]
    consumers = list(old.execute('SELECT * FROM consumer'))
    incoming = {c['id']: c for c in new.execute('SELECT * FROM consumer')}
    assert len(consumers) == len(incoming) == 9501
    for c in consumers:
        n = incoming[c['id']]
        assert all(c[k] == n[k] for k in c.keys() if k != 'payload_key')
        assert paired[c['payload_key']]['new_key'] == n['payload_key']
    ledger = common.read(RUN/'private/EVAL_LEDGER.json')
    tasks = [t for t in common.read(RUN/'private/QUEUES.json')['tasks'] if t['cohort'] == 'P2']
    retro = common.read(RUN/'private/BENCHMARK146_QUEUE.json')['tasks']
    records = []
    for c in consumers:
        if c['folder'] != 'PANEL':
            continue
        q = ledger['queries'][c['query_id']]
        ts = retro if c['method'] == 'TT88_W0_RETRO146' else tasks
        active = {(t['native']['image_sha256'], t['native']['question']) for t in ts[:c['prefix']]} if c['mode'] == 'bank_R0' else set()
        phase = dict(arm=c['method'], node=160 if c['method'] in ('CE_U_MULTI','U_ONLY','HOLD_U') else 0, prefix=c['prefix'])
        # Route counts were already audited; this projection only recomputes scores.
        d = dict(binding=dict(phase=phase, input=q, mode=c['mode'], owner_order=c['edit_order']),
                 active_target=(q['image_sha256'],q['question']) in active, effective_expert=None)
        records.append((d, c['payload_key']))
    combined_scores = {**{'A:'+k:v for k,v in astra.items()}, **{'Q:'+k:v for k,v in qwen.items()}}
    sections, coefficients, groups, exact = {}, {}, {}, set()
    for section, filename in [('core','CORE_RESULTS.json'),('structure8','FINAL_COMPARISON.json'),('benchmark146','BENCHMARK146_RESULTS.json')]:
        panels = []
        for prior in common.read(RUN/'public'/filename)['panels']:
            cohort, arm, node, mode, task, prefix = (prior[k] for k in ('cohort','arm','node','mode','task','prefix'))
            ts = retro if section == 'benchmark146' else tasks[:8] if section == 'structure8' else tasks
            if mode == 'bank_R0':
                ts = ts[:prefix]
            check, coeff, g = r.panel(records, qwen, cohort, arm, node, mode, task, ts, prefix)
            for k in ('known_correct','observations','missing','edit_units'):
                assert check[k] == prior[k], (section, arm, mode, task, prefix, k)
            assert check['macro'] is None and prior['macro'] is None or abs(check['macro']-prior['macro']) < 1e-9
            metric, _, _ = r.panel(records, astra, cohort, arm, node, mode, task, ts, prefix)
            metric.pop('route_ON')
            metric.update(bootstrap(coeff, astra, g))
            delta = {e: {**{'A:'+k:v for k,v in c.items()}, **{'Q:'+k:-v for k,v in c.items()}} for e,c in coeff.items()}
            bound = r.score_bounds(r.combine([{k:v/len(delta) for k,v in c.items()} for c in delta.values()]),combined_scores) if delta else [None,None]
            metric.update(Qwen_macro=prior['macro'], Astra_minus_Qwen_pp=[v*100 if v is not None else None for v in bound], judge_delta_ci=bootstrap(delta,combined_scores,g))
            panels.append(metric)
            coefficients[(section,arm,mode,prefix,task)] = coeff
            groups.update(g)
            if section == 'benchmark146' and mode == 'single_R0' and task in ('T0','T1G'):
                for c in coeff.values():
                    for key in c:
                        b = json.loads(original[key]['binding'])
                        if qwen[key] == 0 and b['reference'] and b['output']['raw_answer'] == b['reference']:
                            exact.add(key)
        sections[section] = panels
    assert len(exact) == 9
    contrasts = []
    def contrast(section, a, b, mode, prefix, task):
        aa = coefficients[(section,a,mode,prefix,task)]
        bb = coefficients[(section,b,mode,prefix,task)]
        cc = {e:r.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa if e in bb}
        bounds = r.score_bounds(r.combine([{k:v/len(cc) for k,v in c.items()} for c in cc.values()]), astra)
        contrasts.append(dict(section=section, comparison=a+'-'+b, mode=mode, prefix=prefix,task=task,
            delta_bounds_pp=[v*100 for v in bounds],paired_edits=len(cc),**bootstrap(cc,astra,groups)))
    for prefix in (8,24):
        for a in ('CE_U_MULTI','U_ONLY','HOLD_U'):
            for b in ('FROZEN_W0','CE_U_MULTI'):
                if a != b:
                    for task in ('T0','T1G','T2G'):
                        contrast('core',a,b,'bank_R0',prefix,task)
    for mode in ('single_R0','bank_R0'):
        for task in ('T0','T1G','T2G'):
            contrast('structure8','LORA_W0','TT88_W0' if mode == 'single_R0' else 'FROZEN_W0',mode,1 if mode == 'single_R0' else 8,task)
    def transitions(keys):
        return dict(Counter(str(qwen[k])+'to'+('missing' if astra[k] is None else str(astra[k])) for k in keys))
    bypath = {c['path']:c['payload_key'] for c in consumers}
    old_losses = [x for x in common.read(RUN/'private/CURRENT_FAILURE_TRANSITIONS.json')['records'] if x['task']=='T1G' and qwen[bypath[x['single']]]==1 and qwen[bypath[x['bank']]]==0]
    assert len(old_losses) == 76
    route_losses = Counter(str(astra[bypath[x['single']]])+'to'+str(astra[bypath[x['bank']]]) for x in old_losses)
    resource = common.read(RUN/'RESOURCE_LEDGER.json')
    result = dict(status='SCORING_TERMINAL_WITH_50_MISSING', model='gpt-6-astra',reasoning_effort='medium',
        all_payloads=2137,valid_payloads=sum(v is not None for v in astra.values()),missing_payloads=sum(v is None for v in astra.values()),
        consumers=9501,missing_consumers=sum(astra[c['payload_key']] is None for c in consumers),
        Judge_attempts=resource['Judge_attempts'],GPU_hours=resource['gpu_seconds_used']/3600,
        panels=sections,contrasts=contrasts,payload_transitions=transitions(qwen),consumer_transitions=transitions(c['payload_key'] for c in consumers),
        exact_reference_nine=transitions(exact),old_76_routing_losses_Astra_transitions=dict(route_losses),
        reconstruction_validation='All77 frozen Qwen panels reproduced: original denominators, masks, correct counts and macro scores',
        full_consumer_and_output_binding_preserved=True,bootstrap_draws=10000,bootstrap_seed=20260912,
        scientific_status='Retrospective development; not independent confirmation',locality='NA_SCOPE_NOT_QUALIFIED',
        failure='One50-record batch reached600s local timeout; preserved missing, no retry',new_inference=False)
    common.write(RUN/'public/ASTRA_MEDIUM_RESULTS.json',result)
    common.write(root/'REPORT_COMPLETE.json',dict(epoch=time.time(),status='AGGREGATES_COMPLETE_PUBLICATION_PENDING',valid=2087,missing=50))
    print(json.dumps({k:v for k,v in result.items() if k not in ('panels','contrasts')},ensure_ascii=False))


if __name__ == '__main__':
    main()
