"""Read frozen artifacts and write aggregate-only review data; never call a model."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(os.environ.get('MEDTRACE_BASELINE_CODE', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from scripts.medtrace import astra_judge_bundle as io
from scripts.medtrace.stage15 import bootstrap
from scripts.medtrace.stage15_sources import norm, KAPPA
from scripts.medtrace.stage16 import summarize, exact_p, source_groups, route_values
from scripts.medtrace.stage4_scope import accepted


def keyed(value):
    # Exact structured tuple matching, not question-only matching or transfer hashing.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def analyze(bundle, output):
    HERE = bundle / 'operator'
    source = io.packet_rows(HERE)
    manifest = io.read(HERE / 'MANIFEST.json')
    execution = io.read(HERE / 'EXECUTION_RECORD.json')
    decisions, visible = [], []
    for spec in manifest['batches']:
        name = spec['batch_id']
        batch = io.read(HERE.parent / 'judge_only' / (name + '.input.json'))
        decisions.extend(io.validate(batch, io.read(HERE / 'responses' / (name + '.json'))))
        visible.extend(batch['records'])
    assert visible == [{k: r[k] for k in sorted(io.VISIBLE_FIELDS)} for r in source]
    verdict_rows = [io.loads(s) for s in (HERE / 'VERDICTS_ASTRA.jsonl').read_text().splitlines()]
    assert len(verdict_rows) == len(source) == len(decisions) == 326
    for r, d, original in zip(verdict_rows, decisions, source):
        assert {k: r[k] for k in d} == d
        assert r['opaque_query_id'] == original['opaque_query_id']
        assert r['adjudication_pass'] == original['adjudication_pass'] == 'SOURCE_ANSWER_JUDGE'
        assert r['parse_valid'] is True and type(r['is_correct']) is bool
        assert r['judge_model'] == execution['actual_model'] == io.MODEL
        assert r['judge_protocol_version'] == io.PROTOCOL
        assert r['source_judge_config_sha256'] == manifest['source_judge_config_sha256']
    verdicts = {r['opaque_query_id']: r['is_correct'] for r in verdict_rows}
    side = io.read(HERE / 'SIDECAR.json')
    identities = {keyed(full): key for key, full in side['bindings'].items()}
    assert len(verdicts) == len(identities) == 326
    evidence = HERE / 'stage16_evidence'
    tasks = io.read(evidence / 'QUEUE.json')
    assert len(tasks) == 133 and len({t['order'] for t in tasks}) == 133
    old = io.read(evidence / 'A_BOUND_ROWS.json')
    assert len(old) == 2514
    rows, replays, paired, used, roles = [], [], {}, set(), defaultdict(set)
    files = list((evidence / 'edits').glob('e*/*/RESULT.json'))
    assert len(files) == 266
    for t in tasks:
        assert len(t['probes']) == 1
        for method in ('C_NO_H', 'BE'):
            result = io.read(evidence / 'edits' / f"e{t['order']:03d}" / method / 'RESULT.json')
            assert result['status'] == 'COMPLETE' and result['method'] == method
            assert result['canonical_edit_id'] == t['canonical_edit_id']
            assert len(result['entries']) == 1
            replays.extend(result['replay'])
            for e in result['entries']:
                p, b, f = e['probe'], e['base'], e['forced']
                assert e['status'] == 'COMPLETE' and p == t['probes'][0]
                assert p['metric'] == 'I_Locality' and p['status'] == 'AVAILABLE'
                assert b['binding'] == f['binding'] and b['binding']['question'] == p['question']
                assert b['binding']['image_source_sha256'] == p['image_sha256']
                assert b['binding']['no_image'] is False
                assert e['checkpoint_binding']['canonical_edit_id'] == t['canonical_edit_id']
                assert e['checkpoint_binding']['method'] == method
                assert e['checkpoint_binding']['stage15_public_commit'] == '0c51b959c80dcda205e14c2de92b7ae43c6e8dcf'
                assert e['route']['activated'] == accepted(e['route'], 0.)
                assert e['rc_on'] == accepted(e['route'], KAPPA)
                pair_id = (t['order'], p['probe_id'])
                shared = (p, b['binding'], b['raw_answer'], b['raw_token_ids'], e['route'], e['rc_on'])
                if pair_id in paired:
                    assert paired[pair_id] == shared
                paired[pair_id] = shared
                keys = []
                for role, out in [('BASE', b), (method, f)]:
                    full = dict(question=p['question'], image=p['image_sha256'], input=out['binding'],
                                reference=p['reference'], raw_answer=out['raw_answer'], raw_tokens=out['raw_token_ids'],
                                protocol=side['protocol_sha256'], judge_type='SOURCE_ANSWER_JUDGE')
                    key = identities[keyed(full)]
                    used.add(key); roles[role].add(key); keys.append(key)
                d, radius, ratio, margin = route_values(e['route'])
                rows.append(dict(edit=t['order'], probe=p['probe_id'], method=method, metric=p['metric'],
                    source_ref=p['source_image'], image_hash=p['image_sha256'],
                    actual_image_hash=b['binding']['image_tensor'], native_hash=t['image']['sha256'],
                    r0=e['route']['activated'], rc=e['rc_on'], d=d, radius=radius, ratio=ratio, margin=margin,
                    base_exact=norm(b['raw_answer']) == norm(p['reference']),
                    forced_exact=norm(f['raw_answer']) == norm(p['reference']),
                    base_semantic=verdicts[keys[0]], forced_semantic=verdicts[keys[1]],
                    forced_preserve=norm(f['raw_answer']) == norm(b['raw_answer']),
                    base_target_copy=norm(b['raw_answer']) == norm(t['raw_record']['alt']),
                    forced_target_copy=norm(f['raw_answer']) == norm(t['raw_record']['alt']),
                    base_cap=b['cap_hit'], forced_cap=f['cap_hit']))
    assert used == set(verdicts) and len(rows) == 266 and len(paired) == 133
    old_local = [r for r in old if r['metric'] == 'I_Locality']
    assert len(old_local) == 134
    assert not {r['edit'] for r in old_local} & {r['edit'] for r in rows}
    assert all(r['passed'] is True for r in replays)
    modes = ('BASE', 'FORCED_ON', 'BE_ROUTE_R0', 'RC_FIXED_OLD16')
    def on(r, mode):
        return mode == 'FORCED_ON' or mode == 'BE_ROUTE_R0' and r['r0'] or mode == 'RC_FIXED_OLD16' and r['rc']
    def value(r, mode, score):
        if score == 'exact':
            return r['forced_preserve'] if on(r, mode) else True
        return r['forced_semantic'] if on(r, mode) else r['base_semantic']
    tables, effects, sensitivity = [], [], []
    components = source_groups(old + rows)
    for cohort, panel in [('OLD_SUPPORTED_QWEN', old_local), ('NEWLY_RESTORED_ASTRA', rows),
                          ('COMBINED_OUTPUT_ONLY_NO_JUDGE', old_local + rows)]:
        scores = ('exact',) if cohort.startswith('COMBINED') else ('exact', 'semantic')
        assert len({r['edit'] for r in panel}) * 2 == len(panel)
        for score in scores:
            for mode in modes:
                for method in ('C_NO_H', 'BE'):
                    subset = [r for r in panel if r['method'] == method]
                    tables.append(dict(cohort=cohort, method=method, mode=mode, score=score,
                                       **summarize(subset, [on(r, mode) for r in subset], score)))
                pairs = defaultdict(dict)
                for r in panel:
                    pairs[r['edit']][r['method']] = value(r, mode, score)
                assert all(set(p) == {'C_NO_H', 'BE'} for p in pairs.values())
                delta = {edit: int(p['C_NO_H']) - int(p['BE']) for edit, p in pairs.items()}
                lo, hi = bootstrap(list(delta.values()))
                wins, losses = list(delta.values()).count(1), list(delta.values()).count(-1)
                effects.append(dict(cohort=cohort, mode=mode, score=score, n=len(delta), delta=mean(delta.values()),
                    ci_low=lo, ci_high=hi, both_correct=sum(all(p.values()) for p in pairs.values()),
                    NOH_only=wins, BE_only=losses, neither=sum(not any(p.values()) for p in pairs.values()),
                    supplemental_exact_two_sided_p=exact_p(wins, losses)))
                groups = defaultdict(list)
                for edit, v in delta.items(): groups[components[edit]].append(v)
                group_means = [mean(v) for v in groups.values()]
                gl, gh = bootstrap(group_means)
                leave_one_out = [mean(v for g, vs in groups.items() if g != omit for v in vs) for omit in groups]
                sensitivity.append(dict(cohort=cohort, mode=mode, score=score, source_components=len(groups),
                    component_equal_weighted_delta=mean(group_means), ci_low=gl, ci_high=gh,
                    leave_one_component_out_min=min(leave_one_out), leave_one_component_out_max=max(leave_one_out)))
    summary = dict(status='LOCAL_REVIEW_AGGREGATION_COMPLETE_NOT_PIPELINE_IMPORTED',
        created_at_utc=datetime.now(timezone.utc).isoformat(), new_training_steps=0, new_generation_calls=0,
        new_judge_calls=0, old_qwen_verdicts_changed=False, mixed_judge_semantic_aggregate_created=False,
        source_binding_check='Exact structured full-tuple equality; no new file hashes',
        result_files=len(files), result_file_bytes=sum(f.stat().st_size for f in files),
        new_edits=len(tasks), new_writer_probes=len(rows), unique_judge_tuples=len(verdicts),
        verdict_counts=dict(Counter(str(v).lower() for v in verdicts.values())),
        unique_judge_tuples_by_role={role:len(ids) for role,ids in roles.items()},
        new_unique_source_images=len({r['image_hash'] for r in rows}),
        recorded_replay_count=len(replays), recorded_replay_signatures=[list(s) for s in sorted({tuple(r['signature']) for r in replays})],
        all_recorded_replays_passed=True, new_no_text_only_probes=True,
        known_components_across_old_plus_new=len(set(components.values())),
        source_component_caveat='Existing identity/PMC-article proxy only; not patient independence or exhaustive near-duplicate audit',
        statistical_method='Existing seed 20260911; 2000 edit bootstrap replicates, order statistics 49/1949; exact paired supplemental p; no multiplicity correction; exploratory',
        tables=tables, paired_effects=effects, component_sensitivity=sensitivity)
    io.write_new(output, summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('tables','paired_effects','component_sensitivity')},indent=2))
    for r in tables:
        if r['mode'] != 'BASE':
            print(r['cohort'],r['score'],r['mode'],r['method'],f"{r['total_success']}/{r['probes']}",
                  'ON',r['on'],'damage',r['base_correct_damage'],'correction',r['base_wrong_correction'],
                  'target_copy',r['target_copy_total'],'cap_hits',r['cap_hits'])
    print('PAIRED', json.dumps([e for e in effects if e['mode'] != 'BASE'],indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.bundle.resolve(), args.output.resolve())
