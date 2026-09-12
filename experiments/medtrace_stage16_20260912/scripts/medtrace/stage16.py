#!/usr/bin/env python3
"""Stage16: offline decomposition of the immutable Stage15 generation/Judge ledger."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
import sys
import time

ROOT = Path(os.environ.get('MEDTRACE_BASELINE_CODE', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from scripts.medtrace.stage15 import csv_write, judge_identity, bootstrap
from scripts.medtrace.stage15_sources import read, write, digest, norm, KAPPA, MODES
from scripts.medtrace.stage4_scope import accepted

BASELINE = '0c51b959c80dcda205e14c2de92b7ae43c6e8dcf'
METHODS = ('C_NO_H', 'BE')
LOCAL = ('T_Locality', 'I_Locality')
POSITIVE = ('Reliability', 'T_Generality', 'I_Generality')


def rate(n, d):
    return n / d if d else None


def route_values(route):
    d, r = route['nearest_distance'], route['radius']
    if d is None or r is None or not math.isfinite(d) or not math.isfinite(r) or d < 0 or r <= 0:
        raise ValueError('Invalid historical distance/radius; do not change the routing rule')
    return d, r, d/r, (r-d)/max(r, 1e-12)


def quantiles(values):
    v = sorted(values)
    def q(p):
        i = (len(v)-1)*p; lo = math.floor(i); hi = math.ceil(i)
        return v[lo] + (v[hi]-v[lo])*(i-lo)
    return dict(n=len(v), minimum=v[0], p05=q(.05), p25=q(.25), median=q(.5),
                p75=q(.75), p95=q(.95), maximum=v[-1], mean=mean(v))


def exact_p(wins, losses):
    n = wins+losses
    return min(1., 2*sum(math.comb(n, k) for k in range(min(wins, losses)+1))/2**n) if n else 1.


def auroc(positive, negative):
    return mean(float(p > n)+.5*float(p == n) for p in positive for n in negative)


def load_rows(old):
    """Keep full identities private; never join outputs by normalized question alone."""
    tasks = read(old/'private/QUEUE.json')
    lock = read(old/'public/PROTOCOL_AND_METHOD_LOCK.json')
    assert len(tasks) == 200 and digest(tasks) == lock['queue_sha256']
    protocol = read(old/'private/JUDGE_LOCK.json')['config_sha256']
    assert protocol == lock['judging']['config_sha256']
    side = read(old/'private/judge/SIDECAR.json')['bindings']
    verdicts = {}
    for j in map(json.loads, (old/'private/judge/OUTPUT.jsonl').open()):
        assert j['parse_valid'] and type(j['is_correct']) is bool
        assert j['opaque_query_id'] not in verdicts
        verdicts[j['opaque_query_id']] = j['is_correct']
    assert set(side) == set(verdicts)
    rows, pair = [], {}
    for task in tasks:
        directory = old/'private/edits'/f"e{task['order']:03d}"
        assert (directory/'ROUTER.pt').is_file()
        for method in METHODS:
            checkpoint = directory/method/('latest.pt' if method == 'C_NO_H' else 'editor_state.pt')
            assert checkpoint.is_file()
            result = read(directory/method/'RESULT.json')
            assert result['canonical_edit_id'] == task['canonical_edit_id'] and result['method'] == method
            for entry in result['entries']:
                if entry['status'] != 'COMPLETE':
                    continue
                probe = entry['probe']; b = entry['base']; f = entry['forced']
                assert b['binding'] == f['binding'] and digest(b['binding']) == entry['eqkey']
                assert b['binding']['question'] == probe['question']
                assert b['binding']['image_source_sha256'] == probe.get('image_sha256')
                assert b['binding']['no_image'] == (probe['image_path'] is None)
                if probe['metric'] == 'T_Locality':
                    assert b['binding']['no_image'] and all(-200 not in ids for ids in b['binding']['prompt_ids'])
                assert accepted(entry['route'], KAPPA) == entry['rc_on']
                d, radius, ratio, margin = route_values(entry['route'])
                assert entry['route']['activated'] == accepted(entry['route'], 0.)
                keys = []
                for output in (b, f):
                    key, _, binding = judge_identity(entry, output, protocol)
                    assert side[key] == binding
                    keys.append(key)
                identity = (task['order'], probe['probe_id'])
                # Both methods must have exactly the same input, Base output and routing.
                shared = dict(probe=probe, base=b, route=entry['route'], eqkey=entry['eqkey'])
                if identity in pair:
                    previous = pair[identity]
                    assert previous['probe'] == probe and previous['route'] == entry['route']
                    assert previous['eqkey'] == entry['eqkey']
                    assert all(previous['base'][k] == b[k] for k in ('raw_answer', 'raw_token_ids', 'binding'))
                else:
                    pair[identity] = shared
                local = probe['metric'] in LOCAL
                rows.append(dict(edit=task['order'], probe=probe['probe_id'], method=method,
                    metric=probe['metric'], subtype=probe.get('hop', probe.get('attack_type', 'ALL')),
                    source_ref=probe.get('source_image'), image_hash=probe.get('image_sha256'),
                    actual_image_hash=b['binding']['image_tensor'] if not b['binding']['no_image'] else None,
                    native_hash=task['image']['sha256'], checkpoint=str(checkpoint),
                    canonical_edit_id=task['canonical_edit_id'], input_identity=entry['eqkey'], judge_keys=keys,
                    d=d, radius=radius, ratio=ratio, margin=margin, r0=entry['route']['activated'], rc=entry['rc_on'],
                    base_exact=norm(b['raw_answer']) == norm(probe['reference']),
                    forced_exact=norm(f['raw_answer']) == norm(probe['reference']),
                    base_semantic=verdicts[keys[0]], forced_semantic=verdicts[keys[1]],
                    forced_preserve=norm(f['raw_answer']) == norm(b['raw_answer']),
                    forced_target_copy=norm(f['raw_answer']) == norm(task['raw_record']['alt']),
                    base_target_copy=norm(b['raw_answer']) == norm(task['raw_record']['alt']),
                    base_cap=b['cap_hit'], forced_cap=f['cap_hit']))
    assert len(rows) == 2514 and all(sum(r['method'] == m for r in rows) == 1257 for m in METHODS)
    return tasks, rows, dict(queue_sha256=lock['queue_sha256'], judge_protocol_sha256=protocol,
                            reused_judge_tuples=len(verdicts), historical_writer_checkpoints=400,
                            full_context_bindings_verified=True, historical_files_opened_read_only=True)


def summarize(rs, on, score):
    local = rs[0]['metric'] in LOCAL
    base = [r['base_'+score] for r in rs]
    forced = [r['forced_preserve'] if local and score == 'exact' else r['forced_'+score] for r in rs]
    baseline = [True]*len(rs) if local and score == 'exact' else base
    result = [f if a else b for a, f, b in zip(on, forced, baseline)]
    n_on = sum(on); off = len(rs)-n_on
    by = defaultdict(list)
    for r, v in zip(rs, result): by[r['edit']].append(v)
    damage = sum(a and b and not f for a, b, f in zip(on, base, [r['forced_'+score] for r in rs]))
    correction = sum(a and not b and f for a, b, f in zip(on, base, [r['forced_'+score] for r in rs]))
    return dict(edits=len(by), probes=len(rs), on=n_on, off=off, activation=rate(n_on, len(rs)),
        on_success=sum(a and f for a, f in zip(on, forced)),
        success_given_on=rate(sum(a and f for a, f in zip(on, forced)), n_on),
        base_correct_given_off=rate(sum(not a and b for a, b in zip(on, base)), off),
        total_success=sum(result), probe_micro=mean(result), edit_macro=mean(mean(v) for v in by.values()),
        base_reference_correct=sum(base), base_correct_damage=damage, base_wrong_correction=correction,
        off_base_preserved=off if local else None,
        on_output_preserved=sum(a and r['forced_preserve'] for a, r in zip(on, rs)) if local else None,
        on_output_damaged=sum(a and not r['forced_preserve'] for a, r in zip(on, rs)) if local else None,
        target_copy_total=sum(r['forced_target_copy'] if a else r['base_target_copy'] for r, a in zip(rs, on)) if local else None,
        target_copy_on=sum(a and r['forced_target_copy'] for a, r in zip(on, rs)) if local else None,
        cap_hits=sum(r['forced_cap'] if a else r['base_cap'] for r, a in zip(rs, on)),
        score_definition='OUTPUT_PRESERVATION' if local and score == 'exact' else 'SOURCE_ACCURACY' if local else 'TARGET_ADHERENCE')


def source_groups(rows):
    parent = {r['edit']:r['edit'] for r in rows}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owners = {}
    for r in rows:
        if r['metric'] == 'U_fit_diagnostic': continue
        keys = [('native', r['native_hash'])]
        for field in ('image_hash', 'actual_image_hash', 'source_ref'):
            if r[field]: keys.append((field, r[field]))
        match = re.search(r'PMC\d+', r['source_ref'] or '')
        if match: keys.append(('article', match.group()))
        for key in keys:
            if key in owners: parent[root(r['edit'])] = root(owners[key])
            else: owners[key] = r['edit']
    return {i:root(i) for i in parent}


def analyze(old, run):
    started = time.time(); tasks, rows, binding = load_rows(old)
    public = run/'public'; write(run/'private/A_BOUND_ROWS.json', rows)
    groups = defaultdict(list)
    for r in rows: groups[r['method'], r['metric'], r['subtype']].append(r)
    decomposition = []
    for (method, metric, subtype), rs in sorted(groups.items()):
        for mode in ('BASE', *MODES):
            on = [mode == 'FORCED_ON' or mode == 'BE_ROUTE_R0' and r['r0'] or mode == 'RC_FIXED_OLD16' and r['rc'] for r in rs]
            for score in ('exact', 'semantic'):
                decomposition.append(dict(method=method, metric=metric, subtype=subtype, mode=mode, score=score,
                                          **summarize(rs, on, score)))
    csv_write(public/'ROUTE_WRITER_DECOMPOSITION.csv', decomposition)
    margins = []; aucs = []
    for (method, metric, subtype), rs in sorted(groups.items()):
        if method != 'C_NO_H': continue
        for variable in ('d', 'radius', 'ratio', 'margin'):
            margins.append(dict(metric=metric, subtype=subtype, variable=variable,
                r0_on=sum(r['r0'] for r in rs), rc_old_on=sum(r['rc'] for r in rs), **quantiles([r[variable] for r in rs])))
    for pos in POSITIVE:
        for neg in LOCAL:
            p = groups['C_NO_H', pos, 'ALL']; n = groups['C_NO_H', neg, 'ALL']
            aucs.append(dict(positive_family=pos, negative_family=neg, positive_n=len(p), negative_n=len(n),
                positive_role='OFFICIAL_IN_SCOPE_PROXY', negative_role='OFFICIAL_LOCALITY_PROXY_NOT_REVIEWED_H',
                AUROC=auroc([r['margin'] for r in p], [r['margin'] for r in n]), status='POST_HOC_DIAGNOSTIC_ONLY'))
    csv_write(public/'ROUTER_MARGIN_SUMMARY.csv', margins)
    csv_write(public/'ROUTER_PAIRWISE_AUROC.csv', aucs)
    # Every reachable rejection transition, not an arbitrary grid or a selected new main threshold.
    ks = sorted({0., KAPPA, *[k for r in rows for k in (r['margin'], math.nextafter(r['margin'], math.inf)) if r['r0'] and k >= 0]})
    thresholds = []
    for k in ks:
        for (method, metric, subtype), rs in sorted(groups.items()):
            for score in ('exact', 'semantic'):
                thresholds.append(dict(kappa=k, method=method, metric=metric, subtype=subtype, score=score,
                    status='HISTORICAL_FIXED_REFERENCE' if k == KAPPA else 'HISTORICAL_R0_REFERENCE' if k == 0 else 'POST_HOC_DIAGNOSTIC_ONLY',
                    **summarize(rs, [r['r0'] and r['margin'] >= k for r in rs], score)))
    csv_write(public/'POSTHOC_THRESHOLD_TRADEOFF.csv', thresholds)
    contingency = []; sensitivity = []; clusters = source_groups(rows)
    for metric in LOCAL:
        left = {r['edit']:r for r in groups['C_NO_H', metric, 'ALL']}
        right = {r['edit']:r for r in groups['BE', metric, 'ALL']}
        assert left.keys() == right.keys()
        for mode in MODES:
            for score in ('output_preservation', 'source_semantic'):
                def outcome(r):
                    on = mode == 'FORCED_ON' or (r['r0'] if mode == 'BE_ROUTE_R0' else r['rc'])
                    return (r['forced_preserve'] if on else True) if score == 'output_preservation' else r['forced_semantic'] if on else r['base_semantic']
                cells = Counter(); differences = {}; by_cluster = defaultdict(list)
                for i in left:
                    a, b = outcome(left[i]), outcome(right[i]); cells[int(a), int(b)] += 1
                    differences[i] = int(a)-int(b); by_cluster[clusters[i]].append(differences[i])
                lo, hi = bootstrap(list(differences.values()))
                wins, losses = cells[1, 0], cells[0, 1]
                contingency.append(dict(metric=metric, mode=mode, score=score, n=len(left),
                    both_correct=cells[1, 1], NOH_only=wins, BE_only=losses, neither=cells[0, 0],
                    delta=mean(differences.values()), historical_edit_bootstrap_low=lo, historical_edit_bootstrap_high=hi,
                    supplemental_exact_two_sided_p=exact_p(wins, losses), status='DIAGNOSTIC_NOT_NEW_CONFIRMATION'))
                cluster_means = [mean(v) for v in by_cluster.values()]
                clo, chi = bootstrap(cluster_means)
                leave_one = [mean([v for i, v in differences.items() if clusters[i] != group]) for group in by_cluster if len(by_cluster)>1]
                sensitivity.append(dict(metric=metric, mode=mode, score=score, edits=len(left), known_source_components=len(by_cluster),
                    edit_weighted_delta=mean(differences.values()), component_equal_weighted_delta=mean(cluster_means),
                    component_bootstrap_low=clo, component_bootstrap_high=chi,
                    leave_one_component_out_min=min(leave_one) if leave_one else None,
                    leave_one_component_out_max=max(leave_one) if leave_one else None,
                    independence='SOURCE_COMPONENT_PROXY; patient/study and unobserved near-duplicates UNKNOWN'))
    csv_write(public/'LOCALITY_PAIRED_CONTINGENCY.csv', contingency)
    csv_write(public/'SOURCE_GROUP_SENSITIVITY.csv', sensitivity)
    ports = []
    for row in decomposition:
        if row['metric'] != 'Portability': continue
        rs = groups[row['method'], row['metric'], row['subtype']]
        base_count = sum(r['base_'+row['score']] for r in rs)
        ports.append(dict(row, base_success=base_count, net_success_delta=row['total_success']-base_count,
                          paired_delta=row['probe_micro']-base_count/len(rs)))
    csv_write(public/'PORTABILITY_BASE_DELTA.csv', ports)
    source_counts = {metric:dict(probes=len(rs), unique_source_images=len({r['image_hash'] for r in rs if r['image_hash']}),
        unique_model_input_images=len({r['actual_image_hash'] for r in rs if r['actual_image_hash']}))
        for (method, metric, subtype), rs in groups.items() if method == 'C_NO_H' and subtype == 'ALL'}
    write(public/'STAGE16_A_AUDIT.json', dict(status='COMPLETE', public_baseline=BASELINE, **binding,
        methods=list(METHODS), rows=len(rows), threshold_points=len(ks), threshold_selection='NONE',
        official_scope_roles='benchmark proxies, not independently reviewed H/U scope labels',
        CPU_seconds=time.time()-started, GPU_seconds=0, new_generation=0, new_judgments=0,
        known_source_components=len(set(clusters.values())), source_reuse=source_counts,
        component_caveat='All standard-probe observed file/preprocessed-image/source-ref/PMC-article connections; excludes intentionally shared U-fit. Not complete patient or near-duplicate closure.',
        U_fit='One reused training QA; 200 edit contexts do not create 200 independent held-out U samples',
        margins_include_off=True, historical_primary='RC_FIXED_OLD16', new_calibration='NOT_PERFORMED'))
    def cell(method, metric, mode, score='exact', subtype='ALL'):
        return next(r for r in decomposition if (r['method'],r['metric'],r['mode'],r['score'],r['subtype']) == (method,metric,mode,score,subtype))
    lines = ['# Stage16 A — read-only routing/writer decomposition', '',
        f'COMPLETE: 400 historical writer checkpoints, {len(rows)} completed writer-probes, {binding["reused_judge_tuples"]} frozen Judge tuples. No training, generation or rescoring.', '',
        'Stage15 primary RC_FIXED_OLD16 and all original results remain unchanged. All new threshold curves are post-hoc diagnostics, not calibration or independent confirmation.', '',
        f'Old kappa={KAPPA}; accepted iff R0_ON and margin >= kappa, margin=(r-d)/max(r,1e-12). Equivalent positive-radius d/r <= {1-KAPPA}.', '',
        '| Writer | I-General R0 | I-General old RC | I-Local R0 preservation | I-Local old RC |',
        '|---|---:|---:|---:|---:|']
    for m in METHODS:
        cs = [cell(m, met, mode) for met,mode in [('I_Generality','BE_ROUTE_R0'),('I_Generality','RC_FIXED_OLD16'),('I_Locality','BE_ROUTE_R0'),('I_Locality','RC_FIXED_OLD16')]]
        lines.append('| '+m+' | '+' | '.join(f'{c["total_success"]}/{c["probes"]}' for c in cs)+' |')
    lines += ['', 'The old RC image-locality preservation is predominantly rejection-to-Base. It does not demonstrate a writer that preserves locality or a transferable cross-image acceptance region.', '',
        'R0 image-locality: 49 OFF in both methods; among 18 ON, C_NO_H preserves 4 and damages 14, BalancEdit preserves 0 and damages 18. Paired discordants 4 vs 0 yield supplemental two-sided exact p=0.125. The original edit-bootstrap interval is retained separately, not overwritten.', '',
        'PORTABILITY_BASE_DELTA.csv compares each hop and exact/semantic score against Base. Source semantic accuracy is not output locality; counterfactual target adherence is not clinical truth. Exact target-copy proportions use whole normalized answers only, not substring or semantic inference.', '',
        f'Known observed standard-probe source components: {len(set(clusters.values()))}. SOURCE_GROUP_SENSITIVITY.csv supplements the edit-based estimates; patient/study identity and unobserved near-duplicate relations remain UNKNOWN.', '',
        'ROUTER_PAIRWISE_AUROC.csv keeps native, text-generality and image-generality denominators separate and compares each with each official locality category. Official categories are scope proxies, not newly clinically reviewed H labels. No combined native+image-general coverage or H-fit-as-held-out claim.', '',
        f'All {len(ks)} attainable nonnegative inclusive/crossing breakpoints are reported. No threshold is adopted from the old 200 edits. Independent Base-only development support is required before RC_EXTCAL_V1.', '']
    (public/'STAGE16_A_READONLY_REPORT.md').write_text('\n'.join(lines))
    print('A_COMPLETE', json.dumps(dict(rows=len(rows), thresholds=len(ks), seconds=time.time()-started)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage15-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.stage15_root, args.run_root)
