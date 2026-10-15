"""Read-only aggregation of frozen scores and separate operator diagnostic labels.

Inputs contain private questions/answers; only anonymous aggregates are exported.
No model, scorer, network, queue write, or training dependency is imported.
"""
import argparse
import collections
import json
from pathlib import Path

METHODS = ('M13', 'FISHER_MIN')
STATES = ('00', '01', '10', '11')
LABELS = {
    'N': 'Non-answer / generic deferral rather than the requested target',
    'C': 'Concrete target/entity/content substitution relative to frozen reference',
    'L': 'Explicit side or comparative-relation mismatch',
    'O': 'Missing target component or required qualifier',
    'E': 'Target present with reference-external additions',
    'D': 'Degenerate repeated output, with missing target components',
}


def analyze(rows, reviews):
    assert len(rows) == len(reviews) == 78
    assert [r['index'] for r in rows] == list(range(1, 79))
    assert len({r['owner'] for r in rows}) == 20
    states = {m: {} for m in METHODS}
    categories = {m: {p: collections.Counter() for p in ('old', 'new')} for m in METHODS}
    errors_by_transition = {m: {s: {p: collections.Counter() for p in ('old', 'new')} for s in STATES} for m in METHODS}
    owners = sorted({r['owner'] for r in rows})
    owner_rows = []
    for row, review in zip(rows, reviews):
        assert (row['index'], row['owner']) == (review['index'], review['owner'])
        assert review['reviewed_all_four_answers']
        for m in METHODS:
            values = row['methods'][m]
            assert all(values[p]['correct'] in (0, 1) for p in ('old', 'new'))
            state = ''.join(str(values[p]['correct']) for p in ('old', 'new'))
            states[m][row['index']] = state
            for p in ('old', 'new'):
                label = review['labels'][m][p]
                if values[p]['correct']:
                    assert label == 'SCORE_CORRECT_NOT_RELABELLED'
                else:
                    assert label in LABELS
                    categories[m][p][label] += 1
                    errors_by_transition[m][state][p][label] += 1
    for owner in owners:
        ids = [r['index'] for r in rows if r['owner'] == owner]
        owner_rows.append(dict(owner=owner, probes=len(ids), **{
            m: dict(collections.Counter(states[m][i] for i in ids)) for m in METHODS}))
    a, b = (states[m] for m in METHODS)
    cross = {s: {t: sum(a[i] == s and b[i] == t for i in a) for t in STATES} for s in STATES}
    overlaps = {}
    for state in STATES:
        left = {i for i in a if a[i] == state}; right = {i for i in b if b[i] == state}
        overlaps[state] = dict(M13=len(left), FISHER_MIN=len(right), intersection=len(left & right),
                               M13_only=len(left-right), FISHER_MIN_only=len(right-left),
                               union=len(left | right), Jaccard=len(left & right)/len(left | right))
    concentration = {}
    for m in METHODS:
        counts = [r[m].get('01', 0)+r[m].get('10', 0) for r in owner_rows]
        repairs = [r[m].get('01', 0) for r in owner_rows]
        regressions = [r[m].get('10', 0) for r in owner_rows]
        concentration[m] = dict(total_flips=sum(counts), affected_edits=sum(v > 0 for v in counts),
            flip_count_histogram=dict(collections.Counter(counts)), max_flips_per_edit=max(counts),
            top3_flip_count=sum(sorted(counts, reverse=True)[:3]), top5_flip_count=sum(sorted(counts, reverse=True)[:5]),
            edits_with_repairs=sum(v > 0 for v in repairs), edits_with_regressions=sum(v > 0 for v in regressions),
            edits_with_both=sum(x > 0 and y > 0 for x, y in zip(repairs, regressions)))
    # Histograms retain concentration evidence without exporting owner identities.
    stable_advantage = [r['FISHER_MIN'].get('11', 0)-r['M13'].get('11', 0) for r in owner_rows]
    unique_stable_errors = collections.Counter()
    for row, review in zip(rows, reviews):
        i = row['index']
        if b[i] == '11' and a[i] != '11':
            labels = {review['labels']['M13'][p] for p in ('old', 'new') if not row['methods']['M13'][p]['correct']}
            unique_stable_errors['+'.join(sorted(labels))] += 1
    result = dict(status='COMPLETE_READONLY_POSTHOC_ANALYSIS', probes=78, edits=20, answers_reviewed=312,
        incorrect_answer_instances_labelled=sum(sum(v.values()) for d in categories.values() for v in d.values()),
        state_definition={'00': 'both_wrong', '01': 'repaired', '10': 'regressed', '11': 'both_correct'},
        cross_matrix_rows_M13_columns_FISHER_MIN=cross, overlaps=overlaps, concentration=concentration,
        same_direction_flip_probes=cross['01']['01']+cross['10']['10'],
        opposite_direction_flip_probes=cross['01']['10']+cross['10']['01'],
        stable_correct_advantage_per_edit_histogram=dict(collections.Counter(stable_advantage)),
        FISHER_MIN_unique_stable_M13_error_label_sets=dict(unique_stable_errors),
        diagnostic_label_definitions=LABELS, diagnostic_counts=categories, error_labels_by_transition=errors_by_transition,
        reference_scope_uncertain_probes=sum(r['reference_scope_uncertain'] for r in reviews),
        rubric_boundary_review_probes=sum(r['rubric_boundary_review'] for r in reviews),
        score_changes=0, new_training=0, new_generation=0, new_Judge=0,
        limitation='Single exposed operator; labels describe reference-relative output failures, not clinical truth or causal mechanisms; original scores unchanged.')
    assert sum(sum(v.values()) for v in cross.values()) == 78
    assert overlaps['01']['M13'] == overlaps['01']['FISHER_MIN'] == 16
    assert overlaps['10']['M13'] == 12 and overlaps['10']['FISHER_MIN'] == 10
    assert result['incorrect_answer_instances_labelled'] == 106
    return result, owner_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('private_input', type=Path)
    parser.add_argument('private_review', type=Path)
    parser.add_argument('public_output', type=Path)
    parser.add_argument('private_owner_output', type=Path)
    args = parser.parse_args()
    result, owners = analyze(json.loads(args.private_input.read_text()), json.loads(args.private_review.read_text()))
    args.public_output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    args.private_owner_output.write_text(json.dumps(owners, ensure_ascii=False, indent=2)+'\n')
