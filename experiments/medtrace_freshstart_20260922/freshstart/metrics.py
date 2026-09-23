#!/usr/bin/env python3
"""Offline metrics for normalized, deidentified fixed-prefix scores.

No model, GPU, network, or Judge access. This is not a production worker.
Missing labels are never treated as wrong. Input contract: see 04_METRICS...md.
"""
from __future__ import annotations
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

BOOL_FIELDS = ('current_correct', 'base_correct', 'strict_unrelated', 'teacher_agreement')


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'Invalid JSON at line {line_no}: {error.msg}') from error
                rows.append(row)
    return validate_rows(rows)


def validate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result, seen, bindings = [], set(), {}
    for i, value in enumerate(rows, 1):
        if not isinstance(value, dict):
            raise ValueError(f'Row {i} is not an object')
        row = dict(value)
        for key in ('arm', 'input_id', 'source_group', 'role'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'Row {i}: {key} must be a nonempty string')
        if type(row.get('prefix')) is not int or row['prefix'] < 1:
            raise ValueError(f'Row {i}: prefix must be a positive integer')
        for key in BOOL_FIELDS:
            row.setdefault(key, None)
            if row[key] is not None and type(row[key]) is not bool:
                raise ValueError(f'Row {i}: {key} must be bool or null, not an integer')
        identity = (row['arm'], row['prefix'], row['input_id'])
        if identity in seen:
            raise ValueError(f'Duplicate fixed-prefix consumer: {identity}')
        seen.add(identity)
        common = (row['prefix'], row['input_id'])
        fixed = (row['role'], row['source_group'])
        old = bindings.setdefault(common, {'fixed': fixed, 'base': None, 'strict': None})
        if old['fixed'] != fixed:
            raise ValueError(f'Cross-arm role/source disagreement: {common}')
        for field, entry in (('base_correct', 'base'), ('strict_unrelated', 'strict')):
            known = row[field]
            if known is not None:
                if old[entry] is not None and old[entry] != known:
                    raise ValueError(f'Cross-arm {field} disagreement: {common}')
                old[entry] = known
        result.append(row)
    return result


def outcome(values: Iterable[bool | None]) -> dict[str, Any]:
    vals = list(values)
    if any(x is not None and type(x) is not bool for x in vals):
        raise ValueError('outcome accepts bool/null only')
    n = len(vals)
    missing = sum(x is None for x in vals)
    correct = sum(x is True for x in vals)
    return {
        'denominator': n, 'scored': n - missing, 'known_correct': correct,
        'missing': missing,
        'rate': correct / n if n and not missing else None,
        'lower_bound': correct / n if n else None,
        'upper_bound': (correct + missing) / n if n else None,
        'status': 'EMPTY_DENOMINATOR' if not n else ('PARTIAL' if missing else 'COMPLETE'),
    }


def conditional(rows: list[dict[str, Any]], base_value: bool, *, strict: bool = False) -> dict[str, Any]:
    unknown_eligibility = 0
    selected = []
    for row in rows:
        if strict:
            if row['strict_unrelated'] is None:
                unknown_eligibility += 1
                continue
            if row['strict_unrelated'] is False:
                continue
        if row['base_correct'] is None:
            unknown_eligibility += 1
        elif row['base_correct'] is base_value:
            selected.append(row['current_correct'])
    result = outcome(selected)
    result['unknown_eligibility'] = unknown_eligibility
    if unknown_eligibility:
        result['known_eligible_count'] = result.pop('denominator')
        result.update(denominator=None, rate=None, lower_bound=None, upper_bound=None,
                      status='UNKNOWN_ELIGIBILITY')
    return result


def source_macro(rows: list[dict[str, Any]], which: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row['source_group']].append(row)
    per_group = {}
    for source, rr in groups.items():
        if which == 'accuracy':
            score = outcome(r['current_correct'] for r in rr)
        else:
            score = conditional(rr, True, strict=(which == 'strict_retention'))
        per_group[source] = score
    eligible = [s for s in per_group.values() if s['status'] != 'EMPTY_DENOMINATOR']
    complete = [s['rate'] for s in eligible if s['rate'] is not None]
    fully_covered = len(complete) == len(eligible)
    return {
        'all_sources': len(groups), 'eligible_or_unknown_sources': len(eligible),
        'complete_sources': len(complete),
        'rate': sum(complete) / len(complete) if complete and fully_covered else None,
        'diagnostic_complete_groups_only_mean': sum(complete) / len(complete) if complete else None,
        'all_eligible_sources_complete': fully_covered, 'per_source': per_group,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = validate_rows(rows)
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row['arm'], row['prefix'], row['role'])].append(row)
    summaries = []
    for (arm, prefix, role), rr in sorted(groups.items()):
        summaries.append({
            'arm': arm, 'prefix': prefix, 'role': role, 'input_count': len(rr),
            'accuracy': outcome(r['current_correct'] for r in rr),
            'base_accuracy': outcome(r['base_correct'] for r in rr),
            'conditional_retention': conditional(rr, True),
            'conditional_fix': conditional(rr, False),
            'strict_unrelated_retention': conditional(rr, True, strict=True),
            'teacher_agreement': outcome(r['teacher_agreement'] for r in rr),
            'source_macro_accuracy': source_macro(rr, 'accuracy'),
            'source_macro_conditional_retention': source_macro(rr, 'retention'),
            'source_macro_strict_retention': source_macro(rr, 'strict_retention'),
        })
    pairs = []
    arms = sorted({r['arm'] for r in rows})
    for a, b in itertools.combinations(arms, 2):
        for prefix in sorted({r['prefix'] for r in rows}):
            roles = sorted({r['role'] for r in rows if r['prefix'] == prefix})
            for role in roles:
                index = {arm: {r['input_id']: r for r in rows if r['arm'] == arm
                              and r['prefix'] == prefix and r['role'] == role} for arm in (a, b)}
                keys = sorted(set(index[a]) | set(index[b]))
                if not keys:
                    continue
                counts = dict(both_right=0, both_wrong=0, wrong_to_right=0,
                              right_to_wrong=0, unscored_or_absent=0)
                for key in keys:
                    ra, rb = index[a].get(key), index[b].get(key)
                    if ra is None or rb is None or ra['current_correct'] is None or rb['current_correct'] is None:
                        counts['unscored_or_absent'] += 1
                        continue
                    x, y = ra['current_correct'], rb['current_correct']
                    term = ('both_right' if x and y else 'both_wrong' if not x and not y
                            else 'wrong_to_right' if not x and y else 'right_to_wrong')
                    counts[term] += 1
                pairs.append({'comparison': f'{b}_vs_{a}', 'direction': f'{a} to {b}',
                              'prefix': prefix, 'role': role, 'union_inputs': len(keys), **counts})
    return {'schema': 'fresh45_metrics_v1', 'rows': len(rows), 'summary': summaries,
            'pairwise': pairs, 'clinical_validation': False,
            'note': 'Fixed-prefix inputs only. Missing and empty-denominator rates remain null.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Normalized deidentified fixed-prefix scores JSONL')
    parser.add_argument('--output', type=Path, help='Write a new JSON result; existing file is not overwritten')
    args = parser.parse_args()
    try:
        result = summarize(load_rows(args.input))
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(text)
        else:
            print(text, end='')
    except (ValueError, OSError) as error:
        parser.exit(2, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
