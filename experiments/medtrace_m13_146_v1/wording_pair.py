"""Finite, source-preserving question-form diagnostic; no training entry."""
import collections
import json
from pathlib import Path

METHODS = ('BASE', 'M13', 'FISHER_MIN')
N = 78
CONSUMERS = 234
EPOCH = 'MEDTRACE_WORDING_PAIR_20261015_V1'
QUEUE = 'judge_wording_pair_astra_medium'
SUSPECT = (18, 33, 67)


def make_plan(original, pairs):
    assert [p['index'] for p in pairs] == list(range(1, N+1))
    assert len({p['original_query_id'] for p in pairs}) == N
    rows = []
    for pair in pairs:
        assert pair['semantic_pair_review'] == 'PASS_TASK_AND_QUALIFIERS_PRESERVED'
        assert pair['reference_scope_uncertain'] == (pair['index'] in SUSPECT)
        for method in METHODS:
            candidates = [r for r in original if r['method'] == method and
                          r['query']['query_id'] == pair['original_query_id'] and
                          (method == 'BASE' or r['owner'] == pair['owner'])]
            assert len(candidates) == 1
            old = candidates[0]
            assert old['query']['question'] == pair['original_question']
            assert pair['question'].strip() and pair['question'] != pair['original_question']
            query = dict(old['query'], question=pair['question'],
                         query_id=f"wording-pair-v1-{pair['index']:03d}")
            rows.append(dict(old, query=query))
    assert collections.Counter(r['method'] for r in rows) == dict.fromkeys(METHODS, N)
    return rows


def frozen(parent, read):
    for method in METHODS[1:]:
        for owner in range(1, 21):
            result = read(parent/f'private/results/{method}/{owner}.json')
            assert result['status'] == 'COMPLETE' and len(result['curve']) == 160
            assert result['save_load_exact']
            weight = parent/f'private/candidates/{method}/{owner}.pt'
            assert weight.is_file() and not weight.is_symlink()
    return True


def check(run):
    import common as c
    read = lambda p: json.loads(Path(p).read_text())
    gate = read(run/'private/WORDING_ADMISSION1.json')
    parent = Path(gate['parent'])
    assert gate['status'] == 'PASS' and gate['launch_authorized']
    assert gate['GPU'] == 2 and gate['generation_cap'] == CONSUMERS
    assert gate['max_GPU_seconds'] == 7200 and gate['training_updates'] == 0
    assert c.digest(read(run/'private/WORDING_PAIRS.json')) == gate['pairs_binding']
    assert c.digest(read(run/'private/WORDING_SCOPE1.json')) == gate['scope_binding']
    assert c.digest(read(run/'private/EVAL_PLAN146.json')) == gate['plan_binding']
    assert read(run/'private/WORDING_SOURCE1.json')['commit'] == gate['source_commit']
    # The inherited emitter reads these before the bank layer rewrites its binding.
    assert read(run/'private/SCOPE_LOCK.json') == read(run/'private/WORDING_SCOPE1.json')
    assert read(run/'private/GPU_SOURCE_VERSION.json') == read(run/'private/WORDING_SOURCE1.json')
    assert read(parent/'private/SELECTED_PREFILL_RECOVERY_SOURCE1.json')['commit'] == gate['parent_source']
    assert read(run/'private/EVAL_PLAN146.json') == make_plan(
        read(Path(gate['original'])/'private/EVAL_PLAN146.json'), read(run/'private/WORDING_PAIRS.json'))
    frozen(parent, read)
    return gate
