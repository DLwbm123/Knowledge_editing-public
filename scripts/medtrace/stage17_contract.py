"""Stage17-A contracts only: no data discovery, training, or Judge invocation."""
from contextlib import contextmanager
from statistics import mean


def require_binding(actual, expected):
    required = {'input', 'runtime', 'generation', 'writer', 'prefix', 'training'}
    if not required <= actual.keys() or not required <= expected.keys() or actual != expected:
        raise ValueError('incomplete or mismatched realized-input/state binding')


@contextmanager
def base_only(hooks):
    """Routing must enter outside a generation request and with every hook OFF."""
    if any(h.generation_routing for h in hooks):
        raise RuntimeError('cannot route inside an active generation request')
    for hook in hooks:
        hook.clear_request_routing()
    try:
        yield
    finally:
        for hook in hooks:
            hook.clear_request_routing()


def prefix_router(entries, prefix, device='cpu'):
    from m3bench_repro.editors.routing import MemoryRouter
    if type(prefix) is not int or not 0 <= prefix <= len(entries):
        raise ValueError('invalid prefix')
    selected = [dict(e, label=[]) for e in entries[:prefix]]
    return MemoryRouter.from_state({'distance': 'euclidean', 'entries': selected}, device=device)


def support_masks(ledger):
    """Tri-state proofs: unknown is not zero legal support and never executable."""
    masks = {name: {} for name in ('E_U', 'E_H', 'E_HG', 'E_HG_eval')}
    requirements = {'E_U': ('U_fit',), 'E_H': ('U_fit', 'H_fit'),
                    'E_HG': ('U_fit', 'H_fit', 'G'),
                    'E_HG_eval': ('U_fit', 'H_fit', 'G', 'H_eval')}
    for row in ledger:
        edit = row['edit_id']
        if edit in masks['E_U']:
            raise ValueError('duplicate ledger occurrence; freeze conflict policy first')
        for panel, roles in requirements.items():
            values = [row['roles'].get(role) for role in roles]
            if any(v is not True and v is not False and v is not None for v in values):
                raise ValueError('role proof must be true, false, or unknown')
            masks[panel][edit] = False if False in values else None if None in values else True
    return masks


def correctness(rows, metric):
    """One task/common support, frozen Base mask; empty denominators stay NA."""
    if metric not in {'retention', 'fix', 'postacc', 'paircorrect'}:
        raise ValueError('unknown metric')
    grouped = {}
    for row in rows:
        c0, ct = row['base_correct'], row['post_correct']
        if type(c0) is not bool or type(ct) is not bool:
            raise ValueError('unjudged or failed output is not a semantic Boolean')
        if metric == 'retention' and not c0 or metric == 'fix' and c0:
            continue
        if metric == 'paircorrect':
            if type(row.get('native_correct')) is not bool:
                raise ValueError('PairCorrect requires native correctness including failures')
            ct = ct and row['native_correct']
        grouped.setdefault(row['edit_id'], []).append(int(ct))
    values = [v for group in grouped.values() for v in group]
    return {'edit_macro': mean(map(mean, grouped.values())) if grouped else None,
            'probe_micro': mean(values) if values else None,
            'supported_edits': len(grouped), 'supported_probes': len(values),
            'correct_probes': sum(values)}
