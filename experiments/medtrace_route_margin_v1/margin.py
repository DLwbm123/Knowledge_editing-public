"""Minimum modal-score advantage before switching a frozen R0 expert.

The 0.02 default was chosen from exposed development failure diagnostics.
It is not independently calibrated confidence or a scope-protection gate.
"""
import math

PRIMARY_MARGIN = 0.02
SENSITIVITY_MARGINS = (0.01, 0.03)


def select(original, candidate, scores, margin=PRIMARY_MARGIN):
    """Indices come from R0 and its radius-restricted modal reranker."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('The minimum score advantage must be finite and nonnegative')
    if original is None:
        if candidate is not None:
            raise ValueError('The candidate must preserve original R0 OFF')
        return None
    if candidate is None or not (0 <= original < len(scores) and 0 <= candidate < len(scores)):
        raise ValueError('Both accepted expert indices must exist')
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Modal scores must be finite')
    return candidate if scores[candidate] - scores[original] >= margin else original


def selfcheck():
    assert select(None, None, None) is None
    assert select(0, 0, [0.8]) == 0
    assert select(0, 1, [0.8, 0.819]) == 0
    assert select(0, 1, [0.8, 0.821]) == 1
    assert select(0, 1, [0., 0.02]) == 1
    assert select(0, 1, [0.8, 0.7]) == 0
    for args in [(None, 0, [0.8]), (0, None, [0.8]), (0, 1, [0.8]),
                 (0, 1, [0.8, float('nan')]), (0, 0, [0.8], -1)]:
        try:
            select(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(args)
    return {'status': 'PASS', 'checks': 11}


if __name__ == '__main__':
    print(selfcheck())
