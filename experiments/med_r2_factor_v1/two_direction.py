"""Experimental two-direction operator; caller supplies the original V133 Gram."""
import math
import torch


def build(primary, secondary, trace, gram_fn, energy_fn):
    p = {k: primary[k].double() for k in ('x', 'y')}
    s = {k: secondary[k].double() for k in ('x', 'y')}
    if (p['x'].ndim != 2 or s['x'].ndim != 2 or p['y'].ndim != 3
            or s['y'].ndim != 3 or p['y'].shape[1] != 1 or s['y'].shape[1] != 1
            or p['x'].shape[0] != p['y'].shape[0] or s['x'].shape[0] != s['y'].shape[0]
            or p['x'].shape[1] != s['x'].shape[1] or p['y'].shape[2] != s['y'].shape[2]):
        raise ValueError('incompatible response factors')
    pe, se = energy_fn(p), energy_fn(s)
    if not all(math.isfinite(v) and v > 0 for v in (pe, se, trace)):
        raise ValueError('zero or nonfinite energy')
    coef = float(gram_fn(p, s)[0, 0]) / pe
    x = torch.cat((p['x'], s['x']))
    y0 = torch.cat((p['y'], torch.zeros_like(s['y'])))
    y1 = torch.cat((-coef * p['y'], s['y']))
    perp = dict(x=x, y=y1)
    qe = energy_fn(perp)
    if not math.isfinite(qe) or qe <= 1e-12 * se:
        raise ValueError('degenerate second direction')
    return dict(x=x, y=torch.cat((y0 * math.sqrt(trace / (2 * pe)),
                                  y1 * math.sqrt(trace / (2 * qe))), dim=1),
                token_counts=[len(x)])
