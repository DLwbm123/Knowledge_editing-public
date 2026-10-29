"""Use the principal original FULL_CTX response orthogonal to MED."""
import math
import torch


def projected_direction(full, primary, trace, gram, energy, build):
    if len(full) != 4 or any(b['y'].shape[1] != 8 for b in full):
        raise ValueError('expected four eight-probe paths')
    pe = energy(primary)
    if not math.isfinite(pe) or pe <= 0 or not math.isfinite(trace) or trace <= 0:
        raise ValueError('invalid primary or trace')
    matrix = torch.cat([torch.cat([gram(a, b) for b in full], dim=1) for a in full], dim=0)
    cross = torch.cat([gram(b, primary)[:, 0] for b in full])
    projected = matrix - torch.outer(cross, cross) / pe
    projected = (projected + projected.T) / 2
    if not torch.isfinite(projected).all():
        raise ValueError('nonfinite projected Gram')
    eigenvalues, eigenvectors = torch.linalg.eigh(projected)
    reference = float(torch.trace(matrix))
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError('invalid probe energy')
    if float(eigenvalues[0]) < -1e-7 * reference:
        raise ValueError('projected Gram is not positive semidefinite')
    top = float(eigenvalues[-1])
    if top <= 1e-10 * reference:
        return None, 'PROJECTED_DEGENERATE'
    if top - float(eigenvalues[-2]) <= 1e-8 * top:
        return None, 'PROJECTED_TOP_TIE'
    v = eigenvectors[:, -1]
    x = torch.cat([b['x'].double() for b in full])
    y = torch.cat([torch.einsum('tso,s->to', b['y'].double(), v[8*i:8*(i+1)])[:, None]
                   for i, b in enumerate(full)])
    result = build(primary, dict(x=x, y=y), trace, gram, energy)
    return result, 'OK'


def runtime_fp32(block, trace, gram):
    """Make the V133 penalty-compatible cache and enforce trace parity."""
    converted = dict(x=block['x'].float(), y=block['y'].float(),
                     token_counts=list(block['token_counts']))
    observed = gram(converted, converted)
    expected = torch.eye(2, dtype=torch.float64, device=observed.device) * (trace / 2)
    if not torch.isfinite(observed).all():
        raise ValueError('nonfinite float32 curvature')
    torch.testing.assert_close(observed, expected, rtol=3e-5, atol=3e-5 * trace)
    return converted


def pack_mapped(blocks):
    """Retain independent source terms while padding fallback to two columns."""
    xs, ys, counts = [], [], []
    for block in blocks:
        x, y = block['x'].float(), block['y'].float()
        if x.ndim != 2 or y.ndim != 3 or x.shape[0] != y.shape[0] or y.shape[1] not in (1, 2):
            raise ValueError('invalid mapped block')
        if y.shape[1] == 1:
            y = torch.cat((y, torch.zeros_like(y)), dim=1)
        xs.append(x)
        ys.append(y)
        counts.append(len(x))
    if not counts:
        raise ValueError('no mapped blocks')
    return dict(x=torch.cat(xs), y=torch.cat(ys), token_counts=counts)


def mapped_block(full, med, audit, gram, energy, build):
    """Transform one V133 source and expose the predeclared fallback reason."""
    result, reason = projected_direction(full, med, audit['full_trace'], gram, energy, build)
    if result is None:
        return dict(x=med['x'].float(), y=med['y'].float()), reason
    return runtime_fp32(result, audit['full_trace'], gram), reason
