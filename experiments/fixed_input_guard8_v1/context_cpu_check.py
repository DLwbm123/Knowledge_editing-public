"""CPU regression of context strata, immutable anchors and bounded restore."""
import torch
from .adapter import context_positions
from .context_guard import InputGuard


def run():
    torch.manual_seed(71)
    labels = torch.full((1, 40), -100)
    labels[0, 30:36] = 1
    attention = torch.ones_like(labels); attention[0, 36:] = 0
    batch = ({'attention_mask': attention}, labels, (3, 23))
    positions = context_positions(batch, 32).tolist()
    pins = context_positions(batch, 4, anchors=True).tolist()
    assert len(positions) <= 32 and len(set(positions)) == len(positions)
    assert all(attention[0,p] for p in positions)
    assert any(3 <= p < 23 for p in positions) and any(p < 3 or 23 <= p < 29 for p in positions)
    assert 29 in positions and 34 in positions and 29 in pins and 34 in pins and len(pins) == 4
    assert positions == context_positions(batch, 32).tolist()
    d = 256
    g = InputGuard(torch.randn(32,d)); first = None
    for i in range(8):
        anchors = torch.randn(4,d); inputs = torch.randn(32,d)
        if first is None:first = anchors.clone()
        before = g.pinned.clone(); n = int(g.pin_count)
        g.admit(inputs, successful=True, anchors=anchors)
        assert torch.equal(before[:,:n],g.pinned[:,:n]), 'Pinned columns rotated'
        assert g.coverage(first)['max_uncovered_energy'] < 1e-10
        assert g.coverage(anchors)['max_uncovered_energy'] < 1e-10
    assert int(g.pin_count) == 32 and float(g.discarded) > 0
    restored = InputGuard.restore(g.state())
    assert torch.allclose(g.basis(),restored.basis(),atol=1e-12,rtol=0)
    bad = g.state(); bad['pin_count'].fill_(33)
    try:InputGuard.restore(bad)
    except ValueError:pass
    else:raise AssertionError('Corrupt pin count accepted')
    before = g.state()
    try:g.admit(torch.randn(32,d),successful=True,anchors=torch.randn(4,d))
    except ValueError:pass
    else:raise AssertionError('Capacity silently exceeded')
    assert all(torch.equal(v,g.state()[k]) for k,v in before.items())
    skipped = InputGuard(torch.randn(32,d)); skipped.admit(torch.randn(32,d),successful=False,anchors=torch.randn(4,d))
    assert int(skipped.pin_count) == int(skipped.pin_sources) == 0
    # Exercise the unchanged native repair at the new maximum combined rank 128.
    native = torch.randn(8,d,dtype=torch.float16)*.02
    certificate = g.write(native,torch.randn(8,d)*1e-4)
    # Small fixtures lack the 640 repair columns, so rejection is valid and atomic.
    assert certificate['accepted'] or certificate['reason'] == 'NATIVE_PROJECTION_REJECTED'
    full = InputGuard(torch.randn(32,14336))
    directions = torch.linalg.qr(torch.randn(14336,128,dtype=torch.float64),mode='reduced').Q.float()
    full.base.copy_(directions[:,:32]); full.pinned.copy_(directions[:,32:64]); full.history.copy_(directions[:,64:])
    full.counts[:] = torch.tensor([32,64,0,8]); full.pin_count.fill_(32); full.pin_sources.fill_(8)
    for dtype in (torch.float16,torch.bfloat16):
        native = (torch.randn(8,14336)*.02).to(dtype); before = native.clone()
        proposal = torch.randn(8,14336)*.001
        certificate = full.write(native,proposal)
        assert certificate['accepted'], certificate
        actual = native.double()-before.double()
        assert float((actual @ full.basis()).norm()) <= certificate['limit']
        assert certificate['relative_direction_distortion'] <= .05
        assert not torch.equal(native,before)
    assert InputGuard.restore(full.state()).check_storage() == full.check_storage()
    assert full.check_storage() == 14680120 and full.check_storage() < 16*1024**2
    return dict(status='PASS',persistent_bytes=full.check_storage(),pin_capacity=32,
                covered_strata=['visual','prompt','answer_predictors','final_target_predictor'],
                compression_retains_first_native_anchors=True, corrupt_restore_rejected=True)
