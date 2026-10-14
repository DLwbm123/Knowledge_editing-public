"""Reuse the current-case executor; replace only curvature and rejection diagnostics."""
import os
import math
import time
import traceback
import torch
import torch.nn.functional as F
import base_learner as old
from geometry import Risk, native_candidate, probe_rows


def optimize(runtime, weight, master, risk, full, removed, anchors, slot):
    assert not removed and anchors is None
    gradient = torch.zeros_like(master); losses = []
    for index in (0, 1+(slot-1)%4):
        z, y = old.b.logits(runtime, full[index]); loss = .5 * F.cross_entropy(z, y)
        losses.append(float(loss.detach())); gradient += old.backward(loss, weight)
    norm = float(gradient.norm()); assert math.isfinite(norm) and torch.isfinite(gradient).all()
    gradient *= min(1., 5/max(norm, 1e-30))
    master, native, event = native_candidate(master, weight, gradient, risk)
    if event['accepted']:
        with torch.no_grad(): weight.copy_(native)
        old.b.COUNTS['committed_updates'] += 1; old.b.call('C')
    event.update(source_CE=sum(losses), native_CE=2*losses[0], FIT_CE=2*losses[1],
        medical_scope_loss=0., unclipped_gradient_norm=norm, risk_compression_envelope=risk.eq)
    old.b.counts(); return master, event


def validate_state(state):
    assert 1 <= state['owner'] <= 3 and 1 <= state['slot'] <= 160
    assert state['completed_slots'] == (state['owner']-1)*160+state['slot']
    assert state['clocks'] == dict(FIT=(state['slot']-1)%4, answer_probe=20260912+state['owner'], evidence_probe=20260912+100000+state['owner'])
    assert state['native'].dtype == torch.float16 and state['master'].dtype == torch.float32
    assert torch.equal(state['native'], state['master'].half())
    assert state['current_anchors'] is None
    assert not state['boundary'] or state['slot'] == 160
    assert state['risk']['q'] <= .01 and state['risk']['rho'] >= 0
    assert len(state['risk']['B']) <= (12 if state['arm'] == 'S' else 4)
    assert state['risk']['B'].shape[1]==state['native'].numel()
    assert state['risk']['v'].numel()==state['risk']['ev'].numel()==state['native'].numel()
    if state['arm'] == 'S': assert state['risk']['rho'] == 0
    return state


old.Risk = Risk; old.native_candidate = native_candidate; old.probe_rows = probe_rows
old.optimize = optimize; old.validate_state = validate_state
main = old.main


def admit():
    from access import install
    with old.lease():
        runtime=old.load_learner();hooks=old.track(runtime)
        try:
            install(old.R,old.WINDOW,os.environ['DATA_ROOT']+'/knowledge_editing/data/m3bench')
            packet=old.wait(old.WINDOW/'current.json');assert packet['task']['order']==1 and not packet['gate']
            full,removed=old.batches(runtime,packet);weight=runtime.get_module(old.c.LAYER).weight
            frozen={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            old.admission(runtime,weight,full,removed,old.b.schema(runtime.model),frozen)
            old.peak_memory()
            old.c.write(old.CONTROL/'LEARNER_ADMITTED.json',dict(status='PASS',formal_slots=0))
        finally:hooks.close();old.b.counts()


if __name__ == '__main__':
    try:
        (admit if os.environ['ACTION'].startswith('admit_') else main)()
        old.c.write(old.R/'private'/('EXIT_'+os.environ['ACTION']+'.json'), dict(exit_code=0, epoch=time.time()))
    except BaseException:
        old.b.counts(); old.c.write(old.R/'private/failures'/(os.environ['ACTION']+'_'+str(os.getpid())+'.json'),
            dict(traceback=traceback.format_exc(), epoch=time.time(), counts=old.b.COUNTS)); raise
