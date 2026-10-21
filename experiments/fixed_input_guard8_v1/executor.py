"""Transactional native edit executor with bounded state and atomic checkpoints."""
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from experiments.fixed_input_guard_v1.guard import ATOL, RTOL, MAX_RELATIVE_DISTORTION
from .context_guard import InputGuard
from .configuration import VERSION, ARMS, SLOTS, OPTIMIZER, CAPS, CHECKPOINT_BYTES, WALL_SECONDS
from .adapter import RuntimeAdapter


def rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=(n[0], torch.from_numpy(n[1].astype('int64')), *n[2:]),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    n = state['numpy']
    np.random.set_state((n[0], n[1].numpy().astype('uint32'), *n[2:]))
    if state['cuda']:
        if not torch.cuda.is_initialized():
            raise ValueError('Cannot restore CUDA RNG without the matching runtime')
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Refuse symlink output')
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class Budget:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = dict(start_epoch=time.time(), counts={k: 0 for k in CAPS})
            atomic_json(self.path, self.data)
        self.check()

    def check(self):
        if time.time()-self.data['start_epoch'] >= WALL_SECONDS:
            raise TimeoutError('Wall budget exhausted')
        if set(self.data['counts']) != set(CAPS):
            raise ValueError('Budget schema mismatch')
        for key, cap in CAPS.items():
            if type(self.data['counts'][key]) is not int or not 0 <= self.data['counts'][key] <= cap:
                raise ValueError('Invalid counter: '+key)

    def consume(self, key):
        self.check()
        if self.data['counts'][key] >= CAPS[key]:
            raise TimeoutError('Budget exhausted: '+key)
        self.data['counts'][key] += 1
        atomic_json(self.path, self.data)


class Editor:
    def __init__(self, runtime, arm, calibration_inputs, budget):
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        self.arm, self.budget = arm, budget
        self.adapter = RuntimeAdapter(runtime, budget)
        self.weight = self.adapter.weight
        self.base_native = self.weight.detach().cpu().clone()
        self.master = torch.nn.Parameter(self.base_native.float().clone())
        self.optimizer = torch.optim.Adam([self.master], **OPTIMIZER)
        inputs = calibration_inputs if arm == 'P' else torch.zeros_like(calibration_inputs)
        self.guard = InputGuard(inputs)
        self.owner = self.slot = 0
        self.boundary = True
        self.source_id = None
        self.anchors = None
        self.views = None
        self.source_binding = None
        self.source_labels = None

    @torch.no_grad()
    def prepare_source(self, record, fits, owner):
        if not self.boundary or owner != self.owner+1 or not 1 <= owner <= 8:
            raise ValueError('Source order/retirement violation')
        current = self.weight.detach().clone()
        rng = rng_state()
        try:
            self.weight.copy_(self.base_native.to(self.weight.device))
            self.views = self.adapter.views(record, fits)
            self.anchors = self.adapter.teacher(self.views)
        finally:
            self.weight.copy_(current)
            restore_rng(rng)
        self.owner, self.slot, self.boundary = owner, 0, False
        self.source_id = record.record_id
        self.source_binding = (tuple(sorted((k, str(v)) for k, v in asdict(record).items())), tuple(fits))
        self.source_labels = [v[1].clone() for v in self.views]

    def attach_source(self, record, fits):
        if self.boundary or record.record_id != self.source_id:
            raise ValueError('Wrong source for checkpoint continuation')
        binding = (tuple(sorted((k, str(v)) for k, v in asdict(record).items())), tuple(fits))
        if binding != self.source_binding:
            raise ValueError('Source binding changed')
        rng = rng_state()
        try:
            self.views = self.adapter.views(record, fits)
        finally:
            restore_rng(rng)
        if len(self.views) != len(self.anchors) or any(int((v[1][:, 1:] != -100).sum()) != len(a) for v, a in zip(self.views, self.anchors)):
            raise ValueError('Source token layout changed')
        if any(not torch.equal(v[1], y) for v, y in zip(self.views, self.source_labels)):
            raise ValueError('Source token identities changed')

    def _rollback(self, native, master, optimizer, rng):
        with torch.no_grad():
            self.weight.copy_(native.to(self.weight.device))
            self.master.copy_(master)
        self.optimizer.load_state_dict(optimizer)
        self.master.grad = None
        restore_rng(rng)

    def attempt(self, slot, *, admission=False):
        if self.views is None or self.boundary or slot != self.slot+1 or not 1 <= slot <= SLOTS:
            raise ValueError('Invalid slot/source')
        self.budget.consume('admission_slots' if admission else 'attempt_slots')
        native = self.weight.detach().cpu().clone()
        master = self.master.detach().clone()
        optimizer = copy.deepcopy(self.optimizer.state_dict())
        rng = rng_state()
        event = dict(owner=self.owner, slot=slot, accepted=False)
        try:
            baseline = self.adapter.objective(self.views, self.anchors, slot)
            if not torch.isfinite(baseline):
                raise ValueError('Nonfinite source loss')
            self.budget.consume('backwards')
            gradient = torch.autograd.grad(128*baseline, self.weight)[0].float().cpu()/128
            if not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite source gradient')
            gradient *= min(1., 5/max(float(gradient.norm()), 1e-30))
            self.master.grad = gradient
            self.optimizer.step()
            proposal = self.master.detach()-master
            self.budget.consume('candidate_checks')
            candidate = native.clone()
            certificate = self.guard.write(candidate, proposal)
            event.update(certificate=certificate, source_loss_before=float(baseline.detach()),
                         proposal_norm=float(proposal.double().norm()))
            if not certificate['accepted']:
                event['reason'] = certificate['reason']
            else:
                with torch.no_grad():
                    self.weight.copy_(candidate.to(self.weight.device))
                actual = self.weight.detach().cpu().double()-native.double()
                q = self.guard.basis()
                projected = proposal.double()-(proposal.double() @ q) @ q.T
                norm = float(actual.norm())
                residual = float((actual @ q).norm())
                distortion = float((actual-projected).norm()/projected.norm()) if projected.norm() else math.inf
                event['device_readback'] = dict(norm=norm, residual=residual, limit=ATOL+RTOL*norm, distortion=distortion,
                                              projected_norm=float(projected.norm()), basis_rank=q.shape[1])
                if not (norm > 0 and math.isfinite(residual) and residual <= ATOL+RTOL*norm and distortion <= MAX_RELATIVE_DISTORTION):
                    event['reason'] = 'DEVICE_READBACK_REJECTED'
                else:
                    diagnostic_rng = rng_state()
                    try:
                        with torch.no_grad():
                            candidate_loss = float(self.adapter.objective(self.views, self.anchors, slot))
                    finally:
                        restore_rng(diagnostic_rng)
                    slope = float((gradient.double()*actual).sum())
                    event.update(source_loss_after=candidate_loss, slope=slope)
                    if not math.isfinite(candidate_loss) or slope >= 0 or candidate_loss > event['source_loss_before']+1e-4*slope:
                        event['reason'] = 'ACTUAL_SOURCE_ARMIJO_REJECTED'
                    else:
                        self.budget.consume('committed_updates')
                        with torch.no_grad():
                            self.master.copy_(self.weight.detach().cpu().float())
                        event.update(accepted=True, reason='ACCEPTED')
                        if admission:
                            path = self.budget.path.parent/'admission-native.pt'
                            with path.open('xb') as handle:
                                torch.save(self.weight.detach().cpu(), handle)
                            reloaded = torch.load(path, map_location='cpu', weights_only=True)
                            if reloaded.dtype != self.weight.dtype or not torch.equal(reloaded, self.weight.detach().cpu()):
                                raise ValueError('Admission native serialization mismatch')
                            with torch.no_grad():
                                self.weight.copy_(native.to(self.weight.device))
                                self.weight.copy_(reloaded.to(self.weight.device))
                                loaded_loss = float(self.adapter.objective(self.views, self.anchors, slot))
                            if loaded_loss != candidate_loss:
                                raise ValueError('Admission native reload changed source logits/loss')
                            event['native_roundtrip'] = 'PASS'
                            path.unlink()
            if not event['accepted'] or admission:
                self._rollback(native, master, optimizer, rng)
            self.master.grad = None
            if not admission:
                self.slot = slot
            return event
        except BaseException:
            self._rollback(native, master, optimizer, rng)
            raise

    @torch.no_grad()
    def finish_source(self):
        if self.boundary or self.slot != SLOTS or self.views is None:
            raise ValueError('Incomplete source')
        rng = rng_state()
        try:
            ce = self.adapter.measure(self.views)
            inputs = self.adapter.inputs(self.views)
            anchors = self.adapter.inputs(self.views, anchors=True)
        finally:
            restore_rng(rng)
        qualified = math.isfinite(ce) and ce < .1
        old_rows = self.guard.sketch[:int(self.guard.counts[2])].clone()
        old_pins = self.guard.pinned[:, :int(self.guard.pin_count)].T.clone()
        coverage = dict(context_before=self.guard.coverage(inputs), native_before=self.guard.coverage(anchors))
        for name, rows in (('retained_sketch', old_rows), ('previous_native_anchors', old_pins)):
            coverage[name+'_before'] = self.guard.coverage(rows) if len(rows) else None
        if self.arm == 'P':
            self.guard.admit(inputs, successful=qualified, anchors=anchors)
        coverage.update(context_after=self.guard.coverage(inputs), native_after=self.guard.coverage(anchors))
        for name, rows in (('retained_sketch', old_rows), ('previous_native_anchors', old_pins)):
            coverage[name+'_after'] = self.guard.coverage(rows) if len(rows) else None
        self.views = self.anchors = None
        self.source_labels = self.source_binding = None
        self.boundary = True
        return dict(owner=self.owner, CE=ce, qualified=qualified, continue_queue=True,
                    inputs_selected=len(inputs), native_anchors_selected=len(anchors), coverage=coverage,
                    protection_bytes=self.guard.check_storage())

    def state(self):
        if not torch.equal(self.master.detach(), self.weight.detach().cpu().float()):
            raise ValueError('Native/master drift')
        return dict(version=VERSION, arm=self.arm, native=self.weight.detach().cpu().clone(),
                    base_native=self.base_native.clone(), master=self.master.detach().clone(),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()), guard=self.guard.state(),
                    owner=self.owner, slot=self.slot, boundary=self.boundary, source_id=self.source_id,
                    source_binding=self.source_binding, source_labels=self.source_labels,
                    anchors=self.anchors, rng=rng_state(), budget=copy.deepcopy(self.budget.data),
                    optimizer_config=OPTIMIZER)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError('Refuse checkpoint symlink')
        temporary = path.with_suffix('.tmp')
        if temporary.exists():
            raise FileExistsError('Interrupted save retained; no automatic overwrite')
        try:
            with temporary.open('xb') as handle:
                torch.save(self.state(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_size > CHECKPOINT_BYTES:
                raise ValueError('Checkpoint cap exceeded')
            temporary.replace(path)
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['version'] != VERSION or state['arm'] != self.arm or state['optimizer_config'] != OPTIMIZER:
            raise ValueError('Checkpoint identity mismatch')
        native = state['native']
        if native.shape != self.weight.shape or native.dtype != self.weight.dtype or not torch.isfinite(native).all():
            raise ValueError('Checkpoint native shape/dtype mismatch')
        if state['master'].dtype != torch.float32 or not torch.equal(state['master'], native.float()):
            raise ValueError('Checkpoint native/master mismatch')
        if not torch.equal(state['base_native'], self.base_native):
            raise ValueError('Original Base binding mismatch')
        if state['budget'] != self.budget.data:
            raise ValueError('Durable ledger and checkpoint differ; ambiguous replay forbidden')
        if not (0 <= state['owner'] <= 8 and 0 <= state['slot'] <= SLOTS):
            raise ValueError('Invalid checkpoint position')
        if state['boundary'] and (state['owner'], state['slot']) != (0, 0) and state['slot'] != SLOTS:
            raise ValueError('Invalid source boundary')
        guard = InputGuard.restore(state['guard'])
        if not torch.equal(guard.base, self.guard.base) or (self.arm == 'U' and int(guard.counts[3]) != 0):
            raise ValueError('Calibration/arm mismatch')
        opt = state['optimizer']
        if len(opt['param_groups']) != 1 or opt['param_groups'][0]['params'] != [0] or set(opt['state']) - {0}:
            raise ValueError('Optimizer binding mismatch')
        for key, value in OPTIMIZER.items():
            if opt['param_groups'][0][key] != value:
                raise ValueError('Optimizer configuration changed')
        for record in opt['state'].values():
            if set(record) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Optimizer schema mismatch')
            for key, value in record.items():
                shape = () if key == 'step' else native.shape
                if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError('Optimizer tensor mismatch')
            if record['step'] < 0 or record['step'] != record['step'].floor() or (record['exp_avg_sq'] < 0).any():
                raise ValueError('Invalid Adam state')
        previous = self.state()
        previous_views = self.views
        try:
            with torch.no_grad():
                self.weight.copy_(native.to(self.weight.device))
                self.master.copy_(state['master'])
            self.optimizer.load_state_dict(opt)
            self.guard = guard
            for key in ('owner', 'slot', 'boundary', 'source_id', 'anchors', 'source_binding', 'source_labels'):
                setattr(self, key, state[key])
            self.views = None
            restore_rng(state['rng'])
        except BaseException:
            self._rollback(previous['native'], previous['master'], previous['optimizer'], previous['rng'])
            self.guard = InputGuard.restore(previous['guard'])
            for key in ('owner', 'slot', 'boundary', 'source_id', 'anchors', 'source_binding', 'source_labels'):
                setattr(self, key, previous[key])
            self.views = previous_views
            raise
