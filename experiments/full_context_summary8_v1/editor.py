"""Reuse transactional training unchanged; replace only qualified-source summary."""
import math
import time
import torch
from experiments.fixed_input_guard8_v1 import executor as old
from experiments.fixed_input_guard8_v1.adapter import context_positions
from .summary import collect, weighted_rows, same_state_diagnostic


class Editor(old.Editor):
    def __init__(self, runtime, arm, calibration_inputs, budget):
        if arm not in ('B','C','N','D','E'):
            raise ValueError('Unknown summary arm')
        super().__init__(runtime, 'P', calibration_inputs, budget)
        self.arm = arm

    @torch.no_grad()
    def finish_source(self):
        if self.boundary or self.slot != 20 or self.views is None:
            raise ValueError('Incomplete source')
        rng = old.rng_state()
        started = time.monotonic()
        reports = []
        try:
            ce = self.adapter.measure(self.views)
            qualified = math.isfinite(ce) and ce < .1
            selected = 0
            if qualified:
                anchors = self.adapter.inputs(self.views, anchors=True)
                self.guard.begin(anchors)
                for v,batch in enumerate(self.views):
                    h,energy,valid,strata,metrics = collect(self.adapter,batch,self.arm in ('D','E'))
                    sampled = context_positions(batch,7 if v < 2 else 6)
                    seed = 62018000+self.owner*10+v
                    rows,weights = weighted_rows(h,energy,valid,strata,sampled,self.arm,seed)
                    report = dict(view=v,weights=weights,numerics=metrics,sketch=self.guard.add(rows))
                    selected += len(rows)
                    if self.arm == 'D' and self.owner == 1 and v == 0:
                        fixed = torch.cat((self.guard.base[:,:int(self.guard.counts[0])],
                                           self.guard.pinned[:,:int(self.guard.pin_count)]),1)
                        report['same_state'] = same_state_diagnostic(h,energy,valid,strata,sampled,fixed,seed)
                    reports.append(report)
                    # Only scalar diagnostics survive; no complete current-source matrix cache.
                    del h,energy,rows
                state = self.guard.end()
            else:
                state = dict(protection_bytes=self.guard.check_storage(),summary_skipped_unqualified=True)
        finally:
            old.restore_rng(rng)
        self.views = self.anchors = self.source_labels = self.source_binding = None
        self.boundary = True
        return dict(owner=self.owner,CE=ce,qualified=qualified,continue_queue=True,
            inputs_selected=selected,native_anchors_selected=len(anchors) if qualified else 0,
            views=reports,summary_elapsed_seconds=time.monotonic()-started,**state)


def summary_admission(runtime,arm,calibration,record,fits,folder):
    editor = Editor(runtime,arm,calibration,old.Budget(folder/'summary-admission-budget.json'))
    rng = old.rng_state()
    before = editor.weight.detach().cpu().clone()
    try:
        editor.prepare_source(record,fits,1)
        rows = []
        for v,batch in enumerate(editor.views):
            h,energy,valid,strata,metrics = collect(editor.adapter,batch,arm in ('D','E'))
            weighted,info = weighted_rows(h,energy,valid,strata,context_positions(batch,7 if v<2 else 6),arm,62018010+v)
            if arm in ('D','E') and not all(metrics['gradient_nonzero_elements']):
                raise ValueError('Zero native token VJP requires numerical diagnosis')
            rows.append(dict(view=v,weights=info,numerics=metrics))
        if not torch.equal(before,editor.weight.detach().cpu()) or editor.optimizer.state or editor.weight.grad is not None:
            raise ValueError('Summary admission mutated learner state')
        return dict(status='PASS',views=rows,training_feedback=False,optimizer_mutated=False,
                    total_supervised_tokens=sum(x['numerics']['supervised_tokens'] for x in rows),
                    counters=editor.budget.data['counts'])
    finally:
        old.restore_rng(rng)
        editor.adapter.close()
