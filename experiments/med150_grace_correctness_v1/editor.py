"""Fixed LLaMA-reference GRACE adaptation with query-anchored cache semantics."""
from contextlib import contextmanager
from pathlib import Path
import torch
import torch.nn as nn
from m3bench_repro.editors.methods import (
    GracePaperSpecEditor, PaperSpecEditor, record_seed, finite_gradients,
)
from m3bench_repro.editors.llava_runtime import seed_everything
from m3bench_repro.editors.routed_layers import GraceValueLinear
from m3bench_repro.editors.routing import GraceCodebook
from dataclasses import asdict


class QueryAnchoredLinear(GraceValueLinear):
    def __init__(self, base):
        super().__init__(base, replacement='replace_last')
        self.generation_mode = False
        self.codebook = None
        self.trace = []

    def forward(self, inputs):
        if self.generation_mode:
            length = inputs.shape[-2]
            # Cached decode contains only new positions. The prompt intervention
            # is already represented in downstream KV state; do not reapply it.
            if length <= self.token_index:
                self.trace.append(dict(length=length, prompt_position_present=False, replaced=0))
                return self.base(inputs)
            if inputs.ndim != 3 or inputs.shape[0] != 1:
                raise ValueError('This frozen adapter requires one query per batch')
            decision = self.codebook.route(inputs[0, self.token_index].detach().float())
            self.active_logical_id = decision.logical_edit_id
            self.trace.append(dict(length=length, prompt_position_present=True,
                                   replaced=int(decision.activated)))
        return super().forward(inputs)


class CorrectedGraceEditor(GracePaperSpecEditor):
    def __init__(self, runtime):
        PaperSpecEditor.__init__(self, runtime)
        self.target = 'model.layers.27.mlp.down_proj'
        base = runtime.get_module(self.target)
        if not isinstance(base, nn.Linear):
            raise TypeError('Reference target is not nn.Linear')
        self.wrapper = QueryAnchoredLinear(base)
        runtime.replace_module(self.target, self.wrapper)
        self.codebook = GraceCodebook(distance='euclidean', eps_init=1.0)
        self.requested_to_effective = {}

    def apply_edit(self, record):
        seed = record_seed(record.record_id, self.method)
        seed_everything(seed)
        key = self._question_key(record)
        batch = self.runtime.build_edit_batch(record)
        insertion = self.codebook.insert_with_source_semantics(record.record_id, key, batch.target_token_ids)
        effective = insertion.effective_logical_edit_id
        if effective not in self.wrapper.logical_to_slot:
            self.wrapper.add_cold_value(effective, seed=seed)
        parameters = self.wrapper.train_only(effective)
        if len(parameters) != 1:
            raise ValueError('Only the selected value may be optimized')
        optimizer = torch.optim.Adam(parameters, lr=1.0)
        losses, gradients, nonzero = [], [], []
        try:
            with self._activated(effective, token_index=batch.key_token_index):
                for _ in range(50):
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.runtime.compute_loss(batch)
                    loss.backward()
                    gradients.append(finite_gradients(parameters))
                    nonzero.append(any(p.grad is not None and bool(torch.any(p.grad != 0)) for p in parameters))
                    optimizer.step()
                    losses.append(float(loss.detach()))
        finally:
            self.wrapper.train_only(None)
            self.wrapper.disable()
        self.requested_to_effective[record.record_id] = effective
        self.edit_history.append(record.record_id)
        return dict(record_id=record.record_id, seed=seed, losses=losses, steps=50,
                    finite_losses=all(torch.isfinite(torch.tensor(losses)).tolist()),
                    finite_gradients=all(gradients), nonzero_gradient_steps=sum(nonzero),
                    insert=asdict(insertion), entry_count=len(self.codebook),
                    key_norm=float(torch.linalg.vector_norm(key)),
                    trainable_parameter_count=sum(p.numel() for p in parameters), target_mask=batch.mask_report())

    @contextmanager
    def route_generation(self, record):
        question = self.runtime.build_question_batch(record)
        decision = self._route(record)
        self.wrapper.codebook = self.codebook
        self.wrapper.trace = []
        self.wrapper.set_active(decision.logical_edit_id, token_index=question.key_token_index)
        self.wrapper.generation_mode = True
        try:
            yield decision
        finally:
            self.wrapper.generation_mode = False
            self.wrapper.disable()

    def save_editor_state(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(method=self.method, target=self.target, codebook=self.codebook.export_state(),
                        wrapper=self.wrapper.export_state(), requested_to_effective=self.requested_to_effective,
                        edit_history=self.edit_history, corrected_config=self.config_lock()), path)
        return dict(size_bytes=path.stat().st_size)

    def load_editor_state(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('corrected_config') != self.config_lock():
            raise ValueError('Corrected state/config binding mismatch')
        super().load_editor_state(path)

    def config_lock(self):
        return dict(method='GRACE_MM_LLAMA_REFERENCE_CACHE_CONSISTENT_V1',
                    target=self.target, distance='euclidean', eps_init=1.0, optimizer='Adam',
                    learning_rate=1.0, steps_per_edit=50, value_init='cold uniform [0,1)',
                    replacement='replace_last at expanded last-prompt position',
                    cache='prefill-only intervention, query-anchored full-context equivalence',
                    label_collision='exact target token tuple equality, declared multimodal adaptation',
                    reference_commit='3a3e2682c37e3e6da4e6dbcfcacc109481e66175',
                    reference_file='hparams/GRACE/llama-7b.yaml')
