"""Scoped teacher-forcing cache for the frozen decoder layers before the declared edited layer."""
import torch


def signature(tensor):
    if tensor is None:
        return None
    return (id(tensor), tensor.data_ptr(), tensor._version, tuple(tensor.shape),
            tensor.dtype, tensor.device, tensor.requires_grad)


class PrefixCache:
    # shortcut: fixed teacher-forcing batches only; never use for generation or hidden-state outputs.
    def __init__(self, runtime, batches, target_layer=21):
        self.runtime = runtime
        self.target_layer = target_layer
        assert 1 <= target_layer < len(runtime.model.model.layers)
        self.layers = list(runtime.model.model.layers[:target_layer])
        assert len(self.layers) == target_layer and not runtime.model.training
        assert not runtime.model.config.output_hidden_states and not runtime.model.config.output_attentions
        assert not any(p.requires_grad for layer in self.layers for p in layer.parameters())
        assert not any(layer._forward_hooks or layer._forward_pre_hooks for layer in self.layers)
        self.batches = {id(b): b for b in batches}
        self.inputs = {id(b): self.batch_signature(b) for b in batches}
        self.parameters = [p for layer in self.layers for p in layer.parameters()]
        self.weights = [signature(p) for p in self.parameters]
        self.cache = {}
        self.current = None
        self.hits = self.misses = 0

    @staticmethod
    def batch_signature(batch):
        kwargs = batch.forward_kwargs()
        assert kwargs['use_cache'] is False
        return {key: signature(value) if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()}

    def __enter__(self):
        self.original_loss = self.runtime.compute_loss
        self.originals = [layer.forward for layer in self.layers]
        for index, layer in enumerate(self.layers):
            def forward(hidden_states, *args, _index=index, **kwargs):
                if self.current is None:
                    return self.originals[_index](hidden_states, *args, **kwargs)
                assert not args and not kwargs.get('use_cache') and not kwargs.get('output_attentions')
                assert kwargs.get('past_key_value') is None
                if self.current in self.cache:
                    return (self.cache[self.current] if _index == self.target_layer - 1 else hidden_states,)
                output = self.originals[_index](hidden_states, **kwargs)
                if _index == self.target_layer - 1:
                    assert len(output) == 1 and not output[0].requires_grad
                    self.cache[self.current] = output[0].detach().clone()
                return output
            layer.forward = forward

        def compute_loss(batch):
            assert self.current is None and id(batch) in self.batches
            assert not self.runtime.model.training
            assert self.batch_signature(batch) == self.inputs[id(batch)]
            assert [signature(p) for p in self.parameters] == self.weights
            self.current = id(batch)
            if self.current in self.cache:
                self.hits += 1
            else:
                self.misses += 1
            try:
                return self.original_loss(batch)
            finally:
                self.current = None
        self.runtime.compute_loss = compute_loss
        return self

    def __exit__(self, *unused):
        self.runtime.compute_loss = self.original_loss
        for layer, forward in zip(self.layers, self.originals):
            layer.forward = forward
        self.cache.clear()

