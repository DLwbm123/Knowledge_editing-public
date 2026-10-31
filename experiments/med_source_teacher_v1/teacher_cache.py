"""Build per-input teacher distributions after the caller verifies identity and roles."""
import hashlib
import torch
import objective


def input_digest(batch):
    digest = hashlib.sha256()
    def frame(tag, data):
        digest.update(tag)
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    def visit(value):
        if isinstance(value, torch.Tensor):
            x = value.detach().cpu().contiguous()
            frame(b'T', repr((str(x.dtype), tuple(x.shape))).encode())
            frame(b'V', x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            frame(b'D', str(len(value)).encode())
            for key in sorted(value):
                if not isinstance(key, str):
                    raise ValueError('Model input keys must be strings')
                frame(b'K', key.encode()); visit(value[key])
        elif isinstance(value, (list, tuple)):
            frame(b'L', repr((type(value).__name__, len(value))).encode())
            for item in value: visit(item)
        elif value is None or isinstance(value, (str, int, float, bool)):
            frame(b'S', repr((type(value).__name__, value)).encode())
        else:
            raise ValueError('Unsupported model input type')
    visit(batch.forward_kwargs())
    visit(batch.labels)
    return digest.hexdigest()


def build(runtime, hook, batches, byte_limit, budget):
    if len(batches) != 5 or byte_limit <= 0:
        raise ValueError('Five source inputs and a bounded cache are required')
    cache, total = [], 0
    for batch in batches:
        budget()
        binding = input_digest(batch)
        hook.set_teacher_routing(batch.labels)
        with torch.no_grad():
            output = runtime.model(**batch.forward_kwargs())
            log_probs, targets = objective.frozen_teacher(output, batch.labels)
        log_probs = log_probs.cpu(); targets = targets.cpu(); labels = batch.labels.detach().cpu().clone()
        total += sum(x.numel() * x.element_size() for x in (log_probs, targets, labels))
        if total > byte_limit:
            raise ValueError('Teacher cache exceeds its fixed bound')
        cache.append(dict(input_digest=binding, labels=labels, targets=targets, log_probs=log_probs))
    return cache, total
