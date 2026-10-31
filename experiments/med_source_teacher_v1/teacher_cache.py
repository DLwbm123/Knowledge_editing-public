"""Build per-input teacher distributions after the caller verifies identity and roles."""
import hashlib
import torch
import objective


def cache_digest(cache):
    digest = hashlib.sha256()
    for row in cache:
        digest.update(row['input_digest'].encode())
        for key in ('labels', 'targets', 'log_probs'):
            x = row[key].detach().cpu().contiguous()
            meta = repr((key, str(x.dtype), tuple(x.shape))).encode()
            digest.update(len(meta).to_bytes(8, 'big')); digest.update(meta)
            raw = x.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, 'big')); digest.update(raw)
    return digest.hexdigest()


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


def freeze(packets, expected_states, vocabulary_size, byte_limit):
    """Validate all fixed owners before any student is trained."""
    if len(expected_states) != 16 or set(packets) != set(expected_states) or vocabulary_size <= 0:
        raise ValueError('Teacher owner coverage invalid')
    records, total = {}, 0
    for owner, packet in packets.items():
        if packet['teacher_state_digest'] != expected_states[owner] or len(packet['cache']) != 5:
            raise ValueError('Teacher identity or input coverage differs')
        for row in packet['cache']:
            labels, targets, logp = row['labels'], row['targets'], row['log_probs']
            if labels.ndim != 2 or logp.ndim != 2 or logp.dtype != torch.float32 or logp.shape != (targets.numel(), vocabulary_size):
                raise ValueError('Teacher mask or vocabulary invalid')
            if not torch.equal(labels[:, 1:][labels[:, 1:] != -100], targets) or not targets.numel():
                raise ValueError('Teacher labels and targets differ')
            if logp.requires_grad or not bool(torch.isfinite(logp).all()) or not torch.allclose(logp.logsumexp(-1), torch.zeros(targets.numel()), atol=1e-5, rtol=0):
                raise ValueError('Teacher distribution invalid')
            total += sum(x.numel() * x.element_size() for x in (labels, targets, logp))
            if total > byte_limit:
                raise ValueError('Aggregate cache bound exceeded')
        records[owner] = dict(teacher_state_digest=expected_states[owner], cache_digest=cache_digest(packet['cache']))
    return dict(status='ALL_80_TEACHERS_FROZEN', count=80, tensor_bytes=total, owners=records)
