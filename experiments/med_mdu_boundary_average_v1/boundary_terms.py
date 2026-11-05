import torch
import torch.nn.functional as F
from average import partition_means


def terms(runtime, hook, batch, row, length, spans, view):
    hook.set_teacher_routing(batch.labels)
    output = runtime.model(**batch.forward_kwargs())
    ce = output.loss
    if ce is None or not torch.isfinite(ce):
        raise ValueError('Invalid CE')
    labels = batch.labels[:, 1:]
    mask = labels != -100
    ids = labels[mask]
    raw = runtime._tokenize_prompt(runtime.adapter._prompt(row['question'], row['reference']))[0]
    start, end = view['answer_start_label'], view['answer_end_label']
    if not (0 <= start < end <= raw.numel()) or end - start != length:
        raise ValueError('Frozen answer positions do not fit the raw prompt')
    target = raw[start:end].cpu()
    hits = [j for j in range(ids.numel() - length + 1) if torch.equal(ids[j:j+length].cpu(), target)]
    if len(hits) != 1:
        raise ValueError('Frozen answer tokens do not uniquely match expanded labels')
    offset = hits[0]
    decoded = runtime.adapter.tokenizer.decode(ids[offset:offset+length].tolist(), skip_special_tokens=False).strip()
    if decoded != row['reference'].strip():
        raise ValueError('Expanded target does not match the source answer')
    per_token = F.cross_entropy(output.logits[:, :-1, :].float().reshape(-1, output.logits.shape[-1]),
                                labels.reshape(-1), ignore_index=-100, reduction='none').reshape(labels.shape)[mask][offset:offset+length]
    if not spans or any(not (0 <= a < b <= length) for a, b in spans):
        raise ValueError('Invalid frozen answer span')
    losses = partition_means(per_token)
    if not torch.isfinite(losses).all():
        raise ValueError('Invalid span loss')
    return ce, losses
