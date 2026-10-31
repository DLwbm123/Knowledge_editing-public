"""One fixed source/FIT gradient step; inputs and caches require prior admission."""
import torch
import objective


def backwards(runtime, hook, expert, batches, teachers, step, budget, counts):
    if len(batches) != 5 or len(teachers) != 5 or not 1 <= step <= 160:
        raise ValueError('Expected native and four fixed FIT batches')
    expert.zero_grad(set_to_none=True)
    rows = []
    for index in (0, 1 + (step - 1) % 4):
        budget()
        batch, teacher = batches[index], teachers[index]
        if not torch.equal(batch.labels, teacher['labels'].to(batch.labels.device)):
            raise ValueError('Teacher answer mask or positions differ')
        hook.set_teacher_routing(batch.labels)
        output = runtime.model(**batch.forward_kwargs())
        log_probs = teacher['log_probs'].to(output.logits.device)
        targets = teacher['targets'].to(output.logits.device)
        total, kl = objective.loss(output, batch.labels, log_probs, targets)
        (0.5 * total).backward()
        counts['backwards'] += 1
        rows.append({'index': index, 'ce': float(output.loss.detach()), 'kl': float(kl.detach())})
    return rows
