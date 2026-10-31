"""Source/FIT teacher KL, operating only on already bound answer positions."""
import torch
import torch.nn.functional as F


def answer_logits(output, labels):
    logits = output.logits
    if labels.ndim != 2 or logits.ndim != 3 or logits.shape[:2] != labels.shape:
        raise ValueError('Logits and label positions differ')
    mask = labels[:, 1:] != -100
    if not bool(mask.any()):
        raise ValueError('Empty answer support')
    selected = logits[:, :-1][mask].float()
    if not bool(torch.isfinite(selected).all()):
        raise ValueError('Nonfinite answer logits')
    return selected, labels[:, 1:][mask]


def frozen_teacher(output, labels):
    logits, targets = answer_logits(output, labels)
    return F.log_softmax(logits.detach(), dim=-1), targets.detach().clone()


def loss(output, labels, teacher_log_probs, teacher_targets):
    logits, targets = answer_logits(output, labels)
    if teacher_log_probs.requires_grad or teacher_log_probs.shape != logits.shape:
        raise ValueError('Teacher must be detached and match all answer positions')
    if not torch.equal(targets, teacher_targets) or not bool(torch.isfinite(teacher_log_probs).all()):
        raise ValueError('Teacher targets or log probabilities differ')
    if not torch.allclose(teacher_log_probs.logsumexp(-1), torch.zeros_like(targets, dtype=torch.float32), atol=1e-5, rtol=0):
        raise ValueError('Teacher distribution is not normalized')
    if output.loss is None or not bool(torch.isfinite(output.loss)):
        raise ValueError('Nonfinite source cross entropy')
    student_log_probs = F.log_softmax(logits, dim=-1)
    kl = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(-1).mean()
    if not bool(torch.isfinite(kl)):
        raise ValueError('Nonfinite teacher divergence')
    return output.loss + kl, kl
