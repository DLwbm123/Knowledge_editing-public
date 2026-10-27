"""Read-only answer likelihoods; these are diagnostics, not correctness labels."""


def answer_scores(logits, labels, eos_id):
    import torch
    shifted = labels[:, 1:]
    mask = shifted != -100
    assert mask.any() and logits.shape[:2] == labels.shape
    targets = shifted[mask]
    logp = torch.log_softmax(logits[:, :-1][mask].float(), dim=-1)
    values = logp.gather(1, targets[:, None]).squeeze(1)
    assert torch.isfinite(values).all()
    content = targets != eos_id
    assert content.any(), "Candidate has no content tokens"
    return dict(target_ids=targets.detach().cpu().tolist(),
                token_logp=values.detach().cpu().double().tolist(),
                answer_tokens=int(content.sum()),
                mean_no_eos=float(values[content].mean()),
                sum_no_eos=float(values[content].sum()),
                mean_with_eos=float(values.mean()),
                sum_with_eos=float(values.sum()),
                eos_logp=values[~content].detach().cpu().double().tolist())


def phrase_span(tokenizer, ids, phrase):
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    begin = decoded.casefold().find(phrase.casefold())
    if begin < 0:
        return dict(status="PHRASE_NOT_FOUND", decoded_target=decoded, indices=[])
    ends = [0]
    for i in range(1, len(ids)+1):
        prefix = tokenizer.decode(ids[:i], skip_special_tokens=True)
        if not decoded.startswith(prefix):
            return dict(status="NON_PREFIX_STABLE_TOKENIZER", decoded_target=decoded, indices=[])
        ends.append(len(prefix))
    indices = [i for i in range(len(ids)) if ends[i] < begin+len(phrase) and ends[i+1] > begin]
    assert indices
    return dict(status="BOUND", decoded_target=decoded, indices=indices,
                note="Tokens overlap phrase characters; boundary tokens can include adjacent text.")


def first_token_difference(generated, candidate, eos_id):
    a = list(generated)
    b = list(candidate)
    if a and a[-1] == eos_id:
        a.pop()
    if b and b[-1] == eos_id:
        b.pop()
    index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return dict(index=None if a == b else index, generated_length=len(a), candidate_length=len(b),
                interpretation="Exact token difference only; not a semantic error position.")
