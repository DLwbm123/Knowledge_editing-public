"""EOS-only loss after a frozen source-answer prefix, with full answer routing."""
from types import SimpleNamespace
import torch


def make_batch(runtime, expanded, prefix_token_ids):
    eos = runtime.adapter.tokenizer.eos_token_id
    assert prefix_token_ids and eos not in prefix_token_ids
    prompt = expanded[4].to(runtime.device).float()
    tokens = torch.tensor([prefix_token_ids + [eos]], device=runtime.device, dtype=torch.long)
    with torch.no_grad():
        inputs = torch.cat((prompt, runtime.model.get_input_embeddings()(tokens)), dim=1)
    labels = torch.full(inputs.shape[:2], -100, device=runtime.device, dtype=torch.long)
    labels[:, -1] = eos
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, prompt.shape[1] - 1:-1] = True
    assert int((labels != -100).sum()) == 1 and int(mask.sum()) == len(prefix_token_ids) + 1
    assert not mask[:, :prompt.shape[1] - 1].any() and not mask[:, -1].any()
    kwargs = dict(inputs_embeds=inputs, attention_mask=torch.ones_like(labels),
                  position_ids=None, labels=labels, use_cache=False, return_dict=True)
    batch = SimpleNamespace(labels=labels, forward_kwargs=lambda: kwargs)
    return batch, mask


def selfcheck():
    embedding = torch.nn.Embedding(10, 4)
    embedding.requires_grad_(False)
    runtime = SimpleNamespace(device='cpu', model=SimpleNamespace(get_input_embeddings=lambda: embedding),
                              adapter=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2)))
    prompt = torch.arange(12, dtype=torch.float16).reshape(1, 3, 4)
    batch, mask = make_batch(runtime, (None, None, None, None, prompt), [4, 5])
    kw = batch.forward_kwargs()
    assert torch.equal(kw['inputs_embeds'][:, :3], prompt.float())
    assert torch.equal(kw['inputs_embeds'][:, 3:], embedding(torch.tensor([[4, 5, 2]])))
    assert mask.tolist() == [[False, False, True, True, True, False]]
    assert batch.labels.tolist() == [[-100, -100, -100, -100, -100, 2]]
    logits = torch.randn(1, 6, 10, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 10),
        batch.labels[:, 1:].reshape(-1), ignore_index=-100)
    expected = torch.nn.functional.cross_entropy(logits[0, 4:5], torch.tensor([2]))
    assert torch.equal(loss, expected)
    loss.backward()
    assert logits.grad[0, 4].norm() > 0 and torch.count_nonzero(logits.grad[0, :4]) == 0
    return dict(status='PASS', only_EOS_supervised=True, full_prior_answer_routing=True,
                prompt_and_prefix_labels_ignored=True, causal_shift_exact=True)


if __name__ == '__main__':
    print(selfcheck())
