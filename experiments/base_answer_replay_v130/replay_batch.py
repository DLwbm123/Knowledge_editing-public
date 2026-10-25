"""Teacher-force a frozen generated continuation without decoding or retokenizing."""
from types import SimpleNamespace
import torch


def make_batch(runtime, expanded, target_token_ids):
    eos = runtime.adapter.tokenizer.eos_token_id
    assert target_token_ids and target_token_ids[-1] == eos and eos not in target_token_ids[:-1]
    prompt = expanded[4].to(runtime.device).float()
    tokens = torch.tensor([target_token_ids], device=runtime.device, dtype=torch.long)
    with torch.no_grad():
        inputs = torch.cat((prompt, runtime.model.get_input_embeddings()(tokens)), dim=1)
    labels = torch.full(inputs.shape[:2], -100, device=runtime.device, dtype=torch.long)
    labels[:, prompt.shape[1]:] = tokens
    assert labels[labels != -100].tolist() == target_token_ids
    kwargs = dict(inputs_embeds=inputs, attention_mask=torch.ones_like(labels),
                  position_ids=None, labels=labels, use_cache=False, return_dict=True)
    return SimpleNamespace(labels=labels, forward_kwargs=lambda: kwargs)


def selfcheck():
    embedding = torch.nn.Embedding(10, 4)
    embedding.requires_grad_(False)
    runtime = SimpleNamespace(device='cpu', model=SimpleNamespace(get_input_embeddings=lambda: embedding),
        adapter=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2)))
    prompt = torch.arange(12, dtype=torch.float16).reshape(1, 3, 4)
    batch = make_batch(runtime, (None, None, None, None, prompt), [4, 5, 2])
    assert batch.labels.tolist() == [[-100, -100, -100, 4, 5, 2]]
    assert torch.equal(batch.forward_kwargs()['inputs_embeds'][:, :3], prompt.float())
    logits = torch.randn(1, 6, 10, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 10),
        batch.labels[:, 1:].reshape(-1), ignore_index=-100)
    expected = torch.nn.functional.cross_entropy(logits[0, 2:5], torch.tensor([4, 5, 2]))
    assert torch.equal(loss, expected)
    loss.backward()
    assert logits.grad[0, 2:5].norm(dim=-1).min() > 0
    assert torch.count_nonzero(logits.grad[0, :2]) == torch.count_nonzero(logits.grad[0, 5:]) == 0
    return dict(status='PASS', prompt_ignored=True, exact_raw_continuation=True,
                each_content_and_EOS_causal_predictor_supervised=True)
