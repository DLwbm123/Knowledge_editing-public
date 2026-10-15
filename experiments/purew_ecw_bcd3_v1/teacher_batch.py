from pathlib import Path
from runtime import c

def teacher_batch(runtime, row, tokens, expected):
    import torch
    raw = runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])), row['question'], None)
    assert raw['image_sha256'] == expected['image_sha256']
    assert raw['input_ids'][0].tolist() == expected['prompt_ids']
    assert raw['attention_mask'][0].tolist() == expected['attention_mask']
    assert 0 < len(tokens) <= 128 and tokens[-1] == runtime.adapter.tokenizer.eos_token_id
    answer = torch.tensor([tokens], device=runtime.device, dtype=raw['input_ids'].dtype)
    ids = torch.cat([raw['input_ids'], answer], 1)
    attention = torch.cat([raw['attention_mask'], torch.ones_like(answer)], 1)
    labels = torch.cat([torch.full_like(raw['input_ids'], -100), answer], 1)
    embeds, attention, positions, labels = runtime._expand_multimodal(
        raw_input_ids=ids, attention_mask=attention, labels=labels, images=raw['images'])
    assert embeds.dtype == torch.float16 and labels[0, labels[0] != -100].tolist() == tokens
    kwargs = dict(inputs_embeds=embeds.detach().cpu(), attention_mask=attention.detach().cpu(),
                  position_ids=positions.detach().cpu() if positions is not None else None,
                  labels=None, use_cache=False, return_dict=True)
    return kwargs, labels.detach().cpu()

