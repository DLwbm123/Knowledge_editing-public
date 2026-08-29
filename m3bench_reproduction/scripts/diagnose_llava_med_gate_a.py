#!/usr/bin/env python3
"""One-case, token-level forensic capture for the frozen LLaVA-Med upstream path."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from PIL import Image

from m3bench_repro.inference import LlavaMedAdapter


def token_text(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def top_tokens(logits: torch.Tensor, tokenizer, eos: int, limit: int = 20) -> dict:
    values, ids = torch.topk(logits.float(), k=limit)
    probabilities = torch.softmax(logits.float(), dim=-1)
    ranking = torch.argsort(logits.float(), descending=True)
    eos_rank = int((ranking == eos).nonzero(as_tuple=False)[0].item()) + 1
    return {
        "nan_count": int(torch.isnan(logits).sum().item()),
        "inf_count": int(torch.isinf(logits).sum().item()),
        "token_ids": [int(x) for x in ids.tolist()],
        "token_strings": [token_text(tokenizer, int(x)) for x in ids.tolist()],
        "logits": [float(x) for x in values.tolist()],
        "eos_rank": eos_rank,
        "eos_logit": float(logits[eos].item()),
        "eos_probability": float(probabilities[eos].item()),
        "top1_minus_eos": float(values[0].item() - logits[eos].item()),
    }


def chunks_around_image(input_ids: list[int], image_token: int, tokenizer) -> list[str]:
    position = input_ids.index(image_token)
    return [
        tokenizer.decode(input_ids[:position], skip_special_tokens=False),
        tokenizer.decode(input_ids[position + 1 :], skip_special_tokens=False),
    ]


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--vision-tower", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    adapter = LlavaMedAdapter(args.model, args.vision_tower)
    adapter.load()
    model, tokenizer = adapter.model, adapter.tokenizer
    model.eval()
    if model.config.mm_use_im_start_end:
        query = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + args.question
    else:
        query = DEFAULT_IMAGE_TOKEN + "\n" + args.question
    conv = conv_templates["mistral_instruct"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    ids = [int(x) for x in input_ids[0].tolist()]
    assert ids.count(IMAGE_TOKEN_INDEX) == 1, "expected exactly one image token"
    # This diagnostic accepts only an image and question; no answer/gold field is
    # present in its CLI or in the assembled prompt.
    supplied_gold_answer = None
    assert supplied_gold_answer is None or supplied_gold_answer not in prompt
    image = Image.open(args.image).convert("RGB")
    processed = process_images([image], adapter.image_processor, model.config)
    image_tensor = processed[0].unsqueeze(0) if isinstance(processed, list) else processed
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    assert torch.isfinite(image_tensor).all().item(), "non-finite processed image"
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        generation = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            num_beams=1,
            max_new_tokens=64,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad,
        )
        forward = model(input_ids=input_ids, images=image_tensor, use_cache=False, return_dict=True)
    sequences = generation.sequences.detach().cpu()
    sequence_ids = [int(x) for x in sequences[0].tolist()]
    full_raw = tokenizer.decode(sequence_ids, skip_special_tokens=False)
    full_clean = tokenizer.decode(sequence_ids, skip_special_tokens=True).strip()
    legacy_ids = sequence_ids[input_ids.shape[1] :]
    legacy_clean = tokenizer.decode(legacy_ids, skip_special_tokens=True).strip()
    eos = int(tokenizer.eos_token_id)
    sequence_prefix_length = len(sequence_ids) - len(generation.scores)
    generated_ids = sequence_ids[sequence_prefix_length:]
    first = generated_ids[0] if generated_ids else None
    first_eos = next((index for index, token in enumerate(sequence_ids) if token == eos), None)
    score_first = generation.scores[0][0].detach().float() if generation.scores else None
    forward_last = forward.logits[0, -1].detach().float()
    result = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "transformers": transformers.__version__, "torch": torch.__version__, "model_class": type(model).__name__,
            "model_dtype": str(model.dtype), "model_device": str(model.device), "training": bool(model.training),
            "tokenizer_class": type(tokenizer).__name__, "tokens": {"bos": tokenizer.bos_token, "eos": tokenizer.eos_token, "pad": tokenizer.pad_token, "unk": tokenizer.unk_token},
            "token_ids": {"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id, "unk": tokenizer.unk_token_id},
            "generation_config": model.generation_config.to_dict(), "mm_use_im_start_end": bool(model.config.mm_use_im_start_end),
            "mm_use_im_patch_token": bool(model.config.mm_use_im_patch_token), "mm_vision_tower": str(model.config.mm_vision_tower), "image_aspect_ratio": model.config.image_aspect_ratio,
        },
        "prompt": {"question": args.question, "repr": repr(prompt), "conversation_mode": "mistral_instruct", "input_ids_shape": list(input_ids.shape), "input_ids": ids,
                   "image_token_count": ids.count(IMAGE_TOKEN_INDEX), "image_token_positions": [i for i, token in enumerate(ids) if token == IMAGE_TOKEN_INDEX],
                   "decoded_chunks_around_image": chunks_around_image(ids, IMAGE_TOKEN_INDEX, tokenizer), "gold_answer_present": False},
        "image": {"mode": image.mode, "size": list(image.size), "tensor_shape": list(image_tensor.shape), "dtype": str(image_tensor.dtype), "device": str(image_tensor.device),
                  "min": float(image_tensor.min().item()), "max": float(image_tensor.max().item()), "mean": float(image_tensor.float().mean().item()), "std": float(image_tensor.float().std().item()), "finite": True},
        "generation": {"sequences_shape": list(sequences.shape), "sequence_ids": sequence_ids, "score_steps": len(generation.scores), "generation_sequence_prefix_length": sequence_prefix_length, "generated_ids": generated_ids,
                       "sequence_length_equals_score_steps": len(sequence_ids) == len(generation.scores),
                       "first_generated_token_id": first, "first_generated_is_eos": first == eos, "first_eos_position": first_eos, "complete_decode_raw": full_raw, "complete_decode_clean": full_clean,
                       "legacy_suffix_prompt_length": int(input_ids.shape[1]), "legacy_suffix_ids": legacy_ids, "legacy_suffix_decode_clean": legacy_clean,
                       "complete_nonempty": bool(full_clean), "legacy_suffix_empty": not bool(legacy_clean), "raw_token_ids_preserved": True},
        "first_step_logits": top_tokens(score_first, tokenizer, eos) if score_first is not None else None,
        "direct_forward": {"final_position_top20": top_tokens(forward_last, tokenizer, eos), "top1_token_id": int(torch.argmax(forward_last).item()),
                           "agrees_with_generate_first": bool(first == int(torch.argmax(forward_last).item()))},
    }
    atomic_json(result, args.output)
    print(json.dumps({"complete_decode_clean": full_clean, "legacy_suffix_decode_clean": legacy_clean, "first_token": first}, ensure_ascii=False))


if __name__ == "__main__":
    main()
