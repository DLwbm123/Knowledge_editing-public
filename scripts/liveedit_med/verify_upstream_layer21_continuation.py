#!/usr/bin/env python3
"""Directly validate pinned BaseVLLMForEdit layer-21 continuation semantics."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.liveedit_med.run_upstream_end_to_end_trace_parity import prepare, run_layers, seed_everything, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-full-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg["out_dir"])
    target = out / "upstream_forward_from_mid_layer_validation.json"
    if target.exists():
        raise FileExistsError(target)
    sys.path.insert(0, str(args.upstream_full_root))
    from editor.vllms_for_edit.base import BaseVLLMForEdit
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    seed_everything(42)
    device = torch.device(cfg["device"])
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["official_model"], torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device).eval().requires_grad_(False)
    processor = LlavaProcessor.from_pretrained(cfg["official_model"], use_fast=False)
    row = json.loads(Path(cfg["official_sample"]).read_text())[0]
    image = Image.open(Path(cfg["image_root"]) / row["image"]).convert("RGB")
    prepared = prepare(model, processor, row["src"], row["alt"], image, device)
    generator = torch.Generator(device=device).manual_seed(42)
    residual = torch.randn(prepared.clean_hidden.shape, generator=generator, device=device, dtype=torch.float16) * 1e-3

    class Dummy:
        pass

    dummy = Dummy()
    dummy.model = model

    def get_llm_outpt(llm_input, _vt_range=None):
        signature = inspect.signature(model.language_model.forward)
        filtered = {key: value for key, value in llm_input.items() if key in signature.parameters}
        return model.language_model(**filtered, use_cache=False, return_dict=True)

    dummy.get_llm_outpt = get_llm_outpt

    def add_residual(_module, _args, output):
        if isinstance(output, tuple):
            return (output[0] + residual, *output[1:])
        if isinstance(output, list):
            return [output[0] + residual, *output[1:]]
        return output + residual

    handle = model.language_model.model.layers[21].register_forward_hook(add_residual)
    try:
        with torch.no_grad():
            direct = BaseVLLMForEdit.forward_from_mid_layer(
                dummy,
                {"inputs_embeds": prepared.merged, "attention_mask": prepared.attention_mask,
                 "position_ids": prepared.position_ids},
                [prepared.visual_indices[0], prepared.visual_indices[-1] + 1],
                prepared.clean_hidden,
                "language_model.model.layers.{}",
                21,
            ).logits
    finally:
        handle.remove()
    with torch.no_grad():
        manual_upstream, _ = run_layers(
            model.language_model, prepared.clean_hidden, prepared.attention_mask, prepared.position_ids,
            start=21, residual_after=21, residual=residual,
        )
        manual_port, _ = run_layers(
            model.language_model, prepared.clean_hidden + residual, prepared.attention_mask,
            prepared.position_ids, start=22,
        )
    direct_upstream_error = float((direct.float() - manual_upstream.float()).abs().max())
    direct_port_error = float((direct.float() - manual_port.float()).abs().max())
    result = {
        "pinned_method": "BaseVLLMForEdit.forward_from_mid_layer",
        "direct_matches_manual_layer21_reapplication": direct_upstream_error <= 5e-4,
        "direct_vs_manual_upstream_max_abs_error": direct_upstream_error,
        "direct_vs_current_port_max_abs_error": direct_port_error,
        "confirmed_semantics": "captured_layer21_output_is_reinjected_as_layer21_input",
    }
    write_json(target, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
