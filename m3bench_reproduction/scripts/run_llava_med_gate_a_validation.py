#!/usr/bin/env python3
"""Three-case, three-repeat native LLaVA-Med Gate A validation (no edits)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from m3bench_repro.inference import LlavaMedAdapter


def write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--vision-tower", required=True, type=Path)
    parser.add_argument("--image-folder", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("manifests/llava_gate_a_three_cases.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = manifest["generation"]
    if config["do_sample"] or config.get("num_beams") != 1:
        raise ValueError("Gate A requires deterministic greedy decoding")
    adapter = LlavaMedAdapter(args.model, args.vision_tower)
    adapter.load()
    records = []
    for case in manifest["cases"]:
        image = args.image_folder / case["image"]
        if not image.is_file():
            raise FileNotFoundError(image)
        prompt_inputs = adapter.prepare_inputs(image, case["question"])
        image_count = int((prompt_inputs["input_ids"] == -200).sum().item())
        image_finite = bool(torch.isfinite(prompt_inputs["images"] if isinstance(prompt_inputs["images"], torch.Tensor) else prompt_inputs["images"][0]).all().item())
        runs = []
        for repeat in range(3):
            result = adapter.generate_with_result(image, case["question"], config)
            runs.append({"repeat": repeat + 1, "raw_token_ids": list(result.raw_token_ids), "answer": result.decoded_text})
        answers = [run["answer"] for run in runs]
        raw_ids = [run["raw_token_ids"] for run in runs]
        records.append({
            **case,
            "image_token_count": image_count,
            "image_tensor_finite": image_finite,
            "runs": runs,
            "all_nonempty": all(bool(answer) for answer in answers),
            "exactly_deterministic_text": len(set(answers)) == 1,
            "exactly_deterministic_raw_ids": len({tuple(ids) for ids in raw_ids}) == 1,
        })
    passed = all(
        record["image_token_count"] == 1
        and record["image_tensor_finite"]
        and record["all_nonempty"]
        and record["exactly_deterministic_text"]
        and record["exactly_deterministic_raw_ids"]
        for record in records
    )
    output = {
        "gate": "A",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_contract": adapter.generation_sequence_contract.value,
        "generation": config,
        "records": records,
        "passed": passed,
        "assertions": {"no_sampling": True, "no_edit_method_executed": True},
    }
    write_json(output, args.output)
    print(json.dumps({"passed": passed, "answers": [[run["answer"] for run in r["runs"]] for r in records]}, ensure_ascii=False))
    if not passed:
        raise SystemExit("LLaVA-Med Gate A validation failed")


if __name__ == "__main__":
    main()
