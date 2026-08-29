#!/usr/bin/env python3
"""Gate A plus deterministic-repeat check; runs no edits and writes one auditable record."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from m3bench_repro.inference import LlavaMedAdapter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--vision-tower", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--question", required=True)
    p.add_argument("--generation", type=Path, default=Path("configs/generation.yaml"))
    p.add_argument("--output", type=Path, default=Path("outputs/logs/llava_gate_a.json"))
    args = p.parse_args()
    generation = yaml.safe_load(args.generation.read_text())
    config_hash = hashlib.sha256(args.generation.read_bytes()).hexdigest()
    adapter = LlavaMedAdapter(args.model, args.vision_tower)
    adapter.load()
    a = adapter.generate(args.image, args.question, generation)
    b = adapter.generate(args.image, args.question, generation)
    result = {"gate": "A", "answer_1": a, "answer_2": b, "exactly_deterministic": a == b,
              "generation_config_hash": config_hash, "timestamp": datetime.now(timezone.utc).isoformat(),
              "editable_mlp_modules": len(adapter.get_editable_mlp_modules()), "base_checksum": adapter.base_checksum()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(args.output)
    print(json.dumps(result, indent=2))
    if not result["exactly_deterministic"]:
        raise SystemExit("determinism gate failed")


if __name__ == "__main__":
    main()
