from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.liveedit_med.router_r1_schema import (
    LocalitySchemaError,
    locality_identity,
    resolve_locality_source,
)


def locality(eqkey: str = "eq-a") -> dict:
    return {
        "category": "image_locality",
        "role": "locality",
        "eqkey": eqkey,
        "source_hash": "source-a",
        "processed_pixel_tensor_sha256": "pixel-a",
        "image": "/frozen/image.png",
        "prompt": "What is shown?",
        "target": "unchanged answer",
        "clean_generation": {"token_ids": [1, 2, 3]},
    }


def frozen_cache(root: Path, name: str, item: dict) -> Path:
    directory = root / name
    directory.mkdir()
    path = directory / "record.safetensors"
    path.write_bytes(b"frozen-cache-reference")
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    tensor_hashes = {key: f"hash-{key}" for key in (
        "image_locality__hidden", "image_locality__labels",
        "image_locality__answer_mask", "image_locality__question_mask",
        "image_locality__vision_mask", "image_locality__base_answer_logits")}
    (directory / "manifest.json").write_text(json.dumps({"records": [{
        "file_path": str(path.resolve()), "file_sha256": file_hash,
        "inputs": [item], "tensor_hashes": tensor_hashes,
    }]}))
    return path


def canonical_family(path: Path, item: dict | None = None) -> dict:
    current = locality() if item is None else item
    return {"family_id": "family-a", "canonical_locality": {"image_locality": current},
            "canonical_views": [{"cache_file_path": str(path)}]}


def test_canonical_locality_only_resolves_frozen_cache(tmp_path: Path):
    item = locality(); path = frozen_cache(tmp_path, "canonical", item)
    resolved = resolve_locality_source(canonical_family(path, item))
    assert resolved.cache_file_path == path.resolve()
    assert resolved.identity == locality_identity(item)
    assert resolved.resolution == "canonical_locality_frozen_cache_match"


def test_legacy_cache_file_path_only_remains_supported(tmp_path: Path):
    path = frozen_cache(tmp_path, "legacy", locality())
    resolved = resolve_locality_source({"family_id": "legacy-family",
                                        "cache_file_path": str(path)})
    assert resolved.cache_file_path == path.resolve()
    assert resolved.resolution == "legacy_cache_file_path"


def test_dual_consistent_representation_uses_canonical_precedence(tmp_path: Path):
    item = locality()
    canonical_path = frozen_cache(tmp_path, "canonical", item)
    legacy_path = frozen_cache(tmp_path, "legacy", item)
    family = canonical_family(canonical_path, item)
    family["cache_file_path"] = str(legacy_path)
    resolved = resolve_locality_source(family)
    assert resolved.cache_file_path == canonical_path.resolve()
    assert resolved.resolution == "canonical_locality_precedes_consistent_legacy"


def test_dual_conflicting_representation_fails_closed(tmp_path: Path):
    item = locality()
    canonical_path = frozen_cache(tmp_path, "canonical", item)
    legacy_path = frozen_cache(tmp_path, "legacy", locality("eq-conflict"))
    family = canonical_family(canonical_path, item)
    family["cache_file_path"] = str(legacy_path)
    with pytest.raises(LocalitySchemaError, match="DUAL_LOCALITY_IDENTITY_CONFLICT:family-a"):
        resolve_locality_source(family)


def test_missing_schema_reports_family_context():
    with pytest.raises(LocalitySchemaError, match="LOCALITY_SCHEMA_MISSING:family-missing"):
        resolve_locality_source({"family_id": "family-missing"})


def test_eqkey_training_uses_source_training_row_contract():
    root = Path(__file__).resolve().parents[2]
    text = (root / "scripts/liveedit_med/train_eqkey_clean_router_r1.py").read_text()
    assert "from scripts.liveedit_med.train_router_r1 import recursive_hash" in text
    assert "from scripts.liveedit_med.evaluate_router_r1_checkpoint import variant" not in text
    assert "def training_variant(tensors" in text
    assert '"vision": raw["vision_mask"].bool()' in text
    assert '"prompt": raw["question_mask"].bool()' in text
    assert '"answer": raw["answer_mask"].bool()' in text
    assert "def query_keys(modules, row" in text
