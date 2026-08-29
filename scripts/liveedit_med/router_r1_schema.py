"""Deterministic locality-cache reference resolution for EqKey-clean Router-R1.

This module resolves references only.  It never creates samples, tensors, or
cache files, and it deliberately keeps the canonical locality identity
separate from the physical cache location.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


IDENTITY_FIELDS = (
    "category",
    "role",
    "eqkey",
    "source_hash",
    "processed_pixel_tensor_sha256",
    "image",
    "prompt",
    "target",
)
REQUIRED_TENSOR_KEYS = (
    "image_locality__hidden",
    "image_locality__labels",
    "image_locality__answer_mask",
    "image_locality__question_mask",
    "image_locality__vision_mask",
    "image_locality__base_answer_logits",
)


class LocalitySchemaError(RuntimeError):
    """Fail-closed error for an unresolvable or conflicting locality schema."""


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def locality_identity(item: Mapping[str, Any]) -> dict[str, Any]:
    generation = item.get("clean_generation")
    token_ids = generation.get("token_ids") if isinstance(generation, Mapping) else None
    return {
        **{field: item.get(field) for field in IDENTITY_FIELDS},
        "clean_generation_token_ids": token_ids,
    }


@dataclass(frozen=True)
class ResolvedLocalitySource:
    cache_file_path: Path
    cache_file_sha256: str | None
    identity: Mapping[str, Any] | None
    resolution: str

    def audit_dict(self) -> dict[str, Any]:
        return {
            "cache_file_path": str(self.cache_file_path),
            "cache_file_sha256": self.cache_file_sha256,
            "identity_hash": None if self.identity is None else canonical_hash(self.identity),
            "resolution": self.resolution,
        }


def _context(family: Mapping[str, Any]) -> str:
    return str(family.get("family_id") or family.get("canonical_record_id")
               or family.get("record_id") or "UNKNOWN")


@lru_cache(maxsize=None)
def _read_manifest(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def _cache_record(path: Path) -> Mapping[str, Any] | None:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_manifest(manifest_path)
    resolved = str(path.resolve())
    matches = [row for row in manifest.get("records", [])
               if str(Path(row.get("file_path", path.parent / row.get("file", ""))).resolve()) == resolved]
    if len(matches) != 1:
        return None
    return matches[0]


def _cache_locality(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    record = _cache_record(path)
    if record is None:
        return None, None
    tensor_hashes = record.get("tensor_hashes", {})
    if not all(key in tensor_hashes for key in REQUIRED_TENSOR_KEYS):
        return None, record.get("file_sha256")
    matches = [item for item in record.get("inputs", [])
               if item.get("category") == "image_locality"]
    if len(matches) != 1:
        return None, record.get("file_sha256")
    return matches[0], record.get("file_sha256")


def _existing_path(value: Any, context: str) -> Path:
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise LocalitySchemaError(f"ROUTER_R1_LOCALITY_CACHE_MISSING:{context}:{path}")
    return path


def resolve_locality_source(family: Mapping[str, Any]) -> ResolvedLocalitySource:
    """Resolve one frozen image-locality cache without changing its identity.

    Canonical inline metadata has precedence.  Its physical cache reference is
    recovered by matching the metadata against the already-frozen cache shard
    manifests referenced by the family's canonical views.  Legacy direct paths
    remain accepted, but dual representations must agree.
    """
    context = _context(family)
    container = family.get("canonical_locality")
    canonical_item = (container.get("image_locality")
                      if isinstance(container, Mapping) else None)
    top_level_legacy = family.get("cache_file_path")
    direct_item = family if family.get("category") == "image_locality" else None
    if canonical_item is None and isinstance(direct_item, Mapping):
        top_level_legacy = direct_item.get("cache_file_path", top_level_legacy)

    if isinstance(canonical_item, Mapping):
        expected = locality_identity(canonical_item)
        view_paths = sorted({str(view["cache_file_path"])
                             for view in family.get("canonical_views", [])
                             if isinstance(view, Mapping) and view.get("cache_file_path")})
        legacy_value = canonical_item.get("cache_file_path", top_level_legacy)
        candidate_paths = [_existing_path(value, context) for value in view_paths]
        matches: list[tuple[Path, str | None]] = []
        for path in candidate_paths:
            cached_item, file_hash = _cache_locality(path)
            if cached_item is not None and locality_identity(cached_item) == expected:
                matches.append((path, file_hash))
        if len(matches) > 1:
            raise LocalitySchemaError(
                f"ROUTER_R1_CANONICAL_LOCALITY_AMBIGUOUS:{context}:{len(matches)}")

        legacy_match: tuple[Path, str | None] | None = None
        if legacy_value is not None:
            legacy_path = _existing_path(legacy_value, context)
            cached_item, file_hash = _cache_locality(legacy_path)
            if cached_item is None or locality_identity(cached_item) != expected:
                raise LocalitySchemaError(
                    f"ROUTER_R1_DUAL_LOCALITY_IDENTITY_CONFLICT:{context}:{legacy_path}")
            legacy_match = (legacy_path, file_hash)

        if matches:
            selected = matches[0]
            if legacy_match is not None and legacy_match[0] != selected[0]:
                # Both are scientifically identical.  Canonical-view resolution
                # wins deterministically; the legacy path remains only a parity check.
                resolution = "canonical_locality_precedes_consistent_legacy"
            else:
                resolution = "canonical_locality_frozen_cache_match"
            return ResolvedLocalitySource(selected[0], selected[1], expected, resolution)
        if legacy_match is not None:
            return ResolvedLocalitySource(
                legacy_match[0], legacy_match[1], expected,
                "canonical_locality_with_consistent_legacy_reference")
        raise LocalitySchemaError(
            f"ROUTER_R1_CANONICAL_LOCALITY_CACHE_UNRESOLVED:{context}")

    if top_level_legacy is not None:
        path = _existing_path(top_level_legacy, context)
        cached_item, file_hash = _cache_locality(path)
        identity = None if cached_item is None else locality_identity(cached_item)
        return ResolvedLocalitySource(path, file_hash, identity, "legacy_cache_file_path")

    raise LocalitySchemaError(f"ROUTER_R1_LOCALITY_SCHEMA_MISSING:{context}")
