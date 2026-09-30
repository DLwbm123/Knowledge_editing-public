"""Private role contracts, temporal memory, accounting and safe artifact cleanup."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import torch
from torch import Tensor

FIT = {"EDIT_FIT", "GEN_FIT", "VIS_PAIR_FIT", "PROTECT_BG_FIT", "PROTECT_NEAR_FIT", "PAST_EDIT_MEMORY"}
EVALUATION = {"DEV_CAL", "REG_EXPOSED", "TEST_CONFIRM"}
SMOKE = "SMOKE_MECHANICAL"
FIELDS = {"id", "source_group", "patient_group", "image_hash", "question_hash", "answer_hash",
          "fact_family", "role", "permission_basis", "available_at_edit_index", "scope_evidence",
          "annotation_source", "ever_developed", "ever_scored", "teacher_reference", "source_split"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def audit_data(rows: list[dict[str, Any]], *, edit_index: int,
               test_ids: set[str], expected_model: str | None) -> dict[str, Any]:
    """Reject unauthorized roles; findings never manufacture clinical counterexamples."""
    findings = []
    def flag(row: dict[str, Any], reason: str) -> None:
        findings.append(dict(id=row.get("id"), reason=reason))
    for row in rows:
        if not FIELDS <= row.keys():
            flag(row, "MISSING_CONTRACT_FIELDS"); continue
        role = row["role"]
        if role not in FIT | EVALUATION | {SMOKE}:
            flag(row, "UNKNOWN_ROLE")
        if role == SMOKE:
            if row["source_split"] != "train" or row["id"] in test_ids or row.get("evaluation_origin") or row.get("original_role") in EVALUATION or row["ever_developed"] or row["ever_scored"]:
                flag(row, "EVALUATION_TO_SMOKE")
            if not row["permission_basis"] or row.get("mechanical_permission_verified") is not True:
                flag(row, "UNVERIFIED_SMOKE_PERMISSION")
        if role in FIT:
            if row.get("original_role") == SMOKE or row.get("mechanical_origin"):
                flag(row, "SMOKE_TO_SCIENTIFIC_ROLE")
            if row["id"] in test_ids or row["source_split"] in {"test", "validation", "heldout"} or row.get("evaluation_origin", False):
                flag(row, "EVALUATION_TO_FIT")
            if not row["permission_basis"] or row.get("fit_permission_verified") is not True:
                flag(row, "UNVERIFIED_FIT_PERMISSION")
            if not isinstance(row["available_at_edit_index"], int):
                flag(row, "UNKNOWN_INFORMATION_AVAILABILITY")
            elif row["available_at_edit_index"] > edit_index:
                flag(row, "FUTURE_INFORMATION")
            if role == "PROTECT_NEAR_FIT" and row["scope_evidence"] != "VERIFIED_OUT_OF_SCOPE":
                flag(row, "NEAR_SCOPE_UNKNOWN")
        if role == "TEST_CONFIRM" and (row["ever_developed"] or row["ever_scored"]):
            flag(row, "EXPOSED_CONFIRMATION")
        if role == "VIS_PAIR_FIT":
            pair = row.get("pair", {})
            required = {"question_equivalence_verified", "both_labels_independent", "incompatible_answers_verified",
                        "negative_out_of_scope_verified", "negative_base_correct_bound"}
            if not all(pair.get(key) is True for key in required) or not pair.get("evidence_reference"):
                flag(row, "UNQUALIFIED_VISUAL_PAIR")
        teacher = row["teacher_reference"]
        if teacher:
            if expected_model is None or teacher.get("model") != expected_model:
                flag(row, "WRONG_MODEL_TEACHER")
            if row.get("corrected_history") and teacher.get("kind") == "BASE":
                flag(row, "HISTORY_TEACHER_CONFLICT")
            if teacher.get("prefix_hash") != row.get("prefix_hash"):
                flag(row, "PREFIX_MISMATCH")
    identities: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = row.get("image_hash"), row.get("question_hash")
        if all(key):
            identities.setdefault(key, []).append(row)
    exact_duplicates, conflicts, role_crossings, same_sources, near_duplicates = [], [], [], [], []
    for i, left in enumerate(rows):
        for right in rows[i+1:]:
            a, b = set(left.get("question_shingles", [])), set(right.get("question_shingles", []))
            if a and b and len(a & b) / len(a | b) >= .8 and left.get("question_hash") != right.get("question_hash"):
                near_duplicates.append([left["id"],right["id"]])
    for key, members in identities.items():
        if len(members) > 1:
            if SMOKE in {r.get("role") for r in members} and {r.get("role") for r in members} & (FIT | EVALUATION):
                for row in members:
                    flag(row, "SMOKE_TO_SCIENTIFIC_ROLE")
            exact_duplicates.append([r["id"] for r in members])
            if len({r.get("answer_hash") for r in members}) > 1:
                conflicts.append([r["id"] for r in members])
                for row in members:
                    if not row.get("supersedes"):
                        flag(row, "UNRESOLVED_FACT_CONFLICT")
            if {r.get("role") for r in members} & FIT and {r.get("role") for r in members} & EVALUATION:
                role_crossings.append([r["id"] for r in members])
                if not all(r.get("benchmark_same_edit_sharing_verified") for r in members):
                    flag(members[0], "UNAUTHORIZED_ROLE_CROSSING")
    # Group separation applies to development vs confirmation, not every same-image probe.
    development = [r for r in rows if r.get("role") != "TEST_CONFIRM" and not (
        r.get("selection_cohort") == "CONFIRM" and r.get("benchmark_same_edit_sharing_verified") is True
        and r.get("fit_permission_verified") is True and r.get("role") in FIT)]
    for row in rows:
        if row.get("role") == "TEST_CONFIRM":
            for field in ("source_group", "patient_group", "fact_family"):
                if row.get(field) and row[field] in {r.get(field) for r in development}:
                    flag(row, "CONFIRMATION_GROUP_OVERLAP:" + field)
    by_source: dict[str, list[str]] = {}
    for row in rows:
        if row.get("source_group"):
            by_source.setdefault(row["source_group"], []).append(row["id"])
    same_sources = [ids for ids in by_source.values() if len(ids) > 1]
    invalid = {f["id"] for f in findings}
    eligible = [r for r in rows if r.get("id") not in invalid and r.get("role") in FIT]
    return dict(status="FAIL" if findings else ("BLOCKED" if not rows else "PASS"),
                rows=len(rows), findings=findings, role_counts=dict(Counter(r.get("role") for r in rows)),
                eligible_counts=dict(Counter(r["role"] for r in eligible)),
                exact_duplicates=exact_duplicates, fact_conflicts=conflicts, role_crossings=role_crossings,
                same_source_groups=same_sources,
                near_duplicates=near_duplicates, near_duplicate_rule="hashed question bigram Jaccard >= .8; lexical signal only",
                image_near_duplicates="PENDING_PIXEL_METADATA; same-source groups counted separately",
                valid_ids=[r["id"] for r in eligible])


def method_eligibility(audit: dict[str, Any]) -> dict[str, str]:
    counts = audit["eligible_counts"]
    common = all(counts.get(role, 0) > 0 for role in ("EDIT_FIT", "GEN_FIT", "PROTECT_BG_FIT", "PROTECT_NEAR_FIT"))
    status = "PENDING_NATIVE_CHECK" if common else "BLOCKED"
    return {"W_FT": status, "W_EUCLIDEAN_QP": status, "W_KEY_QP": status,
            "W_FUNCTIONAL_QP": status,
            "W_EVIDENCE_QP": status if common and counts.get("VIS_PAIR_FIT", 0) else "BLOCKED"}


def validate_cache(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    keys = {"model", "weight_version", "input", "prefix", "mask", "dtype", "backend", "config", "teacher_version"}
    if set(actual) != keys or set(expected) != keys or actual != expected or not all(actual.values()):
        raise ValueError("cache binding mismatch")


@dataclass
class MemoryEntry:
    id: str
    source: str
    family: str
    available_at_edit_index: int
    predictor_positions: int
    teacher: Tensor
    payload: dict[str, Any]
    binding: dict[str, str]

    @property
    def bytes(self) -> int:
        return self.teacher.numel() * self.teacher.element_size() + len(json.dumps(self.payload).encode()) + len(json.dumps(self.binding).encode())


class HistoryMemory:
    """Bounded deterministic source/family-stratified priority retention."""
    def __init__(self, max_inputs: int = 64, max_positions: int = 4096, max_bytes: int = 2 * 1024**3):
        if min(max_inputs, max_positions, max_bytes) <= 0:
            raise ValueError("finite history capacity required")
        self.limits = max_inputs, max_positions, max_bytes
        self.entries: dict[str, MemoryEntry] = {}
        self.evictions: list[dict[str, str]] = []

    def add(self, entry: MemoryEntry, *, current_index: int, accepted: bool,
            supersedes: tuple[str, ...] = ()) -> None:
        if not accepted or entry.available_at_edit_index > current_index or entry.predictor_positions <= 0:
            raise ValueError("unaccepted/future history")
        if entry.payload.get("role") != "PAST_EDIT_MEMORY" or entry.payload.get("fit_permission_verified") is not True:
            raise ValueError("illegal memory support")
        if entry.payload.get("teacher_kind") != "ACCEPTED" or entry.teacher.requires_grad:
            raise ValueError("corrected history must retain accepted detached teacher")
        if not torch.isfinite(entry.teacher).all() or (entry.teacher < 0).any() or not torch.allclose(entry.teacher.sum(-1), torch.ones_like(entry.teacher[..., 0])):
            raise ValueError("invalid accepted teacher")
        validate_cache(entry.binding, entry.binding)
        if any(key not in self.entries for key in supersedes):
            raise ValueError("unknown superseded history")
        same_family = {k for k, e in self.entries.items() if e.family == entry.family}
        if entry.payload.get("conflict", False) and not same_family <= set(supersedes):
            raise ValueError("explicit supersedes required")
        if entry.bytes > self.limits[2] or entry.predictor_positions > self.limits[1]:
            raise ValueError("one entry exceeds history capacity")
        for key in supersedes:
            del self.entries[key]
            self.evictions.append(dict(id=key, reason="SUPERSEDED"))
        self.entries[entry.id] = MemoryEntry(entry.id, entry.source, entry.family, entry.available_at_edit_index,
            entry.predictor_positions, entry.teacher.detach().clone(), dict(entry.payload), dict(entry.binding))
        strata: dict[tuple[str, str], list[MemoryEntry]] = {}
        for e in self.entries.values():
            strata.setdefault((e.source, e.family), []).append(e)
        queues = [sorted(values, key=lambda e: digest([e.source, e.family, e.id]))
                  for _, values in sorted(strata.items())]
        order = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
        retained: dict[str, MemoryEntry] = {}
        positions = used_bytes = 0
        for e in order:
            if len(retained) < self.limits[0] and positions + e.predictor_positions <= self.limits[1] and used_bytes + e.bytes <= self.limits[2]:
                retained[e.id] = e; positions += e.predictor_positions; used_bytes += e.bytes
            else:
                self.evictions.append(dict(id=e.id, reason="FIXED_CAPACITY"))
        self.entries = retained

    def at(self, index: int) -> list[MemoryEntry]:
        if any(e.available_at_edit_index > index for e in self.entries.values()):
            raise ValueError("future memory access")
        return list(self.entries.values())

    def accounting(self) -> dict[str, int]:
        return dict(inputs=len(self.entries), sources=len({e.source for e in self.entries.values()}),
                    positions=sum(e.predictor_positions for e in self.entries.values()),
                    bytes=sum(e.bytes for e in self.entries.values()))


class AttemptLedger:
    """One terminal record per intended edit; missing Judge remains in denominator."""
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, edit_id: str, status: str, *, judged: bool = False, correct: bool | None = None,
               role: str = "EDIT_FIT", result_kind: str = "SCIENTIFIC") -> None:
        if role not in FIT or result_kind != "SCIENTIFIC" or status == "MECHANICAL_NATIVE_VALIDATION":
            raise ValueError("mechanical outputs are excluded from scientific denominators")
        if any(r["id"] == edit_id for r in self.rows) or (judged and type(correct) is not bool) or (not judged and correct is not None):
            raise ValueError("duplicate or incomplete verdict")
        self.rows.append(dict(id=edit_id, status=status, judged=judged, correct=correct))

    def summary(self) -> dict[str, Any]:
        n = len(self.rows); scored = sum(r["judged"] for r in self.rows)
        correct = sum(r["correct"] is True for r in self.rows)
        return dict(intended=n, scored=scored, missing=n-scored, status_counts=dict(Counter(r["status"] for r in self.rows)),
                    correctness_lower=correct/n if n else None,
                    correctness_upper=(correct+n-scored)/n if n else None)


def cleanup(run_root: Path, files: list[Path], consumers: dict[str, list[str]], *,
            completed: set[str], evidence_complete: bool, dry_run: bool = True) -> list[str]:
    """Explicit files only, containment/symlink/dependency checked BEFORE any deletion."""
    root = run_root.resolve(strict=True)
    approved = []
    if not evidence_complete:
        raise ValueError("generation/reproducibility evidence incomplete")
    for file in files:
        resolved = file.resolve(strict=True)
        if file.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("cleanup outside own RUN_ROOT or symlink")
        key = str(resolved.relative_to(root))
        if key not in consumers or not set(consumers[key]) <= completed or "active_resume" in consumers[key]:
            raise ValueError("unresolved consumers/resume state")
        approved.append(resolved)
    if not dry_run:
        for file in approved:
            file.unlink()
    return [str(p.relative_to(root)) for p in approved]


def enforce_storage(root: Path, limit_bytes: int, prospective_bytes: int) -> int:
    if limit_bytes <= 0 or prospective_bytes < 0:
        raise ValueError("invalid storage budget")
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    if used + prospective_bytes > limit_bytes:
        raise ValueError("RUN_ROOT storage hard cap exceeded")
    return used
