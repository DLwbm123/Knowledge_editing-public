"""CPU-only contract patch for Stage24E R2-A.

This module is isolated from the active worker.  It provides the guards that
the next runtime version must call: list-preserving slots, exact teachers,
explicit mask states, versioned consumers, and a native verdict barrier.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json


class ContractError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def group_slots(slots):
    """Retain every slot and original order; never index by prefix alone."""
    out = defaultdict(list)
    seen = set()
    for ordinal, row in enumerate(slots, 1):
        slot_id = str(row.get("slot_id", f"slot-{ordinal:03d}"))
        if slot_id in seen:
            raise ContractError("DUPLICATE_SLOT_ID")
        seen.add(slot_id)
        if not isinstance(row.get("t"), int) or not isinstance(row.get("expert"), int):
            raise ContractError("INVALID_SLOT")
        if row["expert"] > row["t"]:
            raise ContractError("FUTURE_EXPERT")
        out[row["t"]].append(dict(row, slot_id=slot_id, ordinal=ordinal))
    return dict(out)


def resolve_teacher(state, *, input_fp=None, teacher_version=None, prefix=None,
                    store=None, available_from=None, masked_reference=False):
    """Resolve only an exact teacher; no first-item fallback is permitted."""
    if state == "inactive":
        return None
    if state == "masked":
        if not masked_reference or not isinstance(available_from, int) or available_from > prefix:
            raise ContractError("INVALID_MASK")
        return None
    if state != "active" or not isinstance(prefix, int):
        raise ContractError("INVALID_ROLE_STATE")
    if not all(isinstance(x, str) and x for x in (input_fp, teacher_version)):
        raise ContractError("MISSING_EXACT_TEACHER")
    item = (store or {}).get((input_fp, teacher_version))
    if item is None or item.get("input_fp") != input_fp or item.get("version") != teacher_version:
        raise ContractError("MISSING_EXACT_TEACHER")
    if not isinstance(item.get("available_from"), int) or item["available_from"] > prefix:
        raise ContractError("FUTURE_TEACHER")
    return item


def consumer_id(run_id, arm, prefix, event_id, input_fp, effective_state):
    fields = (run_id, arm, event_id, input_fp, effective_state)
    if not all(isinstance(x, str) and x for x in fields) or not isinstance(prefix, int):
        raise ContractError("INVALID_CONSUMER_ID")
    return digest(dict(run_id=run_id, arm=arm, prefix=prefix, event_id=event_id,
                       input=input_fp, effective_state=effective_state))


class OutputRegistry:
    def __init__(self):
        self._rows = {}

    def observe(self, cid, tokens, judge_text):
        current = (tuple(tokens), judge_text)
        if cid in self._rows and self._rows[cid] != current:
            raise ContractError("EXECUTION_IDENTITY_VIOLATION")
        self._rows[cid] = current


def require_native_verdicts(expected, completed):
    if not expected or any(completed.get(cid, {}).get("status") != "COMPLETE"
                           or type(completed[cid].get("correct")) is not bool
                           for cid in expected):
        raise ContractError("NATIVE_RELEASE_BARRIER")
