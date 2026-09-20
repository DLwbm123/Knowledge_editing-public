"""CPU-injected R2-A2 control flow; never deployed to an active worker."""
from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
import json

class ContractError(RuntimeError): pass

def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def validate_slots(slots):
    seen=set(); grouped={}
    for ordinal, row in enumerate(slots, 1):
        t, expert, sid = row.get("t"), row.get("expert"), str(row.get("slot_id", f"slot-{ordinal:03d}"))
        if isinstance(t, bool) or isinstance(expert, bool) or not isinstance(t, int) or not isinstance(expert, int):
            raise ContractError("INVALID_SLOT_TYPES")
        if t < 1 or expert < 1 or expert > t: raise ContractError("INVALID_SLOT_RANGE")
        if sid in seen: raise ContractError("DUPLICATE_SLOT_ID")
        seen.add(sid); grouped.setdefault(t, []).append(dict(row, slot_id=sid, ordinal=ordinal))
    return grouped

@dataclass
class EventStore:
    events: dict = field(default_factory=dict)
    consumers: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)

    def transition(self, event_id, state, **fields):
        old=self.events.get(event_id, {})
        allowed={None:"PREPARED", "PREPARED":"WRITTEN", "WRITTEN":"EVALUATED", "EVALUATED":"RELEASED", "STOPPED":"STOPPED"}
        if old.get("state")=="RELEASED": raise ContractError("RELEASED_REWIND")
        if old.get("state") not in (None, state) and allowed.get(old.get("state")) != state and not (old.get("state")=="EVALUATED" and state=="STOPPED"):
            raise ContractError("INVALID_EVENT_TRANSITION")
        if "version" in old and "version" in fields and old["version"] != fields["version"]:
            raise ContractError("IMMUTABLE_VERSION_CHANGED")
        cur=dict(old, event_id=event_id, state=state, **fields)
        self.events[event_id]=cur
        self.history.setdefault(event_id, []).append(dict(cur))

class CPUWorkerR2:
    def __init__(self, plan, engine, store=None, *, run_id="r2a2-cpu"):
        self.grouped=validate_slots(plan["slots"]); self.engine=engine; self.store=store or EventStore(); self.run_id=run_id
        self.prefixes=sorted(self.grouped); self.plan=plan

    def _consumer(self, arm, prefix, event_id, input_fp, state_id):
        return digest(dict(run=self.run_id, arm=arm, prefix=prefix, event=event_id, input=input_fp, state=state_id))

    def _teacher(self, slot, role, teachers, prefix):
        state=(slot.get("roles") or {}).get(role, {}).get("state", "inactive")
        if state in ("inactive", "masked"): return None
        if state != "active": raise ContractError("INVALID_ROLE_STATE")
        ref=(slot.get("roles") or {}).get(role, {})
        key=(ref.get("input_fp"), ref.get("version")); item=teachers.get(key)
        if item is None or item.get("available_from", 10**9) > prefix: raise ContractError("MISSING_EXACT_TEACHER")
        if item.get("input_fp") != key[0] or item.get("version") != key[1]: raise ContractError("TEACHER_BINDING_MISMATCH")
        return item

    def train_initial(self, slot, *, teachers, prefix):
        """Invoke the same mask/teacher gate for the initial 320-step path."""
        active_u=self._teacher(slot, "U_fit", teachers, prefix)
        role=(slot.get("roles") or {}).get("U_fit", {})
        masked=role.get("state")=="masked"
        return self.engine.update(slot, active_u, masked)

    def run(self, arm, native_inputs, judges, *, teachers, fail_after=None):
        completed=[]; writes=0
        for prefix in self.prefixes:
            for slot in self.grouped[prefix]:
                event_id=f"{arm}:p{prefix}:{slot['slot_id']}"
                current=self.store.events.get(event_id)
                if current and current.get("state")=="RELEASED":
                    continue
                # Validate bindings before any write.
                active_u=self._teacher(slot, "U_fit", teachers, prefix)
                role=(slot.get("roles") or {}).get("U_fit", {})
                masked=role.get("state")=="masked"
                if masked and active_u is not None: raise ContractError("MASKED_TEACHER_LOOKUP")
                if current and current.get("state")=="STOPPED":
                    raise ContractError("EVENT_STOPPED")
                if current and current.get("state")=="EVALUATED":
                    verdicts=current.get("verdicts", [])
                    if any(v is not True for _,v in verdicts): raise ContractError("NATIVE_RELEASE_BARRIER")
                    self.store.transition(event_id, "RELEASED", verdicts=verdicts)
                    completed.append(event_id); continue
                if not current:
                    self.store.transition(event_id, "PREPARED", prefix=prefix, expert=slot["expert"], slot_id=slot["slot_id"])
                    version=self.engine.write(slot, active_u, masked)
                    writes += 1
                    self.store.transition(event_id, "WRITTEN", prefix=prefix, expert=slot["expert"], slot_id=slot["slot_id"], version=version)
                else:
                    version=current["version"]
                native_results=[]
                for inp in native_inputs:
                    state_id=self.engine.execution_state_id(slot, version, inp)
                    cid=self._consumer(arm, prefix, event_id, inp["input_fp"], state_id)
                    out=self.engine.generate(inp, slot, version)
                    self.store.consumers[cid]=dict(event_id=event_id, judge_key=self.engine.judge_key(inp, slot, version), output=out, state_id=state_id)
                    prev=self.store.states.get((inp["input_fp"], state_id))
                    if prev is not None and prev != out: raise ContractError("EXECUTION_STATE_INCONSISTENT")
                    self.store.states[(inp["input_fp"], state_id)]=out
                    native_results.append((cid, out))
                verdicts=[]
                for cid, out in native_results:
                    if cid not in judges: raise ContractError("NATIVE_VERDICT_PENDING")
                    verdicts.append((cid, judges[cid]))
                self.store.transition(event_id, "EVALUATED", verdicts=verdicts)
                if any(v is not True for _,v in verdicts):
                    self.store.transition(event_id, "STOPPED", reason="NATIVE_SAFETY_OR_CORRECTNESS_FAILED")
                    raise ContractError("NATIVE_RELEASE_BARRIER")
                self.store.transition(event_id, "RELEASED", verdicts=verdicts)
                completed.append(event_id)
                if fail_after is not None and writes >= fail_after:
                    raise RuntimeError("SYNTHETIC_CRASH_AFTER_COMMIT")
        return completed
