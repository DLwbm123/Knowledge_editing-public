#!/usr/bin/env python3
"""Build the validation-positive F/H/T/R cross-table from frozen artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.router_r1_oracle import PROTOCOL, ROLES, fhtr_counts


def flat(values):
    return values[0] if values and isinstance(values[0], list) else values


def write_jsonl(path: Path, rows) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step80", type=Path, required=True)
    parser.add_argument("--step640", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_root / "sample_cross_table"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    summary = {"protocol": PROTOCOL, "diagnostic_only": True,
               "not_for_selection": True, "steps": {}}
    for path, step in ((args.step80, 80), (args.step640, 640)):
        value = json.loads(path.read_text())
        if int(value["step"]) != step or value.get("split") != "validation":
            raise RuntimeError("ROUTER_R1_ORACLE_CROSS_TABLE_INPUT_MISMATCH")
        rows = []
        for family in value["rows"]:
            target = family["family_id"]
            role_by_eqkey = {view["eqkey"]: view["role"] for view in family["views"]}
            for eqkey, routed in family["repositories"]["32"]["routed"].items():
                role = role_by_eqkey[eqkey]
                forced = family["forced_on"][eqkey]
                route = routed["route"]; ids = route.get("candidate_ids", [])
                weights = flat(route.get("final_weights", []))
                hard = target in ids
                top = False
                target_weight = None
                if hard:
                    target_weight = float(weights[ids.index(target)])
                    top = not any(float(weight) > target_weight for candidate, weight in zip(ids, weights)
                                  if candidate != target)
                rows.append({
                    "protocol": PROTOCOL, "step": step, "family_id": target,
                    "eqkey": eqkey, "role": role,
                    "F": bool(forced["match"]["success"]), "H": hard, "T": top,
                    "R": bool(routed["match"]["success"]),
                    "candidate_ids": ids, "candidate_count": len(ids),
                    "target_final_weight": target_weight,
                    "diagnostic_only": True, "not_for_selection": True,
                })
        if len(rows) != 128:
            raise RuntimeError("ROUTER_R1_ORACLE_CROSS_TABLE_COUNT")
        write_jsonl(out / f"step_{step:04d}.jsonl", rows)
        summary["steps"][str(step)] = {
            role: fhtr_counts([row for row in rows if row["role"] == role]) for role in ROLES}
        summary["steps"][str(step)]["overall"] = fhtr_counts(rows)
    with (out / "conditional_retention_summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps({"status": "ROUTER_R1_ORACLE_CROSS_TABLE_COMPLETE",
                      "rows_per_step": 128}, sort_keys=True))


if __name__ == "__main__":
    main()
