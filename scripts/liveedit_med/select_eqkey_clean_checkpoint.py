#!/usr/bin/env python3
"""Freeze strict-source checkpoint selection on purged validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from methods.liveedit_med.eqkey_clean_fast import PROTOCOL


EXPECTED_STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3200)
ROLES = ("native", "textual", "visual", "paired")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("protocol") != PROTOCOL or value.get("split") != "validation":
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:checkpoint_result")
    primary = value["repository_sizes"]["32"]
    return {
        "step": int(value["step"]), "result_path": str(path.resolve()), "result_sha256": sha256_file(path),
        "family_ids_hash": hashlib.sha256(json.dumps([row["family_id"] for row in value["rows"]],
                                                       separators=(",", ":")).encode()).hexdigest(),
        "routed_native": int(primary["routed"]["native"]),
        "routed_generality": sum(int(primary["routed"][role]) for role in ("textual", "visual", "paired")),
        "routed": {role: int(primary["routed"][role]) for role in ROLES},
        "locality_exact_preservation": int(primary["locality_exact_preservation"]),
        "locality_total": int(primary["locality_total"]),
        "routing_false_positives": int(primary["routing_false_positives"]),
        "target_contaminations": int(primary["target_contaminations"]),
        "clinical_canonical_failures": int(primary["clinical_canonical_failures"]),
        "forced_on": {role: int(value["forced_on"][role]) for role in ROLES},
        "forced_on_total": sum(int(value["forced_on"][role]) for role in ROLES),
        "source_validation_loss": float(value["source_validation_loss"]),
        "diagnostic_repository_sizes": value["repository_sizes"],
    }


def selection_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (-row["routed_native"], -row["routed_generality"],
            -row["locality_exact_preservation"], row["routing_false_positives"],
            row["target_contaminations"], -row["forced_on_total"],
            row["source_validation_loss"], row["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    rows = [summary(path) for path in args.result]
    if tuple(sorted(row["step"] for row in rows)) != EXPECTED_STEPS:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:checkpoint_set")
    if len({row["family_ids_hash"] for row in rows}) != 1:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:validation_panel_drift")
    selected = min(rows, key=selection_key)
    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "checkpoint_results.jsonl").open("x") as handle:
        for row in sorted(rows, key=lambda value: value["step"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    selection = {
        "protocol": PROTOCOL,
        "label": "POST_PURGE_CHECKPOINT_SELECTION__NO_TEST_LEAKAGE",
        "selected_step": selected["step"], "selected_result_path": selected["result_path"],
        "selected_result_sha256": selected["result_sha256"],
        "selection_key": list(selection_key(selected)), "selected_metrics": selected,
        "heldout_loaded_or_used": False, "record953_loaded_or_used": False,
        "sealed_blind_loaded_or_used": False,
    }
    (args.out_dir / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "CHECKPOINT_RESELECTION_REPORT.md").write_text(
        "# Checkpoint Reselection Report\n\n"
        "Selection: `POST_PURGE_CHECKPOINT_SELECTION__NO_TEST_LEAKAGE`\n\n"
        f"- Selected strict-source checkpoint: **{selected['step']}**\n"
        f"- Routed native (repo 32): **{selected['routed_native']}**\n"
        f"- Routed textual+visual+paired (repo 32): **{selected['routed_generality']}**\n"
        f"- Exact locality: **{selected['locality_exact_preservation']}/{selected['locality_total']}**\n"
        f"- Routing false positives: **{selected['routing_false_positives']}**\n"
        f"- Target contaminations: **{selected['target_contaminations']}**\n"
        f"- Forced-on total: **{selected['forced_on_total']}**\n"
        f"- Source validation loss: **{selected['source_validation_loss']:.8f}**\n"
        "- Held-out used for selection: **No**\n"
        "- Record 953 used: **No**\n"
        "- Sealed blind used: **No**\n"
    )
    print(json.dumps({"status": selection["label"], "selected_step": selected["step"]}, sort_keys=True))


if __name__ == "__main__":
    main()
