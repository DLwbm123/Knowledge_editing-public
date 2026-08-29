#!/usr/bin/env python3
"""Read-only audit of local M3Bench inputs; does not alter datasets or metadata."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def image_ok(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def atomic_json_dump(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_question_counts(paths: list[str], dataset: str) -> dict:
    rows = []
    for raw_path in paths:
        rows.extend(json.loads(Path(raw_path).read_text(encoding="utf-8")))
    if dataset == "slake":
        qids = [f"xmlab{row['img_id']}_{row['qid']}" for row in rows]
        english_rows = sum(row.get("q_lang") == "en" for row in rows)
    else:
        qids = [str(row["qid"]) for row in rows]
    result = {"all_rows": len(rows), "eligible_rows": len(qids), "duplicate_question_ids": len(qids) - len(set(qids))}
    if dataset == "slake":
        result["english_rows"] = english_rows
    return result


def parse_qa_list(raw: str) -> list[dict]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def metadata_ids(metadata: Path) -> tuple[set[str], set[str], dict, list[dict], list[dict]]:
    slake, vqarad, details = set(), set(), {}
    for path in sorted(metadata.glob("*.csv")):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        details[path.name] = len(rows)
        for row in rows:
            for key, value in row.items():
                if not value:
                    continue
                if key in {"image_id", "image_id_1", "image_id_2", "image_A", "single_image", "multi_image"}:
                    (vqarad if value.lower().endswith((".jpg", ".png", ".jpeg")) else slake).add(value)
    slake_rows = list(csv.DictReader((metadata / "slake_metadata.csv").open(encoding="utf-8-sig")))
    vqarad_rows = list(csv.DictReader((metadata / "vqarad_metadata.csv").open(encoding="utf-8-sig")))
    return slake, vqarad, details, slake_rows, vqarad_rows


def aligned_metadata_questions(rows: list[dict], source_rows: list[dict], dataset: str) -> dict:
    if dataset == "slake":
        source = {(Path(x["img_name"]).parts[0], x["question"], str(x["answer"]).strip()) for x in source_rows}
    else:
        source = {(x["image_name"], x["question"], str(x["answer"]).strip()) for x in source_rows}
    expected = []
    for row in rows:
        for qa in parse_qa_list(row["qa_list"]):
            expected.append((row["image_id"], qa["question"], str(qa["answer"]).strip()))
    missing = [x for x in expected if x not in source]
    qids = [str(qa["qid"]) for row in rows for qa in parse_qa_list(row["qa_list"])]
    return {"metadata_questions": len(expected), "duplicate_metadata_question_ids": len(qids) - len(set(qids)),
            "source_alignment_missing": len(missing), "missing_examples": missing[:20]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=Path("configs/data_paths.yaml"))
    parser.add_argument("--report", type=Path, default=Path("outputs/logs/data_audit.json"))
    args = parser.parse_args()
    config = load_yaml(args.paths)
    metadata = Path(config["metadata"])
    slake_ids, vqarad_ids, metadata_rows, slake_metadata_rows, vqarad_metadata_rows = metadata_ids(metadata)
    slake_source = [x for split in config["slake"]["splits"] for x in json.loads(Path(split).read_text(encoding="utf-8"))]
    vqarad_source = json.loads(Path(config["vqarad"]["questions"]).read_text(encoding="utf-8"))
    slake_root = Path(config["slake"]["images"])
    vqarad_root = Path(config["vqarad"]["images"])
    missing_slake = sorted(x for x in slake_ids if not (slake_root / x / "source.jpg").is_file())
    missing_vqarad = sorted(x for x in vqarad_ids if not (vqarad_root / x).is_file())
    readable_slake = sum(image_ok(slake_root / x / "source.jpg") for x in slake_ids if x not in missing_slake)
    readable_vqarad = sum(image_ok(vqarad_root / x) for x in vqarad_ids if x not in missing_vqarad)
    report = {
        "read_only": True,
        "metadata_rows": metadata_rows,
        "slake": {
            "metadata_image_ids": len(slake_ids), "missing_image_ids": missing_slake,
            "readable_images": readable_slake,
            "source_questions": source_question_counts(config["slake"]["splits"], "slake"),
            "question_id_alignment": aligned_metadata_questions(slake_metadata_rows, slake_source, "slake"),
        },
        "vqarad": {
            "metadata_image_ids": len(vqarad_ids), "missing_image_ids": missing_vqarad,
            "readable_images": readable_vqarad,
            "source_questions": source_question_counts([config["vqarad"]["questions"]], "vqarad"),
            "question_id_alignment": aligned_metadata_questions(vqarad_metadata_rows, vqarad_source, "vqarad"),
        },
        "padchest_gr": {"status": "INSPECT_ONLY", "root_exists": Path(config["padchest_gr"]["root"]).exists()},
    }
    atomic_json_dump(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
