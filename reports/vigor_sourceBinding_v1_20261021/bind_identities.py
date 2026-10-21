"""Metadata-only VQA-RAD identity join. No images, answers, or model calls.

Inputs: original JSON list and dataset-filtered identity-role list.
Only qid/image_name/phrase_type and dataset/image_id/roles are used.
Print anonymous counts; never print joined identities or QA.
"""
import argparse
import json
from pathlib import Path


def summarize(original, role_records):
    by_qid = {}
    for row in original:
        qid = str(row["qid"])
        if qid in by_qid:
            raise ValueError("original qid is not unique")
        name = row["image_name"]
        if Path(name).name != name or not Path(name).suffix:
            raise ValueError("unexpected image-name format")
        if row["phrase_type"] not in {"freeform", "para", "test_freeform", "test_para"}:
            raise ValueError("unrecognized split convention")
        by_qid[qid] = row
    groups = {}
    seen = set()
    for role in role_records:
        if role["dataset"] != "VQA-RAD":
            continue
        qid = str(role["qid"])
        if qid in seen or qid not in by_qid:
            raise ValueError("role qid duplicate or unmatched")
        seen.add(qid)
        row = by_qid[qid]
        if Path(row["image_name"]).stem != role["image_id"]:
            raise ValueError("qid agrees but image identity does not")
        g = groups.setdefault(role["image_id"], {"names": set(), "roles": set(), "test": False, "train": False, "rows": 0})
        g["names"].add(row["image_name"])
        g["roles"].update(role["roles"])
        g["test"] |= row["phrase_type"].startswith("test_")
        g["train"] |= not row["phrase_type"].startswith("test_")
        g["rows"] += 1
    if seen != set(by_qid) or any(len(g["names"]) != 1 for g in groups.values()):
        raise ValueError("join incomplete or stem collision")
    reserved = {"FORMAL_OR_RESERVED_SOURCE", "CAL_CHECK_SUPPORT_RESERVED"}
    train_only = [g for g in groups.values() if g["train"] and not g["test"]]
    remaining = [g for g in train_only if not (g["roles"] & reserved) and "FROZEN_FIT_SOURCE" not in g["roles"]]
    return {
        "joined_rows": len(seen), "image_groups": len(groups),
        "qid_missing_or_duplicate": 0, "image_mismatch": 0, "stem_collision": 0,
        "non_test_only_images": len(train_only),
        "non_test_only_reserved_images": sum(bool(g["roles"] & reserved) for g in train_only),
        "non_test_only_fit_images": sum("FROZEN_FIT_SOURCE" in g["roles"] for g in train_only),
        "remaining_candidate_images": len(remaining),
        "remaining_candidate_rows": sum(g["rows"] for g in remaining),
        "purpose_admitted": False, "patient_exclusion_certified": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original_json")
    p.add_argument("filtered_role_json")
    args = p.parse_args()
    print(json.dumps(summarize(json.loads(Path(args.original_json).read_text()),
                               json.loads(Path(args.filtered_role_json).read_text())), indent=2))
