"""Bind frozen packets and deduplicate exact responses; never assign semantic labels."""
import importlib.util
import json
from pathlib import Path


def collect(roots):
    unique = {}
    total = packets = 0
    image_bindings = {}
    for version, root in roots:
        root = Path(root)
        spec = importlib.util.spec_from_file_location(f"consumer_{version}", root / "tools/consumers.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        outputs = module.rows(root)
        packet = {r["code"]: r for r in module.read(root / "BLIND_PACKET.json")}
        mapping = module.read(root / "BLIND_MAPPING.json")
        verdicts = module.read(root / "BLIND_VERDICTS.json")
        assert set(packet) == set(mapping) == set(verdicts)
        packets += len(packet)
        seen = set()
        for code, indices in mapping.items():
            for index in indices:
                assert index not in seen
                seen.add(index)
                row = outputs[index]
                q = row["q"]
                assert all(packet[code][k] == q[k] for k in ("question", "reference"))
                assert packet[code]["answer"] == row["answer"]
                identity = (q["source_group"], Path(q["image_path"]).name)
                # A basename alone cannot establish identity across different datasets.
                assert image_bindings.setdefault(identity, q["image_path"]) == q["image_path"]
                key = (*identity, q["question"], q["reference"], row["answer"])
                item = unique.setdefault(key, dict(
                    id=f"D{len(unique)+1:03d}", case_identity=identity,
                    question=q["question"], reference=q["reference"], answer=row["answer"],
                    frozen_label=verdicts[code]["label"], bindings=[]))
                assert item["frozen_label"] == verdicts[code]["label"], "Historical label inconsistency"
                item["bindings"].append(dict(version=version, index=index, code=code,
                    case=row["case"], arm=row["arm"], query_id=q["query_id"],
                    group=q["eval_group"], role=q["relation_role"]))
        assert seen == set(range(len(outputs)))
        total += len(outputs)
    return list(unique.values()), dict(output_bindings=total, packet_rows=packets,
        unique_rows=len(unique), removed_packet_duplicates=packets-len(unique))


def validate_review(rows, reviewed):
    by_id = {r["id"]: r for r in reviewed}
    assert len(by_id) == len(rows)
    for row in rows:
        review = by_id[row["id"]]
        for key in ("case_identity", "question", "reference", "answer", "frozen_label"):
            assert json.dumps(row[key]) == json.dumps(review[key]), key
        assert review["taxonomy"]["status"] == "MANUALLY_REVIEWED"
        assert review["reference_audit"]["status"]
    return {"status": "PASS", "reviewed": len(rows), "labels_changed": 0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs=3, required=True)
    parser.add_argument("--review", required=True)
    args = parser.parse_args()
    rows, counts = collect(zip((30, 31, 32), args.roots))
    print(json.dumps(dict(counts=counts, validation=validate_review(
        rows, json.loads(Path(args.review).read_text()))), indent=2))
