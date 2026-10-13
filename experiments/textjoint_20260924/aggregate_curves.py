"""Export numeric stage/step medians without edit IDs or private examples."""
import csv
import json
import os
from pathlib import Path
from statistics import median


ROOT = Path(os.environ["RUN_ROOT"])
FIELDS = (
    "native_ce", "fit_ce", "U_kl", "grad_norm",
    "weighted_native_gradient_norm", "weighted_fit_gradient_norm",
    "weighted_U_gradient_norm",
)
STAGES = ("native_CP", "A2", "CP_W0", "C_NO_H")


def main():
    rows = []
    for arm in ("B0", "P", "E", "U", "EU", "PEU"):
        curves = [json.loads(p.read_text()) for p in sorted((ROOT / "runs/s20260924" / arm).glob("e*/TRAINING_CURVES.json"))]
        assert len(curves) == 12, (arm, len(curves))
        for stage in STAGES:
            by_step = {}
            for curve in curves:
                for sample in curve[stage]:
                    by_step.setdefault(sample["step"], []).append(sample)
            for step, samples in sorted(by_step.items()):
                if len(samples) < 5:
                    continue
                row = {"arm": arm, "stage": stage, "step": step, "edits": len(samples)}
                for field in FIELDS:
                    values = [x[field] for x in samples if isinstance(x.get(field), (int, float))]
                    row[field + "_median"] = median(values) if values else ""
                rows.append(row)
    out = ROOT / "public/CURVES_AGGREGATE.csv"
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), out.stat().st_size)


if __name__ == "__main__":
    main()
