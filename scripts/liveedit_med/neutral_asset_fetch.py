#!/usr/bin/env python3
"""Fetch one pinned public model snapshot into the shared remote-home store."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "llava-hf/llava-1.5-7b-hf"
REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
DESTINATION = Path("/remote-home/wangbomin/models/llava_15_7b_hf_b234b804")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=DESTINATION,
        max_workers=4,
        resume_download=True,
    )
    rows = []
    for path in sorted(item for item in DESTINATION.rglob("*") if item.is_file()
                       and ".cache" not in item.parts
                       and path_name(item) != "download_manifest.json"):
        rows.append({
            "path": str(path.relative_to(DESTINATION)),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "status": "COMPLETE",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "destination": str(DESTINATION),
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = DESTINATION / "download_manifest.json"
    temporary = DESTINATION / "download_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def path_name(path: Path) -> str:
    return path.name


if __name__ == "__main__":
    main()
