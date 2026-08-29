#!/usr/bin/env python3
"""Run the frozen T1 calibration queue with neutral process command lines."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from pathlib import Path


ROOT = Path("/remote-home/wangbomin/Knowledge_editing")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / "outputs/liveedit_med_router_r2_minimal_causal_rescue_v1/20260817T081151Z"
PARENT = ROOT / "outputs/liveedit_med_eqkey_clean_fast_confirmation_v1/20260815T122907Z"
OUT = RUN / "r2_t1/calibration_recovery_02"
PREVIOUS = RUN / "r2_t1/calibration_recovery_01"
TRAIN = PARENT / "router_r1/clean_train_family_manifest.json"
HARD = PARENT / "router_r1/hard_negative_cache_manifest.json"
NEGATIVE_PLAN = PARENT / ".runtime_router_r1/hard_negative_plan.json"
EXPERTS = PARENT / ".runtime_router_r1/fixed_experts"
CALIBRATION = RUN / "CALIBRATION_SPLIT_MANIFEST.json"
E0_SELECTION = RUN / "r2_e0/E0_CALIBRATION_SELECTION.json"
TRAINING = RUN / "r2_t1/training_recovery_05"
PYTHON = "/root/anaconda3/bin/python"
GPU_STEPS = {2: (80, 160, 240, 320), 3: (400, 480, 560, 640)}


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def invoke(module_name: str, argv: list[str]) -> None:
    module = __import__(module_name, fromlist=["main"])
    previous = sys.argv
    try:
        sys.argv = [module_name, *argv]
        module.main()
    finally:
        sys.argv = previous


def common(checkpoint: Path, partition: str, gpu: int, out: Path) -> list[str]:
    return ["--checkpoint", str(checkpoint), "--train-manifest", str(TRAIN),
            "--hard-manifest", str(HARD), "--negative-plan", str(NEGATIVE_PLAN),
            "--calibration-manifest", str(CALIBRATION), "--partition", partition,
            "--fixed-experts", str(EXPERTS), "--e0-selection", str(E0_SELECTION),
            "--physical-gpu", str(gpu), "--out", str(out)]


def evaluate_partition(step: int, partition: str, gpu: int, out: Path) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    checkpoint = TRAINING / f"checkpoint_{step:04d}"
    invoke("scripts.liveedit_med.evaluate_router_r2_t1",
           ["worker", *common(checkpoint, partition, gpu, out)])


def gpu_queue(gpu: int, steps: tuple[int, ...]) -> None:
    try:
        for step in steps:
            target = OUT / f"checkpoint_{step:04d}"
            target.mkdir(parents=True, exist_ok=False)
            fit = target / "calibration_fit.jsonl"
            lock = target / "calibration_lock.jsonl"
            evaluate_partition(step, "calibration_fit", gpu, fit)
            evaluate_partition(step, "calibration_lock", gpu, lock)
            from argparse import Namespace
            from scripts.liveedit_med.select_router_r2_t1 import checkpoint
            checkpoint(Namespace(
                calibration_manifest=CALIBRATION,
                checkpoint_manifest=TRAINING / f"checkpoint_{step:04d}/manifest.json",
                fit_shard=[fit], lock_shard=[lock], out=target / "calibration_result.json"))
        write_new(OUT / f"gpu_{gpu}_complete.json", {"gpu": gpu, "steps": list(steps),
                  "status": "COMPLETE"})
    except BaseException as error:
        write_new(OUT / f"gpu_{gpu}_failure.json", {"gpu": gpu, "steps": list(steps),
                  "status": "FAILED", "error_type": type(error).__name__,
                  "error": str(error), "traceback": traceback.format_exc()})
        raise


def parity_worker() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    invoke("scripts.liveedit_med.evaluate_router_r2_t1",
           ["parity", *common(TRAINING / "checkpoint_0080", "calibration_fit", 2,
                              OUT / "T1_RUNTIME_PARITY.json")])


def assert_free_gpus() -> None:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    used = {int(line.split(",")[0].strip()): int(line.split(",")[1].strip())
            for line in output.splitlines() if line.strip()}
    if any(used.get(gpu, 10**9) > 512 for gpu in GPU_STEPS):
        raise RuntimeError(f"ROUTER_R2_T1_GPU_NOT_FREE:{used}")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    assert_free_gpus(); OUT.mkdir(parents=True)
    write_new(OUT / "launch_manifest.json", {
        "protocol": "LIVEEDIT_MED_ROUTER_R2_MINIMAL_CAUSAL_RESCUE_V1",
        "stage": "T1_CALIBRATION_EVALUATION", "gpu_steps": GPU_STEPS,
        "previous_attempt_preserved": str(PREVIOUS),
        "previous_attempt_disposition": "STOPPED_AFTER_PARITY_BEFORE_FIRST_FAMILY__SUPERVISOR_GPU_CACHE",
        "training_checkpoint_set": str(RUN / "r2_t1/CHECKPOINT_SET.json"),
        "calibration_manifest": str(CALIBRATION),
        "e0_frozen_reference": str(E0_SELECTION),
        "evaluation_data_loaded": False, "record953_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False})
    context = mp.get_context("spawn")
    parity = context.Process(target=parity_worker, name="r2c-parity")
    parity.start(); parity.join()
    if parity.exitcode != 0:
        raise RuntimeError("ROUTER_R2_T1_RUNTIME_PARITY_FAILURE")
    processes = [context.Process(target=gpu_queue, args=(gpu, steps), name=f"r2c-g{gpu}")
                 for gpu, steps in GPU_STEPS.items()]
    for process in processes: process.start()
    for process in processes: process.join()
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("ROUTER_R2_T1_CALIBRATION_WORKER_FAILURE")
    results = [OUT / f"checkpoint_{step:04d}/calibration_result.json"
               for step in sorted(sum((list(value) for value in GPU_STEPS.values()), []))]
    from argparse import Namespace
    from scripts.liveedit_med.select_router_r2_t1 import finalize
    finalize(Namespace(result=results, out=OUT / "T1_CALIBRATION_SELECTION.json",
                       report=OUT / "T1_CALIBRATION_REPORT.md"))
    selection = json.loads((OUT / "T1_CALIBRATION_SELECTION.json").read_text())
    write_new(OUT / "completion_status.json", {
        "status": selection["decision_label"], "selected_step": selection["selected_step"],
        "candidate_frozen_for_clean_evaluation": selection["candidate_frozen_for_clean_evaluation"],
        "evaluation_data_loaded": False, "record953_loaded": False,
        "sealed_blind_loaded": False, "stage2_loaded": False})


if __name__ == "__main__":
    main()
