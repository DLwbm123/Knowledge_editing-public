import inspect
import json
from pathlib import Path

from m3bench_repro.inference.llava_med import LlavaMedAdapter, build_llava_med_query


def test_mistral_instruct_and_single_native_image_marker() -> None:
    assert LlavaMedAdapter.conversation_mode == "mistral_instruct"
    query = build_llava_med_query("Is the image abnormal?", mm_use_im_start_end=False)
    assert query == "<image>\nIs the image abnormal?"
    assert query.count("<image>") == 1


def test_im_start_end_are_not_injected_when_disabled() -> None:
    query = build_llava_med_query("Is the image abnormal?", mm_use_im_start_end=False)
    assert "<im_start>" not in query
    assert "<im_end>" not in query


def test_inference_query_has_no_gold_answer_parameter() -> None:
    query = build_llava_med_query("Are regions infarcted?", mm_use_im_start_end=False)
    assert "Yes" not in query
    assert "answer" not in inspect.signature(build_llava_med_query).parameters


def test_gate_a_manifest_is_greedy_and_has_no_sampling() -> None:
    manifest = json.loads((Path(__file__).parents[1] / "manifests/llava_gate_a_three_cases.json").read_text())
    assert manifest["generation"]["do_sample"] is False
    assert manifest["generation"]["num_beams"] == 1
