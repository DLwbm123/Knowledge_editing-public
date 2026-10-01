"""Fail closed before any real-model loader, backend or paid service import."""
from __future__ import annotations
from typing import Any, Callable
import argparse
import json
import math
from .contracts import digest

STAGES = {"NATIVE_SMOKE": "NATIVE_SMOKE_ALLOWED", "PILOT": "PILOT_ALLOWED", "SEQUENTIAL": "SEQUENTIAL_ALLOWED", "EXPLORATORY_PILOT": "EXPLORATORY_ALLOWED"}
BINDINGS = {"code", "dependencies", "model", "data", "protocol", "config", "budget"}


def execution_bindings(config: dict[str, Any]) -> dict[str, str]:
    """Canonical actual execution payload. Only its self-reference is excluded.

    This computes digests, never approval. The independent human receipt must
    bind these values before any execution is admitted.
    """
    payload = {k: v for k, v in config.items() if k != "bindings"}
    budget_keys = ("budget_limits", "authorized_gpu_hours", "authorized_judge_calls",
        "native_wall_seconds_cap", "maximum_GGN_calls", "maximum_CG_iterations",
        "prior_native_seconds", "prior_GGN_calls", "max_active_constraints",
        "max_edit_steps", "mechanical_smoke_inputs", "new_run_storage_limit_gib",
        "teacher_cache_limit_gib")
    try:
        # Reject JSON NaN/Infinity rather than accepting their nonstandard hashes.
        json.dumps(payload, allow_nan=False)
        return dict(config=digest(payload), protocol=digest(config["protocol"]),
                    budget=digest({k: config[k] for k in budget_keys if k in config}))
    except (KeyError, TypeError, ValueError) as error:
        raise PermissionError("missing/noncanonical execution binding") from error


def advance_state(current: str, target: str, *, cpu_passed: bool = False,
                  audit_complete: bool = False, stage: str | None = None,
                  bindings: dict[str, str] | None = None, approval: dict[str, Any] | None = None,
                  config: dict[str, Any] | None = None,
                  trusted_authorization: dict[str, Any] | None = None) -> str:
    """Explicit stage progression; stage-A completion grants no native phase."""
    transitions = {"IMPLEMENTING":"CPU_TESTS_COMPLETE", "CPU_TESTS_COMPLETE":"DATA_AUDIT_COMPLETE",
        "DATA_AUDIT_COMPLETE":"WAITING_FOR_EXTERNAL_REVIEW", "NATIVE_SMOKE_ALLOWED":"NATIVE_SMOKE_COMPLETE",
        "PILOT_ALLOWED":"PILOT_COMPLETE", "SEQUENTIAL_ALLOWED":"COMPLETE"}
    if target in STAGES.values():
        expected = {"NATIVE_SMOKE_ALLOWED":"WAITING_FOR_EXTERNAL_REVIEW", "PILOT_ALLOWED":"NATIVE_SMOKE_COMPLETE",
                    "SEQUENTIAL_ALLOWED":"PILOT_COMPLETE", "EXPLORATORY_ALLOWED":"WAITING_FOR_EXTERNAL_REVIEW"}[target]
        if current != expected or not stage or STAGES.get(stage) != target:
            raise PermissionError("phase transition dependency mismatch")
        require_external_approval(stage, bindings or {}, approval, {**(config or {}),"current_state":target},
                                  trusted_authorization=trusted_authorization)
    elif transitions.get(current) != target:
        raise PermissionError("invalid state transition")
    elif target == "CPU_TESTS_COMPLETE" and not cpu_passed:
        raise PermissionError("CPU tests incomplete")
    elif target in {"DATA_AUDIT_COMPLETE","WAITING_FOR_EXTERNAL_REVIEW"} and not audit_complete:
        raise PermissionError("data audit incomplete")
    elif current in STAGES.values():
        if not (config or {}).get("phase_validation_passed"):
            raise PermissionError("phase validation incomplete")
    return target


def default_config() -> dict[str, Any]:
    return dict(run_name="directw_evidence_v1", execution_stage="IMPLEMENT_ONLY",
        allow_native_model_execution=False, allow_gpu=False, allow_training=False, allow_paid_judge=False,
        external_review_required=True, approved_phases=[], approval_reference=None,
        model_binding=None, editable_weight_path=None, data_manifest_digest=None, approved_protocol_digest=None,
        curvature_backend="exact_ggn_matvec", qp_variant="l2_slack_with_preservation_linear_term",
        max_active_constraints=8, cg_max_iter=32, cg_rtol=1e-4, max_edit_steps=20,
        line_search_factors=[1., .5, .25, .125, .0625, .03125, .015625],
        trust_radius=None, behavior_thresholds=None, preservation_budgets=None,
        history_max_inputs=64, history_max_predictor_positions=4096, new_run_storage_limit_gib=20,
        teacher_cache_limit_gib=2, gpu_physical_candidates=[5,6,7], leased_gpu_uuids=[],
        authorized_gpu_hours=0, authorized_judge_calls=0, data_audit_status="BLOCKED",legal_fit_inputs=0,
        smoke_data_audit_status="BLOCKED", mechanical_smoke_inputs=0, native_wall_seconds_cap=0)


def require_external_approval(stage: str, bindings: dict[str, str], approval: dict[str, Any] | None,
                              config: dict[str, Any], *, trusted_authorization: dict[str, Any] | None = None) -> None:
    """Trusted receipt comes from human instruction channel, NEVER approval-file claims.

    This CLI deliberately cannot fabricate that receipt. A later human-authorized
    orchestration must supply it separately and preserve original review wording.
    """
    if stage not in STAGES or config.get("execution_stage") != stage:
        raise PermissionError("stage A does not authorize native execution")
    if not trusted_authorization or trusted_authorization.get("origin") not in {"USER_MESSAGE", "VERIFIED_EXTERNAL_REVIEW"}:
        raise PermissionError("no independently trusted human authorization")
    if not approval or approval.get("approved") is not True or stage not in approval.get("approved_phases", []):
        raise PermissionError("missing explicit phase approval")
    if not trusted_authorization.get("original_text") or approval.get("approval_reference") != trusted_authorization.get("reference"):
        raise PermissionError("approval provenance mismatch")
    if stage not in trusted_authorization.get("phases", []) or trusted_authorization.get("bindings") != bindings:
        raise PermissionError("trusted receipt stage/bindings mismatch")
    if set(bindings) != BINDINGS or not all(isinstance(v,str) and len(v)==64 for v in bindings.values()) or approval.get("bindings") != bindings:
        raise PermissionError("scientific binding changed or missing")
    actual = execution_bindings(config)
    if any(bindings[k] != value for k, value in actual.items()) or config.get("approved_protocol_digest") != actual["protocol"]:
        raise PermissionError("actual execution config/protocol/budget differs from approved binding")
    if config.get("current_state") != STAGES[stage] or not config.get("allow_native_model_execution"):
        raise PermissionError("state/native permission mismatch")
    if not all(config.get(k) for k in ("model_binding", "editable_weight_path", "data_manifest_digest",
            "approved_protocol_digest", "trust_radius", "behavior_thresholds", "preservation_budgets")):
        raise PermissionError("unapproved critical scientific configuration")
    if stage == "NATIVE_SMOKE":
        limits = config.get("budget_limits", {})
        def finite_number(value: Any) -> bool:
            return type(value) in (int, float) and math.isfinite(value)
        total_seconds = limits.get("total_native_seconds")
        total_calls = limits.get("total_GGN_calls")
        prior_seconds = config.get("prior_native_seconds")
        prior_calls = config.get("prior_GGN_calls")
        calls, iterations = config.get("maximum_GGN_calls"), config.get("maximum_CG_iterations")
        cg_limit = limits.get("maximum_CG_iterations")
        if not all(finite_number(v) for v in (total_seconds, prior_seconds, config.get("native_wall_seconds_cap"))) or not (
            0 <= prior_seconds < total_seconds <= 3600 * config.get("authorized_gpu_hours", 0)
            and 0 < config["native_wall_seconds_cap"] <= total_seconds - prior_seconds
        ):
            raise PermissionError("cumulative native time budget exceeded or missing")
        if not all(type(v) is int for v in (total_calls, prior_calls, calls, iterations, cg_limit)) or not (
            0 <= prior_calls < total_calls and 0 < calls <= total_calls - prior_calls and 0 < iterations <= cg_limit
        ):
            raise PermissionError("cumulative GGN/CG budget exceeded or missing")
        if config.get("cg_max_iter") != iterations or config["protocol"].get("cg_max_iter") != iterations or config["protocol"].get("cg_rtol") != config.get("cg_rtol"):
            raise PermissionError("conflicting execution and protocol CG settings")
        if config.get("smoke_data_audit_status") != "PASS" or not 1 <= config.get("mechanical_smoke_inputs", 0) <= 2:
            raise PermissionError("mechanical smoke data audit not admitted")
        if config.get("allow_training") or config.get("allow_paid_judge") or config.get("authorized_judge_calls", 0):
            raise PermissionError("mechanical smoke cannot grant scientific training or Judge")
        if not (0 < config.get("native_wall_seconds_cap", 0) <= 3600 * config.get("authorized_gpu_hours", 0)):
            raise PermissionError("finite native smoke wall budget missing")
        if not 1 <= config.get("max_active_constraints", 0) <= 2 or config.get("max_edit_steps") != 1:
            raise PermissionError("mechanical transaction must be bounded to one step and two constraints")
    elif stage == "EXPLORATORY_PILOT":
        waivers = {"scientific_fit_admission", "independent_calibration", "previous_smoke_budget", "external_review_stop"}
        if set(config.get("waived_requirements", [])) != waivers or set(trusted_authorization.get("waived_requirements", [])) != waivers:
            raise PermissionError("exploratory exceptions require explicit independently trusted user instruction")
        if config.get("result_kind") != "EXPLORATORY_TRAINING_ONLY" or not config.get("verified_training_source") or config.get("exploratory_inputs", 0) <= 0:
            raise PermissionError("exploratory source/scope must be explicit; no scientific FIT relabeling")
        if not config.get("prelaunch_native_validation_required") or config.get("allow_paid_judge") or config.get("authorized_judge_calls", 0):
            raise PermissionError("exploration requires repaired native-path validation and no Judge")
    else:
        counts = config.get("scientific_role_counts", {})
        required = {"EDIT_FIT", "GEN_FIT", "PROTECT_BG_FIT", "PROTECT_NEAR_FIT"}
        if config.get("method") == "W_EVIDENCE_QP":
            required.add("VIS_PAIR_FIT")
        if config.get("data_audit_status") != "PASS" or config.get("legal_fit_inputs", 0) <= 0 or not all(counts.get(r, 0) > 0 for r in required):
            raise PermissionError("scientific data role audit not admitted")
        if not config.get("frozen_dev_cal_digest") and not config.get("preapproved_global_rule_digest"):
            raise PermissionError("independent calibration or preapproved global rule missing")
    if min(config.get("new_run_storage_limit_gib", 0),config.get("teacher_cache_limit_gib", 0)) <= 0:
        raise PermissionError("storage budget missing")
    hours = config.get("authorized_gpu_hours", 0)
    if not isinstance(hours, (int, float)) or not math.isfinite(hours) or not config.get("allow_gpu") or hours <= 0 or not config.get("leased_gpu_uuids"):
        raise PermissionError("GPU budget/lease missing")
    if config.get("allow_paid_judge") and config.get("authorized_judge_calls", 0) <= 0:
        raise PermissionError("paid Judge budget missing")
    if stage != "NATIVE_SMOKE" and not config.get("allow_training"):
        raise PermissionError("training phase not permitted")
    if stage in {"PILOT", "SEQUENTIAL"} and config.get("native_smoke_status") != "PASS":
        raise PermissionError("native smoke incomplete")
    if stage == "SEQUENTIAL" and config.get("pilot_gate_status") != "PASS":
        raise PermissionError("pilot prerequisite incomplete")


def gated_load(stage: str, bindings: dict[str, str], approval: dict[str, Any] | None,
               config: dict[str, Any], loader: Callable[[], Any], *,
               trusted_authorization: dict[str, Any] | None = None) -> Any:
    require_external_approval(stage, bindings, approval, config, trusted_authorization=trusted_authorization)
    return loader()


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent stage-A review gate; native dispatch requires a trusted human receipt")
    parser.add_argument("--stage", choices=["IMPLEMENT_ONLY", *STAGES], default="IMPLEMENT_ONLY")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--approval")
    args = parser.parse_args(argv)
    config = json.load(open(args.config)) if args.config else default_config()
    approval = json.load(open(args.approval)) if args.approval else None
    if args.stage != "IMPLEMENT_ONLY" or args.run:
        require_external_approval(args.stage, {}, approval, config)
    print(json.dumps(dict(current_state="WAITING_FOR_EXTERNAL_REVIEW", native_gpu_runs=0,
                         real_model_training_started=False, paid_judge_calls=0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
