"""Build review artifacts from CPU checks and pre-read PRIVATE metadata only.

No SSH, model loader, GPU invocation, paid Judge or scheduler in this program.
"""
from __future__ import annotations
import ast
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
from .contracts import audit_data, digest, method_eligibility, enforce_storage
from .gate import default_config, advance_state
from .tests import main as run_cpu_tests


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def build(run_root: Path, metadata_path: Path) -> int:
    """Refuse existing run roots; public output contains aggregates, never row IDs."""
    repo=Path.cwd();source=repo/"experiments/directw_evidence_v1"
    run_root.mkdir(parents=True,exist_ok=False)
    public=repo/"reports/directw_evidence_v1_stage_a_20260930";public.mkdir(parents=True,exist_ok=True)
    (source/"configs").mkdir(exist_ok=True)
    config=default_config();write(source/"configs/default.json",config)
    status=dict(current_state="IMPLEMENTING",native_gpu_runs=0,real_model_training_started=False,
                paid_judge_calls=0,native_cpu_model_runs=0,automatic_continuations_created=0)
    write(run_root/"RUN_STATUS.json",status)
    if run_cpu_tests(run_root/"CPU_TEST_RESULTS.json"):
        status["current_state"]="FAILED";write(run_root/"RUN_STATUS.json",status);return 1
    status["current_state"]=advance_state(status["current_state"],"CPU_TESTS_COMPLETE",cpu_passed=True)
    metadata=json.loads(metadata_path.read_text())
    write(run_root/"STATIC_METADATA.private.json",metadata)
    rows=metadata["private_manifest"]
    (run_root/"DATA_MANIFEST.private.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
    audited=audit_data(rows,edit_index=0,test_ids=set(),expected_model=None)
    write(run_root/"DATA_AUDIT.private.json",audited)
    data=dict(status="BLOCKED_DATA",audit_execution="COMPLETE",static_read_only=True,
        observed_utc=metadata["observed_utc"],candidate_rows=len(rows),candidate_source_groups=len({r["source_group"] for r in rows}),
        candidate_train_annotation_rows=metadata["legacy_overlay_rows"],existing_candidate_image_paths=sum(r["image_exists"] for r in rows),
        verified_legal_edit_count=0,verified_independent_sources=0,verified_gen_fit_count=0,qualified_visual_pairs=0,
        verified_background_protection=0,verified_near_protection=0,verified_history_support=0,
        independent_dev_cal="NOT_ESTABLISHED",independent_test_confirm="NOT_ESTABLISHED",
        legacy_joins_are_directw_permission=False,legacy_cache_reused=False,base_correct_masks="PENDING_BOUND_LEGAL_CACHE_OR_PHASE_B",
        finding_counts=dict(Counter(f["reason"] for f in audited["findings"])),
        exact_duplicate_groups=len(audited["exact_duplicates"]),lexical_near_duplicate_pairs=len(audited["near_duplicates"]),
        conflicting_fact_groups=len(audited["fact_conflicts"]),same_source_groups=len(audited["same_source_groups"]),
        image_near_duplicates=audited["image_near_duplicates"],patient_and_fact_group_join="BLOCKED_MISSING_COMPLETE_LEDGER",
        evaluated_role_ids_read=0,test_question_answers_read=0,medical_counterexamples_generated=0,
        permission_interpretation="User authorizes stage-A static audit; old training roles do not prove new per-role training semantics or independent confirmation.",
        existing_evaluation_counts="REG_EXPOSED_OR_EXPOSURE_UNKNOWN; never reassigned to FIT/TEST_CONFIRM")
    write(public/"DATA_AUDIT.json",data)
    eligibility=method_eligibility(audited)
    table={branch:dict(status=state,legal_edits=0,independent_sources=0,gen_fit=0,visual_pairs=0,
                       protection_inputs=0,dev_cal=False,test_confirm=False,
                       blocker="no verified DirectW role ledger / protection semantics / calibration",native="PENDING_NATIVE_CHECK")
           for branch,state in eligibility.items()}
    table["W_EVIDENCE_QP"]["blocker"] += "; no independently verified incompatible-answer visual pair"
    write(public/"METHOD_DATA_ELIGIBILITY.json",table)
    policy=dict(fit_roles=sorted({"EDIT_FIT","GEN_FIT","VIS_PAIR_FIT","PROTECT_BG_FIT","PROTECT_NEAR_FIT","PAST_EDIT_MEMORY"}),
        excluded_from_gradients_curvature_mining=["DEV_CAL","REG_EXPOSED","TEST_CONFIRM"],
        source_role_alone_is_permission=False,group_isolation=["edit","source/patient","fact_family"],
        same_image_original_and_official_probe="benchmark sharing allowed; evaluation QA cannot be repurposed for fit",
        future_information=False,clinical_negative_synthesis=False,
        base_anchor="fixed clean Base for unrelated facts",history_anchor="accepted edited distribution, explicit supersedes",
        cache_bindings=["model","weight_version","input","prefix","mask","dtype","backend","config","teacher_version"])
    write(public/"ROLE_ACCESS_POLICY.json",policy)
    state=advance_state(status["current_state"],"DATA_AUDIT_COMPLETE",audit_complete=True)
    status["current_state"]=advance_state(state,"WAITING_FOR_EXTERNAL_REVIEW",audit_complete=True)
    package_versions={name:importlib.metadata.version(name) for name in ("torch","numpy","scipy")}
    dependencies=dict(python=platform.python_version(),packages=package_versions,platform=platform.system(),
                      CPU_dtype="float64",native_environment="NOT_LOADED_OR_UPGRADED",backend="torch.func JVP+VJP on synthetic CPU only")
    write(public/"ENVIRONMENT.json",dependencies)
    public_base=dict(status="STATIC_HEADER_VERIFIED_NATIVE_PENDING",proposal_module="model.layers.21.mlp.down_proj",
        proposal_weight="model.layers.21.mlp.down_proj.weight",selected_shape=metadata["weight_headers"]["selected"]["shape"],
        serialized_dtype=metadata["weight_headers"]["selected"]["dtype"],selected_bias_in_checkpoint=metadata["weight_headers"]["selected_bias_exists"],
        model_config=metadata["model_config"],config_digest=metadata["config_digest"],
        checkpoint_tensor_elements=metadata["weight_headers"]["total_tensor_elements"],
        actual_instantiated_parameter_count=None,editable_original_parameter_count=4096*14336,new_deployment_parameters=0,
        full_clean_base_content_digest=None,full_base_binding_status="BLOCKED_NOT_CONTENT_BOUND",
        native_shared_storage="PENDING_NATIVE_CHECK",native_backend="PENDING_NATIVE_CHECK",
        all_branches_clean_base="required; no adapter/optimizer loaded",native_deployment_dtype=None)
    write(public/"BASE_BINDINGS.json",public_base)
    imports=[]
    for path in sorted(source.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom) and node.module:imports.append(node.module)
    forbidden=[i for i in imports if i.startswith(("methods.","m3bench_repro.","peft","easyeditor"))]
    if forbidden:raise RuntimeError("legacy editor import coupling")
    isolation=dict(status="PASS_STAGE_A_ISOLATION",branch="research/directw-evidence-v1",parent_commit="2c5bfc1537c553c2623a49eea023a49749f23eaf",
        namespace="experiments/directw_evidence_v1",run_root="separate ignored outputs tree in independent worktree",
        modified_old_files=[],legacy_runtime_imports=forbidden,legacy_adapter_loads=0,legacy_optimizer_loads=0,
        legacy_queues_changed=0,real_model_instantiations=0,only_native_matrix_changes="verified on CPU fixture; native pending",
        remote_actions="read-only config/weight headers/train metadata/GPU inventory; no remote files written",
        remote_occupation="four visible A100 40GB devices already occupied; no lease acquired; GPU5/6/7 availability not established")
    write(public/"ISOLATION_AUDIT.json",isolation)
    write(run_root/"ISOLATION_AUDIT.private.json",dict(isolation,worktree=str(repo),run_root=str(run_root),original_checkout=str(repo.parent/"Knowledge_editing-medtrace-stage17-20260912")))
    n=4096*14336;gib=1024**3
    resource=dict(status="STATIC_ESTIMATE_NATIVE_MEASUREMENT_REQUIRED",matrix_elements=n,
        matrix_bf16_bytes=n*2,matrix_fp32_bytes=n*4,checkpoint_tensor_elements=public_base["checkpoint_tensor_elements"],
        serialized_model_bytes=sum(s["size"] for s in metadata["shards"]),
        solver_minimum_working_vector_budget=dict(A_rows=8,Qinv_A_columns=8,additional_vectors=12,
            fp32_bytes=28*n*4,GiB=28*n*4/gib,excludes_autodiff_activations_and_full_logits=True),
        runtime_host_state_bytes_estimate=2*public_base["checkpoint_tensor_elements"]*2,
        runtime_host_note="Base reset snapshot plus edit transaction snapshot, tensor-count lower bound; dtype/allocator overhead pending",
        gpu_peak_bytes=None,seconds_per_forward=None,seconds_per_backward=None,seconds_per_matvec=None,
        worst_case_draft=dict(groups=3,CG_solves=9,CG_iter_cap=32,GGN_calls_per_QP_step=3*9*(32+3)+3,
            GGN_calls_per_edit_20_steps=(3*9*(32+3)+3)*20,
            note="each exact GGN entails a JVP and VJP; three stationarity/evaluation probes included; extra constraint/true-forward cost excluded"),
        history_max_inputs=64,history_max_predictor_positions=4096,
        history_full_vocab_teacher_fp32_bytes=4096*32000*4,history_cache_cap_bytes=2*gib,
        run_storage_cap_bytes=20*gib,shared_base_copies=0,per_edit_backbone_checkpoints=0,
        native_gpu_hours_authorized=0,paid_judge_calls_authorized=0,GPU_candidates=[5,6,7],leased_GPU_uuids=[],
        native_revalidation_required=["actual derivative backend","peak VRAM/host RAM","matvec timing","full paired-block generation/scoring budget","current physical GPU lease"],
        storage_lifecycle="small original-matrix final/reference/resume snapshots only; dependency-aware dry-run cleanup in own RUN_ROOT")
    write(public/"RESOURCE_AND_STORAGE_ESTIMATE.json",resource)
    results=json.loads((run_root/"CPU_TEST_RESULTS.json").read_text())
    results["command"]="python - < neutral stdin runner (see experiments/directw_evidence_v1/README.md)"
    write(public/"CPU_TEST_RESULTS.json",results)
    code=digest({str(p.relative_to(repo)):p.read_text() for p in sorted(source.rglob("*"))
                 if p.is_file() and p.suffix in {".py",".json"}})
    bindings=dict(code=code,dependencies=digest(dependencies),model=None,
        static_model_config=metadata["config_digest"],data=digest(rows),config=digest(config),
        protocol=digest((public/"EXPERIMENT_PROTOCOL_DRAFT.yaml").read_text()),budget=digest(resource))
    write(public/"REVIEW_BINDINGS.json",bindings)
    write(public/"APPROVAL.template.json",dict(approved=False,approved_phases=[],approval_reference=None,
        original_approval_text=None,bindings={k:bindings.get(k) for k in ("code","dependencies","model","data","protocol","config","budget")},
        authorized_gpu_hours=0,authorized_judge_calls=0,leased_gpu_uuids=[],
        warning="TEMPLATE ONLY. Self-authored reviewer names are not authorization; material scientific changes invalidate approval."))
    status.update(code_digest=code,data_manifest_digest=bindings["data"],cpu_test_counts=results["counts"],
                  data_readiness="BLOCKED_DATA",native_status="PENDING_NATIVE_CHECK",
                  publication="See Git branch HEAD and private DELIVERY_RECEIPT.json; scientific state is independent of publication")
    write(public/"RUN_STATUS.json",status);write(run_root/"RUN_STATUS.json",status)
    cmd=subprocess.check_output(["ps","-p",str(__import__('os').getpid()),"-o","command="],text=True).strip()
    if any(word in cmd.lower() for word in ("wangbomin","medtrace","directw","lora")):
        raise RuntimeError("process argv privacy violation")
    write(run_root/"ENVIRONMENT.private.json",dict(dependencies,python_executable=sys.executable,process_argv=cmd,process_privacy="PASS"))
    enforce_storage(run_root,20*gib,0)
    print(json.dumps(dict(current_state=status["current_state"],tests=results["counts"],code_digest=code,
                          static_candidate_rows=len(rows),branches=eligibility)))
    return 0
