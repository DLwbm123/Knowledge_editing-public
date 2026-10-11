"""Eight admitted edits: reuse C, compare unrestricted U and history-supervised H."""
import copy
from shared_math import ARMS, NORM_RTOL, NORM_ATOL


def specification(parent, mechanism, journal, reference):
    p=copy.deepcopy(parent)
    for k in ('status','engineering_recovery','mechanical_selfcheck','journal_selfcheck'):
        p.pop(k,None)
    p.update(epoch='PUREW_SCOPE_V4_20261011',arms=list(ARMS),new_arms=['U','H'],GPUS=[5],trajectories=2,
        GPU_UUID='GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca',
        method='Native original W, shared committed C norms, fixed nullspace versus unrestricted and history CE',
        interventions=dict(C='reuse round3 C_FULL_FIXED outputs and 1280 committed norms; no new C training',
                           U='unrestricted current CE + BASIS KL',H='U plus .25 past native CE and .25 past GFIT CE'),
        contrasts=dict(geometry='U minus C at shared actual update norms',history_supervision='H minus U at shared actual update norms'),
        ordered_edits=8,steps_per_edit=160,rank_private=None,rank_shared=None,basis_seed=None,
        refresh_geometry='none',nullspace='C fixed initial Base full right nullspace; U/H unrestricted',
        norm_matching='each U/H step consumes the immutable C actual FP32 norm at the same owner/step; no same-state reference recomputation',
        norm_reference=reference,actual_norm_relative_tolerance=NORM_RTOL,actual_norm_absolute_tolerance=NORM_ATOL,
        positive_target_zero_direction='stop, never skip or retune',
        current_edit_loss='.25 native + .25 cyclic FIT + .5 cyclic GFIT',protection_loss='BASIS KL weight1; same61 admitted prompts and original Base teacher',
        history_update='H only: past admitted native/GFIT, sorted(owner,query_id), cycles(step-1); every past owner represented, no success filtering',
        history_loss=dict(native=.25,GFIT=.25),first_edit_H_equals_U=True,history_replay=dict(C=False,U=False,H=True),
        all_gamma=1.,low_dimensional_space=False,historical_RLS=False,history_supervision_is_RLS=False,
        diagnostic_nodes=[1,40,80,160],diagnostics='CE/KL/history norms and cosine; update dot each gradient; rowspace energy; shared actual norm per step',
        maximum_updates=2560,inherited_C_updates=1280,formal_backwards=12480,maximum_backwards=12490,mechanical_temporary_updates=2,
        maximum_generations=754,maximum_Judge=752,prefix_generations=216,base_generations=0,final_generations=534,reload_generations=2,
        inherited_Base_consumers=267,inherited_C_consumers=376,base_identity_probe_generations=2,consumers=1395,
        maximum_non_generation_forwards=17000,deadline_hours_from_first_install=24,
        owned_weight_limit_bytes=2*1024**3,minimum_free_bytes=8*1024**3,
        GPU_schedule='GPU5 only, one worker; new U/H frozen before final evaluation; inherited C unchanged',
        primary_arm=None,selection=False,automatic_tuning=False,round_one_read_only=True,round_two_read_only=True,round_three_read_only=True,
        mechanical_selfcheck=mechanism,journal_selfcheck=journal,
        resume='exact committed dense FP32 W, owner/step and norm-reference binding; checkpoint authoritative, independent durable event journal',
        scoring='same one-attempt isolated Astra medium, exact prior valid scores reused, prior failed exact bindings remain missing, no retries',
        checkpoint_lifecycle='active state after committed candidate; U/H candidates after prefix/final/native reload/durable Judge inputs; teacher cache after both arms; historical assets read-only',
        supervision_boundary='only original admitted56 NATIVE/FIT/GFIT rows; H past NATIVE/GFIT only; T1G/T2G/T1L/T2L/HELDOUT excluded from CE',
        data_review='M3Bench T2G intentionally same-image text generality; no labels derived from seven old T2G errors; extra clinical cohort optional and not used',
        adaptation_notes_zh=['用户明确选择原8编辑严格机制对照；第三轮C只复用，不重跑。','U/C只检验固定零空间限制；H/U只检验已准入历史监督；逐步实际范数匹配。','不使用448/64空间、RLS、传播门控或刷新；无新增推理参数/分支。','这是已暴露DEV顺序的机制诊断，不能声称独立临床/SOTA或完整论文复现。'])
    p['success_criteria']['attribution']='U-C geometry and H-U history at common committed norms; complete owner/prefix denominators, missing-aware bounds; no winner selection'
    return p
