"""Finite diagnostic protocol, frozen before any new GPU write."""
from diagnostic_math import ARMS

def specification(parent, mechanism, journal):
    import copy
    p=copy.deepcopy(parent);p.pop('status',None);p.pop('engineering_recovery',None)
    p.update(epoch='PUREW_SCOPE_V3_20261010',arms=list(ARMS),GPUS=[5],trajectories=5,
        GPU_UUID='GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca',
        method='Pure original W: dimension, norm, fixed/refreshed geometry and historical RLS attribution',
        interventions=dict(A_LOW='fixed512, P=I, gamma=1',B_NORM='A directions; same-state full-null update norm',
            C_FULL_FIXED='full initial right nullspace, P=I',D_FULL_REFRESH='full current-W right nullspace refreshed each edit, P=I',E_RLS='fixed512, round2 ALWAYS_ON historical RLS, gamma=1'),
        contrasts={'amplitude':'B_NORM minus A_LOW','directions_conditional_on_norm':'C_FULL_FIXED minus B_NORM','refresh':'D_FULL_REFRESH minus C_FULL_FIXED','RLS_removal':'A_LOW minus E_RLS'},
        rank_private=448,rank_shared=64,basis_seed=20261010,full_null_dimension=10240,
        refresh_geometry='Arm-specific, specified in interventions',nullspace='fixed512 versus full10240; no rank sweep',
        norm_matching='B: -min(.05,.01*||G Pi_initial_full||)*d512/||d512||; zero d512 skips; includes same weighted CE+KL gradient',
        all_gamma=1.,history_update='E only; after each edit, original seven source rho and one private BASIS key; no normalization change',
        diagnostic_nodes=[1,40,80,160],diagnostics='separate CE/KL norms, full/low energy, current-gradient RLS attenuation, update dot CE/KL; no shadow writes',
        maximum_updates=6400,formal_backwards=25600,maximum_backwards=25604,mechanical_temporary_updates=5,
        maximum_generations=1882,maximum_Judge=1880,prefix_generations=540,base_generations=0,
        final_generations=1335,reload_generations=5,inherited_Base_consumers=267,base_identity_probe_generations=2,consumers=2147,
        maximum_non_generation_forwards=31000,deadline_hours_from_first_install=24,owned_weight_limit_bytes=2*1024**3,minimum_free_bytes=8*1024**3,
        GPU_schedule='GPU5 only; one GPU worker at a time; all5 train/freeze before all5 final evaluations',
        primary_arm=None,selection=False,automatic_tuning=False,round_one_read_only=True,round_two_read_only=True,
        mechanical_selfcheck=mechanism,journal_selfcheck=journal,
        engineering_attempts='Preserve all attempted calls and failures; resume only uncommitted work; scientific updates/output consumers fixed, counters cannot reset',
        resume='exact per-step dense W or low-coordinate state, D boundary QR, E historical inverse; atomic commit and independent step journal',
        scoring='same isolated Astra medium; valid exact old scores reused; prior failed same-binding payloads remain missing; no retries',
        checkpoint_lifecycle='active state retires after bound candidate and train receipt; candidates after prefix/final/native reload/durable Judge inputs; geometry/cache after all consumers',
        adaptation_notes_zh=['五臂纯原W诊断；不增加推理参数或分支。','学习率、数据、顺序、步数、保护KL和成功阈值保持；B仅改变预先指定的幅度规则。','完整零空间用薄QR隐式投影，固定与每编辑刷新分别检验。','旧失败评分不重试；新候选一次评分，缺失阻止成功声明。','仅一次已暴露DEV顺序，不能声称独立临床/SOTA或作者完整复现。'])
    p['success_criteria'].pop('projection_contribution',None)
    p['success_criteria']['insertion_correct']='all24 native/GFIT at own insertion and every later prefix'
    p['success_criteria']['attribution']='registered within-round contrasts; full denominators, owner-wise regressions, missing-aware bounds; no winner selection or compensation'
    return p
