"""One frozen specification shared by local preregistration and GPU admission."""

def specification(parent, mechanism_check, journal_check):
    import copy
    lock=copy.deepcopy(parent)
    lock.pop('status',None);lock.pop('engineering_recovery',None)
    lock.update(epoch='PUREW_SCOPE_V2_20261010',arms=['GATED','ALWAYS_ON'],
        method='ScopeEdit core: evidence-gated orthogonal branch writes and historical RLS, merged into original W',
        refresh_geometry='fixed initial-Base weight nullspace; two orthogonal frozen bases',
        nullspace='512 fixed seeded orthogonal coordinates inside initial W right nullspace',
        rank_private=448,rank_shared=64,basis_seed=20261010,
        scope_gate=dict(tau=.2,beta=10.,eps=1e-6,formula='sigmoid(beta*(cos-tau))*min_norm/max_norm',
                        control='ALWAYS_ON gamma=1 for nonzero modalities'),
        branch_write='dCp=-lr*G*Ap.T*Pp; dCs=-lr*gamma*G*As.T*Ps; jointly clip coordinate norm5',
        scope_reference='paper equations14-24; not literal author-code variant',
        feature_pooling='causal answer-predictor down_proj input mean versus visual-patch mean; count-weighted multimodal mean',
        feature_cache='exact for one down_proj layer with frozen upstream; GPU checks exact feature invariance',
        KL_shared_gate='same source-weighted gamma; no ungated shared-write bypass',
        history_update='once AFTER each edit; seven source keys weighted .25/.0625x4/.25x2; shared rho multiplied by gamma; one fixed BASIS private-only key',
        history_preconditioners='identity initially; Sherman-Morrison with sqrt(rho)*key; no current edit in its own preconditioner',
        geometry_precision='FP64 preconditioners, cast to FP32 only for native-weight coordinate gradient',
        deployment='only original W; editing-only coefficient bookkeeping, bases, keys, P discarded after consumers',
        no_new_trainable_model_parameters=True,mechanical_temporary_updates=2,
        inherited_Base_consumers=267,base_generations=0,base_identity_probe_generations=2,
        maximum_generations=754,maximum_Judge=752,consumers=1019,
        formal_backwards=10240,maximum_backwards=10244,
        maximum_non_generation_forwards=13000,
        engineering_attempts='separately cumulative; only uncommitted work may repeat after engineering failure; no extra scientific steps',
        resume='atomic coordinate/P checkpoint EVERY step; binding-validated stages and outputs; reconcile checkpoint-last-event gap',
        diagnostic_retention='immutable per-committed-step JSON, attempted-call receipts, geometry/gate summaries independent of weights',
        primary_arm='GATED',paired_comparison='GATED minus ALWAYS_ON; first-round arms are historical references only',
        mechanism_selfcheck=mechanism_check,journal_selfcheck=journal_check,
        deadline_hours_from_first_install=24,owned_weight_limit_bytes=2*1024**3,
        minimum_free_bytes=8*1024**3,
        round_one_read_only=True,
        adaptation_notes_zh=[
            '补入 ScopeEdit 写入时的图文一致性门控、正交私有/共享空间和历史 RLS；两臂仅门控不同。',
            '遵循论文公式14—24的 sqrt(rho) 历史更新，不声称逐行复现作者代码的另一归一化实现。',
            '固定原 Base 单层 W 零空间、448/64维；只对原 W 求梯度与写入，坐标系不是新增模型参数。',
            '训练采用已准入源题/FIT/GFIT 与61条 BASIS 的原 Base 分布 KL；T1G/T2G/96题仅评价。',
            '门控控制共享写入和历史传播，不在推理时选专家；新进程只装载原 W，检查参数键/形状/数量和回答一致性。',
            '原生 FP32、固定步数/顺序/种子；不含 LOKI 动态 HSIC 选层或 DOW-KE 多层参数化。',
            '每一步提交恢复状态；独立保留每步诊断、失败与调用账本，已完成训练/生成/评分不重复。',
            '首轮结果与资产只读；新轮有限24小时，保留累计成本，不按评价错误修改训练标签。',
            '余弦门控不等于临床作用域有效性；DEV 成功不等于独立临床验证或 SOTA。'])
    lock['success_criteria']['projection_contribution']='not tested this round; compare scope gate increment conditional on fixed dual geometry'
    lock['success_criteria']['insertion_correct']='all24 native/GFIT queries correct at their own insertion and all later prefixes'
    lock['success_criteria']['scope_contribution']='report paired complete denominators; no superiority claim unless GATED improves a registered behavioral panel without owner-wise native/GFIT/generalization or Base-correct damage regression'
    return lock
