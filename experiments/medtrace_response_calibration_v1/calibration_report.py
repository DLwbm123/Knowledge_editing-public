"""Anonymous numerical calibration panels; no candidate selection or new method gate."""
import json
import os
import statistics
from pathlib import Path
import common as c

RUN=Path(os.environ['RUN_ROOT'])
mean=statistics.mean
PRECISIONS=('ORIGINAL_FP16','FP32_SAME_INPUTS')
ARMS=('RAW','BOUNDED')
SCALES=(.25,.5,1.)


def panel(rows,precision,arm,scale):
    items=[x for x in rows if x['precision']==precision and x['arm']==arm and x['scale']==scale]
    basis=[x for x in items if x['role']=='BASIS'];edit=[x for x in items if x['role']=='EDIT']
    assert len(basis)>0 and len(edit)>0
    value={key:mean(x[key] for x in basis) for key in ('KL','observed_logit_quadratic','JVP_quadratic','response_error_quadratic','sketch_quadratic','logit_RMS','changed_logit_fraction')}
    ratio=lambda numerator,denominator:value[numerator]/value[denominator] if value[denominator]>0 else None
    value.update(precision=precision,arm=arm,scale=scale,
        actual_over_observed_quadratic=ratio('KL','observed_logit_quadratic'),
        actual_over_JVP=ratio('KL','JVP_quadratic'),
        JVP_over_original_sketch=ratio('JVP_quadratic','sketch_quadratic'),
        response_relative_error=ratio('response_error_quadratic','observed_logit_quadratic')**.5 if value['observed_logit_quadratic']>0 else None,
        source_correct_token_losses=sum(x['lost_correct_tokens'] for x in basis),
        edit_progress=-sum((.5 if x['index']==0 else .125)*x['NLL_change'] for x in edit)/(len(edit)/5),
        edit_correct_token_losses=sum(x['lost_correct_tokens'] for x in edit))
    return value


def selfcheck():
    common=dict(precision='ORIGINAL_FP16',arm='RAW',scale=1.,KL=2.,observed_logit_quadratic=2.,
        JVP_quadratic=1.,response_error_quadratic=.5,sketch_quadratic=.5,
        logit_RMS=.1,changed_logit_fraction=.2,lost_correct_tokens=0)
    rows=[dict(common,role='BASIS',index=0)]
    rows += [dict(common,role='EDIT',index=i,NLL_change=-2. if i==0 else -1.) for i in range(5)]
    value=panel(rows,'ORIGINAL_FP16','RAW',1.)
    assert value['edit_progress']==1.5 and value['actual_over_JVP']==2. and value['response_relative_error']==.5
    rows[0].update(KL=0.,observed_logit_quadratic=0.,JVP_quadratic=0.,response_error_quadratic=0.)
    assert panel(rows,'ORIGINAL_FP16','RAW',1.)['response_relative_error'] is None
    return dict(status='PASS',weighted_edit_gain=True,zero_response_NA=True)


def main():
    selfcheck()
    data=[c.read(path) for path in sorted((RUN/'private/results').glob('*.json'))]
    reconstruction=[c.read(path) for path in sorted((RUN/'private/reconstruction').glob('*.json'))]
    mechanical=c.read(RUN/'private/MECHANICAL.json')
    assert len(data)==len(reconstruction)==8 and len({x['expert_order'] for x in data})==8
    assert not list((RUN/'private').glob('FAILURE*'))
    assert all(x['state_restored_exact'] and x['original_dtypes_restored'] for x in data+[mechanical])
    assert all(x['parent_geometry_reproduced'] for x in reconstruction)
    forwards=sum(x['forward_calls'] for x in reconstruction)+sum(x['counts']['forward_calls'] for x in data+[mechanical])
    backwards=sum(x['backward_passes'] for x in reconstruction)+sum(x['counts']['backward_passes'] for x in data+[mechanical])
    lock=c.read(RUN/'private/CALIBRATION_LOCK.json')
    assert forwards==lock['forward_calls']==10102 and backwards==lock['backward_passes']==19904
    assert all(len(x['rows'])==792 for x in data) and len(mechanical['rows'])==12
    rows=[r for x in data for r in x['rows']]
    panels=[panel(rows,precision,arm,scale) for precision in PRECISIONS for arm in ARMS for scale in SCALES]
    per=[dict(expert_order=x['expert_order'],panels=[panel(x['rows'],precision,arm,scale) for precision in PRECISIONS for arm in ARMS for scale in SCALES]) for x in data]
    ledger=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    assert len(ledger['gpu_sessions'])==13 and all(x.get('ended_epoch') for x in ledger['gpu_sessions'])
    counts={k:sum(x['counts'][k] for x in data+[mechanical]) for k in mechanical['counts']}
    result=dict(status='COMPLETE',decision='CALIBRATION_ONLY_NO_METHOD_PROMOTION',panels=panels,per_expert=per,
        precision_scope=mechanical['float32_scope'],source_observations_per_cell=488,
        random_probes_for_JVP=0,old_Fisher_probes_reconstructed=15616,
        mechanical_scientific_candidate_selection=False,old_PR47_decision='NO_LOCAL_SUPPORT',
        heldout_remeasured=False,clinical_protection=False,independent_confirmation=False,
        resource=dict(forward_calls=forwards,backward_passes=backwards,
            new_GPU_process_hours=(ledger['gpu_seconds_used']-inherited['gpu_seconds_used'])/3600,
            cumulative_GPU_process_hours=ledger['gpu_seconds_used']/3600,cumulative_Judge=ledger['Judge_attempts'],
            new_generations=0,new_Judge=0,candidate_reconstructions=8,new_optimized_candidates=0))
    c.write(RUN/'public/RESULTS.json',result)
    c.write(RUN/'public/COMPLETION_AUDIT.json',dict(status='PASS',counts=counts,total_forwards=forwards,
        total_backwards=backwards,restored_states=9,parent_candidates_reproduced=8,sessions_ended=13))
    report=['# 固定候选的响应校准诊断','',
        '结论：CALIBRATION_ONLY_NO_METHOD_PROMOTION。旧 PR47 的 NO_LOCAL_SUPPORT 不变。',
        'FP32 对照沿用原 FP16 权重数值与冻结多模态嵌入，转为 FP32 执行语言模型；TF32 关闭。不是更换预训练权重或重新优化候选。','',
        '| 精度 | 方向 | 尺度 | 源 KL | 实测 logit 二次型 | JVP 二次型 | KL/JVP | 响应相对误差 | 编辑收益 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in panels:
        values=[x[k] for k in ('KL','observed_logit_quadratic','JVP_quadratic','actual_over_JVP','response_relative_error','edit_progress')]
        report.append(f"| {x['precision']} | {x['arm']} | {x['scale']} | "+' | '.join('NA' if v is None else f'{v:.8g}' for v in values)+' |')
    report+=['','源问题等权；误差为 Fisher 加权 logits 响应误差。原随机二次型只在 ORIGINAL_FP16 路径下与 JVP 作直接估计比较。跨精度时 Jacobian 与基线概率也可变化。',
        '固定 0、1/4、1/2、1 的尺度用于校准曲线，不挑选尺度或候选；没有新留出集评估、自由生成、Judge 或完整训练。',
        f"成本与调用计数：{result['resource']}。"]
    (RUN/'public/REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    # The eight tiny reconstructed packages have no consumers after durable diagnostics.
    paths=[RUN/'private/candidates'/f"{x['expert_order']}.pt" for x in data]
    assert all(path.is_file() and not path.is_symlink() and path.resolve().parent==(RUN/'private/candidates').resolve() for path in paths)
    sizes=[path.stat().st_size for path in paths]
    c.write(RUN/'private/CANDIDATE_DELETION_PLAN.json',dict(paths=[str(x) for x in paths],bytes=sum(sizes),reason='All frozen-scale and precision consumers completed; source and geometry receipts permit reconstruction'))
    for path in paths:path.unlink()
    c.write(RUN/'public/LIFECYCLE.json',dict(status='COMPLETE',temporary_candidate_packages_deleted=8,bytes_deleted=sum(sizes),historical_artifacts_deleted=0,persistent_generated_checkpoints=0))
    c.write(RUN/'private/REPORT_COMPLETE.json',dict(status='COMPLETE',decision=result['decision']))


if __name__=='__main__':main()
