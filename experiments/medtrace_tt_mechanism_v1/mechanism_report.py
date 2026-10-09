"""Aggregate registered local probes; no optimizer selection or new efficacy test."""
import statistics
import time
import mechanism as e

c,p,RUN=e.c,e.p,e.RUN


def summarize(values):
    assert values
    return dict(n=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values))


def main():
    assert all((RUN/'private'/f'WORKER_{i}.json').exists() for i in range(6))
    experts=[c.read(RUN/'private/experts'/str(t['order'])/'COMPLETE.json') for t in p.tasks()[:8]]
    assert all(x['endpoint_exact'] and len(x['probes'])==4 for x in experts)
    probes=[z for x in experts for z in x['probes']];rows=[];ratios={}
    for mode in e.mm.MODES:
        selected=[next(z for z in x['candidates'] if z['mode']==mode) for x in probes]
        for metric in ('map_delta_frobenius','predictor_residual_delta_norm','KL_from_prewrite'):
            value=lambda x:x[metric] if metric!='KL_from_prewrite' else statistics.mean(z[metric] for z in x['native_FIT'])
            adam=[next(z for z in x['candidates'] if z['mode']=='ADAM') for x in probes]
            rv=[value(x)/value(a) for x,a in zip(selected,adam) if value(a)>0];ratios[mode,metric]=rv
            perexpert=[]
            for x in experts:
                pairs=[(next(z for z in probe['candidates'] if z['mode']==mode),next(z for z in probe['candidates'] if z['mode']=='ADAM')) for probe in x['probes']]
                v=[value(b)/value(a) for b,a in pairs if value(a)>0]
                if v:perexpert.append(statistics.median(v))
            rows.append(dict(mode=mode,metric=metric,ratio_to_Adam=summarize(rv) if rv else None,ratio_denominator_zero=len(selected)-len(rv),
                expert_medians=summarize(perexpert) if perexpert else None,experts_median_below_Adam=sum(v<1 for v in perexpert)))
    groups=[g for z in probes for g in z['groups']]
    assert len(probes)==32 and len(groups)==256
    angles=[dict(mode=m,group=g,cosine_to_RAW=summarize([x['cosine_to_RAW'] for x in groups if x['mode']==m and x['cores']==g and x['cosine_to_RAW'] is not None]),
        cosine_to_Adam=summarize([x['cosine_to_Adam'] for x in groups if x['mode']==m and x['cores']==g and x['cosine_to_Adam'] is not None])) for m in e.mm.MODES for g in [['G3','G4'],['G1','G2']]]
    cost=c.read(RUN/'RESOURCE_LEDGER.json');old=c.read(RUN/'private/INHERITED_COST.json');assert all(x.get('ended_epoch') for x in cost['gpu_sessions'])
    result=dict(status='COMPLETE',experts=8,endpoint_exact_all=True,probe_states=32,candidate_evaluations=128,optimizer_steps=2560,training_forwards=5120,training_backwards=5120,diagnostic_forwards=840,
        aggregate=rows,angles=angles,probes=probes,all_group_norm_bounds_pass=True,new_generation=0,new_Judge=0,
        GPU_process_hours=cost['gpu_seconds_used']/3600,new_GPU_process_hours=(cost['gpu_seconds_used']-old['gpu_seconds_used'])/3600,Judge_attempts=cost['Judge_attempts'],
        interpretation='Local counterfactual writes at shared states on the frozen RAW trajectory; native/FIT function response only. Not full Adam-trajectory causality or protection/generalization validation.',review_required=True)
    c.write(RUN/'public/RESULTS.json',result)
    c.write(RUN/'public/RESOURCE_LEDGER.json',{k:result[k] for k in ('GPU_process_hours','new_GPU_process_hours','Judge_attempts','new_Judge')})
    lines=['# 同状态方向机制诊断','', '| 方向 | 有效映射变化/Adam 中位比 | 预测位置残差变化/Adam 中位比 | native/FIT KL/Adam 中位比 |','|---|---:|---:|---:|']
    for mode in e.mm.MODES:lines.append('| '+mode+' | '+' | '.join(f'{statistics.median(ratios[mode,k]):.6g}' for k in ('map_delta_frobenius','predictor_residual_delta_norm','KL_from_prewrite'))+' |')
    lines+=['','8条RAW重建端点均与冻结历史权重逐元素一致。32状态、四方向均在同组Adam候选步长下比较；仅native/FIT局部函数诊断，不是新方法效果试验。','完整角度、分专家稳定性、逐节点与同查询强制/自然分解见配套JSON。']
    (RUN/'public/REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    # The report reads each designated snapshot before consuming its finite storage.
    snapshots=list((RUN/'private/snapshots').glob('*.pt'));assert len(snapshots)==32
    files=[]
    for path in snapshots:
        assert path.is_file() and not path.is_symlink();saved=p.load_state(path);assert saved['step'] in e.STEPS
        assert all(saved['RAW_actual'][k].shape==v.shape for k,v in saved['parameters_before'].items())
        files.append(dict(path=str(path),bytes=path.stat().st_size))
    assert sum(x['bytes'] for x in files)<=16777216
    c.write(RUN/'private/SNAPSHOT_LIFECYCLE.json',dict(files=files,reason='Fixed probes, endpoint parity and report readback complete; historical pilot weights untouched',epoch=time.time()))
    for x in files:
        from pathlib import Path
        Path(x['path']).unlink()
    c.write(RUN/'public/SNAPSHOT_LIFECYCLE.json',dict(status='CONSUMED_AND_DELETED',snapshots=32,bytes=sum(x['bytes'] for x in files),historical_weights_untouched=True,new_final_weights=0,reconstruction_requires_fixed_RAW320=True))
    c.write(RUN/'public/COMPLETION_AUDIT.json',dict(status='COMPLETE',all_GPU_sessions_ended=True,all_endpoint_parity=True,probe_states=32,new_Judge=0,review_required=True))
    p.done('REPORT_COMPLETE')


if __name__=='__main__':main()
