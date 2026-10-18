"""Summarize already completed anonymous batch traces; performs no model calls."""
import csv
import json
import statistics
from pathlib import Path


def summarize(root):
    root=Path(root);result=json.loads((root/'RESULT.json').read_text())
    samples=result['samples'];summary=[];trace=[];precision={'reference':[],'proposal':[],'correction':[]}
    for s in samples:
        method=s.get('method',{});metrics=s.get('metrics',{})
        base=s.get('baseline',{}).get('native',[None,None]);ref=s.get('reference',{})
        final=method.get('final_accepted_native_nll',[None,None])
        summary.append(dict(sample=s['sample'],status=s['status'],joint_pass=s['joint_pass'],
            accepted_steps=metrics.get('accepted_steps',0),corrections=metrics.get('correction_attempts',0),
            base_visual=base[0],reference_visual=ref.get('native_after',[None])[0],final_visual=final[0],
            edit_pass=method.get('edit_threshold_met',False),max_abs_null=metrics.get('max_abs_accepted_null'),
            boundary_margin=metrics.get('boundary_margin'),failures=';'.join(metrics.get('failures',[]))))
        def add(kind,step,value,diagnostics,verdict=None,status=''):
            if not value:return
            q=value.get('quantization',{})
            row=dict(sample=s['sample'],step=step,kind=kind,status=status,
                native_visual=value.get('native_after',[None,None])[0],native_null=value.get('native_after',[None,None])[1],
                native_global_null=diagnostics.get('native_cumulative_null_change'),
                fp32_global_null=diagnostics.get('fp32_cumulative_null_change'),
                null_gate_flip=diagnostics.get('null_budget_decision_flip'),
                absolute_rounding_l2=q.get('absolute_rounding_l2'),relative_rounding_l2=q.get('relative_rounding_l2'),
                delta_cosine=q.get('delta_cosine'),combined_native_fisher_cost=(verdict or {}).get('combined_native_fisher_cost'))
            trace.append(row)
            if q:precision[kind].append(row)
        add('reference',0,ref,s.get('reference_diagnostics',{}),status='REFERENCE_ONLY')
        for step in method.get('steps',s.get('method_candidates',[])):
            add('proposal',step['step'],step.get('proposal'),step.get('proposal_diagnostics',{}),step.get('proposal_verdict'),
                status='RAW_REQUIRES_CORRECTION' if step.get('correction') else step['status'])
            add('correction',step['step'],step.get('correction'),step.get('correction_diagnostics',{}),step.get('final_verdict'),status=step['status'])
    def distribution(values):
        return dict(count=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values)) if values else dict(count=0)
    quant={kind:dict(count=len(rows),null_gate_flips=sum(r['null_gate_flip'] is True for r in rows),
        **{k:distribution([r[k] for r in rows if r[k] is not None]) for k in ('absolute_rounding_l2','relative_rounding_l2','delta_cosine')}) for kind,rows in precision.items()}
    aggregate=dict(denominator=8,passed=sum(s['joint_pass'] for s in samples),
        accepted_path_margins=distribution([s['boundary_margin'] for s in summary if s['boundary_margin'] is not None]),
        passing_path_margins=distribution([s['boundary_margin'] for s in summary if s['joint_pass']]),
        near_boundary_passes=sum(s.get('metrics',{}).get('near_boundary_pass',False) for s in samples),
        corrections=sum(s['corrections'] for s in summary),precision=quant)
    aggregate['pass_rate']=aggregate['passed']/8
    (root/'SUMMARY.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    for name,rows in [('SAMPLE_SUMMARY.tsv',summary),('FULL_TRAJECTORY.tsv',trace)]:
        with (root/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');writer.writeheader()
            writer.writerows({k:('NA' if v is None or v=='' else v) for k,v in row.items()} for row in rows)
    return summary,aggregate
