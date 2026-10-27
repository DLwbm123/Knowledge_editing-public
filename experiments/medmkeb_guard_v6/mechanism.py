"""Sanitized complete training-side accounting, separate from semantic outcomes."""
import argparse,json
from pathlib import Path

def summarize(private):
    read=lambda p:json.loads(p.read_text())
    remote=private/'remote';initial=read(remote/'POOL_0.json');rows=[]
    for partition,arms in [(0,('NONE','POOL_MEAN','NORM_REPLAY')),(1,('POOL_PAIR','LOOKAHEAD'))]:
        for task in ('E01','E02'):
            for arm in arms:
                d=read(remote/str(partition)/task/(arm+'_TRAIN.json'));curve=d['curve'];assert d['status']=='FROZEN' and d['save_load_exact'] and len(curve)==160
                final=d['final_support'];assert len(final['nll'])==54 and len(final['margins'])==27
                rows.append(dict(task=task,arm=arm,steps=len(curve),source_loss_initial=curve[0]['source_loss'],source_loss_last=curve[-1]['source_loss'],retention_gradient_active_steps=sum(x['active_pairs']>0 for x in curve),zero_displacement_steps=sum(x['actual_norm']==0 for x in curve),total_operator_path_length=sum(x['actual_norm'] for x in curve),proposal_checks=sum(x['proposal_checks'] for x in curve),alpha_histogram={str(a):sum(x['alpha']==a for x in curve) for a in sorted({x['alpha'] for x in curve})} if arm=='LOOKAHEAD' else None,final_margin_regressions=sum(a-b>1e-4 for a,b in zip(initial['margins'],final['margins'])),final_correct_NLL_regressions=sum(final['nll'][i]-initial['nll'][i]>1e-4 for i in range(0,54,2)),max_accepted_constraint_violation=max(x['post_violation'] for x in curve) if arm=='LOOKAHEAD' else None))
    replay=[]
    for task in ('E01','E02'):
        a=read(remote/'1'/task/'LOOKAHEAD_TRAIN.json');b=read(remote/'0'/task/'NORM_REPLAY_TRAIN.json')
        replay.append(dict(task=task,max_norm_difference=max(abs(x['actual_norm']-y['actual_norm']) for x,y in zip(a['curve'],b['curve'])),final_support_NLL_max_difference=max(abs(x-y) for x,y in zip(a['final_support']['nll'],b['final_support']['nll'])),source_loss_last_difference=abs(a['curve'][-1]['source_loss']-b['curve'][-1]['source_loss'])))
    return dict(status='ALL_TRAINING_COMPLETE',training_side_only=True,not_semantic_effect_evidence=True,rows=rows,replay_comparison=replay,zero_gradient_control_limitation='NONE native source-only path differs from POOL_MEAN canonicalization/norm matching; zero-active differences are not a semantic gradient effect.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--private',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.write_text(json.dumps(summarize(a.private),indent=2)+'\n')
