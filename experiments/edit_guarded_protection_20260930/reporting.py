"""Public aggregates only; keep medical answers, score keys and item scores private."""
import json,time,collections
from resources import ROOT,read,write
from phases import candidates,references,arms,curves,delete_adapters
from judge_protocol import read_scores
from policy import stats,mechanism
def markdown(name,title,data):
    from storage import Store
    Store(ROOT).write('public/'+name,(title+'\n\n```json\n'+json.dumps(data,ensure_ascii=False,indent=2)+'\n```\n').encode())

def report(phase):
    rows=candidates(phase);scores=read_scores(ROOT)
    if phase=='ROUTE':
        from route_removal import report as route_report
        d=route_report(rows,scores);write(ROOT/'public/ROUTE_REMOVAL_CAUSAL.json',d);markdown('ROUTE_REMOVAL_CAUSAL_ZH.md','# 完整冻结 REG12 路由移除诊断\n\n该块使用真实 R0 与生成；不用于 rho 选择。',d);return
    result={}
    for method in arms(phase):
        panels={}
        for r in rows:
            if r['arm']==method:panels.setdefault(r['mode']+'/'+str(r['prefix'])+'/'+r['task'],[]).append(r)
        result[method]={k:stats(rs,scores) for k,rs in panels.items()}
    frozen={}
    for r in references(phase):frozen.setdefault(r['arm'],{}).setdefault(r['mode']+'/'+str(r['prefix'])+'/'+r['task'],[]).append(r)
    reference_stats={m:{k:stats(rs,scores) for k,rs in panels.items()} for m,panels in frozen.items()}
    cc=curves(phase);mm=mechanism(cc)
    for method,summary in mm.items():
        for edit in summary['edits']:
            edit['parameter_update_norm_definition']='sum of per-step L2 update norms; path length'
            edit['parameter_net_update_norm']=cc[method][str(edit['edit_order'])][-1]['parameter_net_update_norm']
    write(ROOT/f'public/{phase}_AGGREGATES.json',dict(methods=result,references=reference_stats,mechanisms=mm))
    extra={}
    if phase in ['DEV','REG']:
        start=1 if phase=='DEV' else 25
        for method in arms(phase):
            early=[r for r in rows if r['arm']==method and r['order'] in range(start,start+8) and r['task'] in ['T0','T1G','T2G'] and r['mode']=='sequential']
            extra[method]={str(p):{t:stats([r for r in early if r['prefix']==p and r['task']==t],scores) for t in ['T0','T1G','T2G']} for p in [8,12,24]}
        write(ROOT/f'public/{phase}_FIXED_EARLY_COHORT.json',extra)
        if phase=='DEV':
            refs=references(phase);analysis={}
            from exact import identity
            for mode,prefix in [('single',1)]+[('sequential',p) for p in [4,8,12,24]]:
                baseline={identity(r):r for r in refs if r['arm']=='E_orig' and r['mode']==mode and r['prefix']==prefix and r['task']=='T2G'}
                for control in ['S','SP']:
                    cr={identity(r):r for r in refs if r['arm']==control and r['mode']==mode and r['prefix']==prefix and r['task']=='T2G'}
                    for method in arms(phase):
                        new={identity(r):r for r in rows if r['arm']==method and r['mode']==mode and r['prefix']==prefix and r['task']=='T2G'};counts=collections.Counter()
                        for k,c in cr.items():
                            b=baseline[k];v=scores.get(c['judge_key']);bv=scores.get(b['judge_key']);nv=scores.get(new[k]['judge_key'])
                            if bv is True and v is False:counts['historical_known_failure']+=1;counts['repaired' if nv is True else 'not_repaired' if nv is False else 'missing']+=1
                        analysis[mode+'/'+str(prefix)+'/'+control+'/'+method]=dict(counts)
            write(ROOT/'public/GRADIENT_MECHANISM_ANALYSIS.json',dict(status='COMPLETE',methods=mm,exposed_T2G_posthoc=analysis,selection_usage=False,retraining_usage=False,causal_claim=False,raw_D_plus_not_Adam_guarantee=True))
    markdown(phase+'_RESULTS_ZH.md','# '+phase+' 结果\n\n仅发布脱敏聚合。共享 rho 只依据 CHECK；正式结果不用于重新选择或训练。'+ ('历史 SP REG 未执行，不能声称有 SP REG 配对基线。' if phase=='REG' else ''),dict(aggregates=result,references=reference_stats,mechanisms=mm,fixed_early_cohort=extra))
def close(status,reason,reg=None):
    q=read(ROOT/'QUEUE.json');assert all(j['status']=='COMPLETE' for j in q)
    chosen=read(ROOT/'public/JOINT_GAIN_DECISION.json')['REG_methods'] if (ROOT/'public/JOINT_GAIN_DECISION.json').exists() else []
    # Final selected models alone need binary retention; all recovery is now idle.
    delete_adapters(lambda r:r.startswith('checkpoints/') or (r.startswith('adapters/') and r.split('/')[2] not in chosen))
    from storage import Store
    store=Store(ROOT)
    with store.lock() as d:
        kept=[dict(path=r,bytes=a['bytes'],reason='selected final adapter') for r,a in d['artifacts'].items() if a['status']=='READY' and r.startswith('adapters/')]
        cleanup=dict(deleted_count=d['deleted_count'],deleted_bytes=d['deleted_bytes'],retained_adapters=kept,one_latest_per_active_worker=True,active_workers=0,old_PR7_PR8_untouched=True)
    write(ROOT/'public/CHECKPOINT_LIFECYCLE.json',cleanup)
    ledger=read(ROOT/'RESOURCE_LEDGER.json');jobs=collections.Counter(j['phase']+'/'+j['mode'] for j in q);formal=sum(len(j['methods']) for j in q if j['mode']=='train');sessions=ledger['gpu_sessions']
    audit=dict(status=status,reason=reason,jobs=dict(jobs),formal_continuations=formal,formal_steps=80*formal,REG_status=reg or ('COMPLETE' if status=='COMPLETE' else 'NOT_ADMITTED'),inherited_judge_attempts=ledger['historical_judge_attempts'],new_judge_attempts=ledger['current_judge_attempts'],cumulative_judge_attempts=ledger['judge_submission_attempt_items'],new_physical_requests=ledger['physical_requests']-read(PathRoot()/'RESOURCE_LEDGER.json')['physical_requests'],gpu_hours=ledger['current_gpu_seconds']/3600,all_gpu_leases_ended=all(s.get('ended_epoch') for s in sessions),GPU_only=[5,6,7],old_results_immutable=True,public_delivery='PENDING_GITHUB',epoch=time.time())
    write(ROOT/'public/FINAL_EXECUTION_AUDIT.json',audit)
    if not (ROOT/'public/REG_RESULTS_ZH.md').exists():markdown('REG_RESULTS_ZH.md','# REG 未启动',dict(status='NOT_ADMITTED',reason=reason,no_REG_rho_adjustment=True))
    if not (ROOT/'public/DEV_RESULTS_ZH.md').exists():markdown('DEV_RESULTS_ZH.md','# Full DEV 未启动',dict(status='NOT_ADMITTED',reason=reason))
    if not (ROOT/'public/JOINT_GAIN_DECISION.json').exists():write(ROOT/'public/JOINT_GAIN_DECISION.json',dict(status='NOT_ADMITTED',reason=reason,REG_allowed=False,REG_methods=[]))
    if not (ROOT/'public/GRADIENT_MECHANISM_ANALYSIS.json').exists():write(ROOT/'public/GRADIENT_MECHANISM_ANALYSIS.json',dict(status='NOT_EVALUATED',reason=reason))
    markdown('FINAL_RESULTS_ZH.md','# Edit-Guarded Protection 最终执行结果\n\n未建立明确 PASS 时不启动 REG；不把机制诊断或 D_plus 当作 Adam 单调保证。',audit)
def PathRoot():
    from pathlib import Path
    return Path(read(ROOT/'PREDECESSOR.json')['pr8_root'])
