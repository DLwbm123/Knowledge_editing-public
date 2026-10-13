"""All preregistered panels and paired edit/source uncertainty, without selection."""
import sqlite3
import astra_report as ar
import paper as exp
import paper_queue as queue
c,RUN,r=exp.c,exp.RUN,ar.r

def main():
    root=queue.q.ROOT;assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')};records=[]
    for x in db.execute('SELECT * FROM consumer'):
        d=c.read(x['path']);assert c.digest(d)==x['output_binding'];records.append((d,x['payload_key']))
    assert len(records)==10395
    labels=['A_W0_'+route for route in exp.A_ROUTES]+['B_'+arm+'_'+route for arm in exp.ARMS for route in exp.ROUTES]+['D_'+arm+'_INTRINSIC_MODAL_MIX' for arm in ('TT_ALIGN',)]
    panels=[];coefs={};groups={}
    for label in labels:
        n=146 if label.startswith('A_') else 24
        for task in ('T0','T1G','T2G','T1L','T2L'):
            m,coef,group=r.panel(records,scores,'RETRO146' if n==146 else 'P2',label,0 if n==146 else 160,'bank_R0',task,exp.tasks(n),n)
            m.update(ar.bootstrap(coef,scores,group));panels.append(m);coefs[label,task]=coef;groups.update(group)
    pairs=[('A_W0_MODAL','A_W0_NORM_MEAN'),('A_W0_INTRINSIC_MODAL','A_W0_INTRINSIC_ALL')]
    for route in exp.ROUTES:
        pairs += [('B_'+kind+'_ALIGN_'+route,'B_'+kind+'_CE_'+route) for kind in ('TT',)]
        pairs += [('B_TT_FIXED_A_'+route,'B_TT_CE_'+route)]
    pairs += [('D_'+arm+'_INTRINSIC_MODAL_MIX','B_'+arm+'_INTRINSIC_MODAL') for arm in ('TT_ALIGN',)]
    contrasts=[]
    for a,b in pairs:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            aa,bb=coefs[a,task],coefs[b,task];assert set(aa)==set(bb)
            cc={e:r.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa}
            bound=r.score_bounds(r.combine([{k:v/len(cc) for k,v in value.items()} for value in cc.values()]),scores) if cc else [None,None]
            contrasts.append(dict(a=a,b=b,task=task,paired_edits=len(cc),delta_bounds_pp=[100*v if v is not None else None for v in bound],**ar.bootstrap(cc,scores,groups)))
    # Aggregate CHECK by condition directly: each row is a bank context, not a patient.
    u=[]
    for label in labels:
        ds=[d for d,_ in records if d['binding']['arm']==label and d['binding']['input'].get('role')=='CHECK']
        assert len(ds)==4 and all(d['U_KL'] is not None for d in ds)
        u.append(dict(arm=label,bank_contexts=1,observations=4,source_groups=len({d['binding']['input']['source_group'] for d in ds}),mean_KL=sum(d['U_KL'] for d in ds)/4,Base_token_consistency=sum(d['Base_token_consistency'] for d in ds)/4,route_ON=sum(d['effective_expert'] is not None for d in ds),patient='UNKNOWN',medical_accuracy=False))
    resource=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    result=dict(status='TERMINAL_WITH_MISSING' if any(x is None for x in scores.values()) else 'COMPLETE_NO_MISSING',panels=panels,contrasts=contrasts,CHECK=u,scoring=c.read(root/'READY.json'),missing_payloads=sum(x is None for x in scores.values()),Judge_attempts=resource['Judge_attempts'],new_Judge_attempts=resource['Judge_attempts']-inherited['Judge_attempts'],GPU_hours=c.used()/3600,new_GPU_hours=(c.used()-inherited['gpu_seconds_used'])/3600,independent_confirmation=False,medical_protection='NA_SCOPE_NOT_QUALIFIED',scope_amendment=c.read(RUN/'private/TT_ONLY_AMENDMENT.json'),coverage=dict(TIME='modality summaries, intrinsic routing, contrastive routing alignment, sparse soft mixture',MORE='modality summaries and fixed input coordinates ablation',not_implemented=['shared factor generator requires separate editor-training protocol','shared recursive write is incompatible with independent moving TT expert coordinates']))
    c.write(RUN/'public/RESULTS.json',result)
    text='# TIME / M-ORE 模块迁移开发实验\n\n全部固定条件及完整面板，未按结果择优。原146及24编辑都是已暴露DEV；医学作用域资格与独立确认均未满足。\n\n|条件|任务|区间/指标|\n|---|---|---|\n'
    for m in panels:text+='|'+str(m.get('arm',m.get('method','')))+'|'+str(m.get('task',''))+'|'+str(m['macro_bounds'])+'|\n'
    text+='\n完整面板、成对差值、编辑/来源bootstrap、缺失界与CHECK诊断见RESULTS.json。全部专家保持TT88；CP正式训练前已按用户指令取消。没有复现TIME共享生成器或M-ORE共享递推权重。\n'
    (RUN/'public/REPORT_ZH.md').write_text(text)
    exp.done('REPORT_COMPLETE');exp.progress('RESULTS_COMPLETE_PUBLICATION_PENDING')

if __name__=='__main__':main()
