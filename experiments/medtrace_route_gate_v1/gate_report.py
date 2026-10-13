"""All fixed route contrasts, uncertainty and provenance; no winner selection."""
import sqlite3
import collections
import astra_report as ar
import gate as exp
import gate_queue as queue
c,RUN,r=exp.c,exp.RUN,ar.r

def main():
    root=queue.q.ROOT;assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')};records=[]
    for x in db.execute('SELECT * FROM consumer'):
        d=c.read(x['path']);assert c.digest(d)==x['output_binding'];records.append((d,x['payload_key']))
    assert len(records)==10591
    labels=list(exp.ARMS['G']+exp.ARMS['P']);panels=[];coefs={};groups={}
    for label in labels:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            m,coef,group=r.panel(records,scores,'RETRO146',label,0,'bank_R0',task,exp.p.tasks(),146)
            m.update(ar.bootstrap(coef,scores,group));panels.append(m);coefs[label,task]=coef;groups.update(group)
    pairs=[('G_MODAL_R0_GATE','G_OLD_MODAL'),('G_MODAL_R0_GATE','G_R0'),('G_MODAL_R0_CANDIDATES','G_R0'),('G_MODAL_R0_CANDIDATES','G_MODAL_R0_GATE'),('P_LAST_SUBSPACE_R0_CANDIDATES','P_MEAN_SUBSPACE_R0_CANDIDATES'),('P_LAST_RESIDUAL_R0_CANDIDATES','P_LAST_SUBSPACE_R0_CANDIDATES')]+[(a,'G_R0') for a in exp.ARMS['P']]
    contrasts=[]
    for a,b in pairs:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            aa,bb=coefs[a,task],coefs[b,task];assert set(aa)==set(bb)
            cc={e:r.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa}
            bound=r.score_bounds(r.combine([{k:v/len(cc) for k,v in value.items()} for value in cc.values()]),scores) if cc else [None,None]
            contrasts.append(dict(a=a,b=b,task=task,paired_edits=len(cc),delta_bounds_pp=[100*v if v is not None else None for v in bound],**ar.bootstrap(cc,scores,groups)))
    checks=[];admission={};generation={}
    for label in labels:
        ds=[d for d,_ in records if d['binding']['arm']==label]
        if label!='G_OLD_MODAL':assert all(d['route']['activated']==d['route']['original_R0']['activated'] for d in ds)
        admission[label]=dict(queries=len(ds),ON=sum(d['route']['activated'] for d in ds),gate_changed=sum(d['route']['activated']!=d['route']['original_R0']['activated'] for d in ds))
        generation[label]=dict(reused=sum(d['reused_output_binding'] is not None for d in ds),new_generation=sum(d['reused_output_binding'] is None for d in ds))
        ds=[d for d in ds if d['binding']['input'].get('role')=='CHECK'];assert len(ds)==4 and all(d['U_KL'] is not None for d in ds)
        checks.append(dict(arm=label,observations=4,source_groups=4,mean_KL=sum(d['U_KL'] for d in ds)/4,Base_token_consistency=sum(d['Base_token_consistency'] for d in ds)/4,route_ON=sum(d['route']['activated'] for d in ds),medical_accuracy=False))
    resource=c.read(RUN/'RESOURCE_LEDGER.json');inherited=c.read(RUN/'private/INHERITED_COST.json')
    result=dict(status='TERMINAL_WITH_MISSING' if any(x is None for x in scores.values()) else 'COMPLETE_NO_MISSING',panels=panels,contrasts=contrasts,CHECK=checks,admission=admission,generation=generation,scoring=c.read(root/'READY.json'),missing_payloads=sum(x is None for x in scores.values()),Judge_attempts=resource['Judge_attempts'],new_Judge_attempts=resource['Judge_attempts']-inherited['Judge_attempts'],GPU_process_hours=c.used()/3600,new_GPU_process_hours=(c.used()-inherited['gpu_seconds_used'])/3600,new_training=0,CP_enabled=False,independent_confirmation=False,medical_protection='NA_SCOPE_NOT_QUALIFIED',later_training='DATA_GATE_NOT_MET',preregistration=c.read(RUN/'private/GATE_LOCK.json')['public_protocol'])
    c.write(RUN/'public/RESULTS.json',result)
    text='# 冻结 TT 路由分解实验\n\n全部固定条件；已暴露DEV；没有独立确认。区间为评分缺失界。\n\n|条件|任务|宏平均百分比界|\n|---|---|---|\n'
    for m in panels:text+='|'+m['arm']+'|'+m['task']+'|'+str(m['macro_bounds'])+'|\n'
    text+='\n完整配对bootstrap、放行不变性、CHECK与成本见RESULTS.json。独立门槛学习、ALIGN与混合未通过校准数据门槛，未启动；CP保持取消。\n'
    (RUN/'public/REPORT_ZH.md').write_text(text);exp.p.done('REPORT_COMPLETE');exp.p.progress('RESULTS_COMPLETE_PUBLICATION_PENDING')

if __name__=='__main__':main()
