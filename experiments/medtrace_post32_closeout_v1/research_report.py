"""Paired mechanism report; no independent confirmation or medical-protection claim."""
import json
import sqlite3
import time
import astra_report as ar
import research_queue as queue
import research

RUN,r,common=ar.RUN,ar.r,ar.common


def main():
    root=queue.q.ROOT
    assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')}
    records=[]
    for c in db.execute('SELECT * FROM consumer'):
        d=common.read(c['path']);assert common.digest(d)==c['output_binding'];records.append((d,c['payload_key']))
    # Compact original146 baseline with all accepted/recovered Astra scores.
    parent=sqlite3.connect('file:'+str(RUN/'private/judge_astra_medium/queue.sqlite')+'?mode=ro',uri=True);parent.row_factory=sqlite3.Row
    scores.update({k:v['correct'] for k,(v,_) in queue.parent_rows().items()})
    ts146=common.read(RUN/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=common.read(RUN/'private/EVAL_LEDGER.json')
    active={(t['native']['image_sha256'],t['native']['question']) for t in ts146}
    for c in parent.execute("SELECT * FROM consumer WHERE method='TT88_W0_RETRO146' AND mode='bank_R0' AND prefix=146"):
        q=ledger['queries'][c['query_id']]
        records.append((dict(binding=dict(phase=dict(arm=c['method'],node=0,prefix=146),input=q,mode=c['mode'],owner_order=c['edit_order']),active_target=(q['image_sha256'],q['question']) in active,effective_expert=None),c['payload_key']))
    panels=[];coefs={};groups={};small=research.selected()
    specs=[('P2',arm,160,mode,n,small[:n] if mode=='bank_R0' else small[:8]) for arm in research.ARMS for mode,n in [('single_R0',1),('bank_R0',8),('bank_R0',24)]]
    specs += [('RETRO146',arm,0,'bank_R0',146,ts146) for arm in ('TT88_W0_RETRO146',research.ROUTE_ARM)]
    for cohort,arm,node,mode,n,ts in specs:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            m,c,g=r.panel(records,scores,cohort,arm,node,mode,task,ts,n)
            m.update(ar.bootstrap(c,scores,g));panels.append(m);coefs[(arm,mode,n,task)]=c;groups.update(g)
    contrasts=[]
    for a,b,mode,n in [(research.ARMS[1],research.ARMS[0],mode,n) for mode,n in [('single_R0',1),('bank_R0',8),('bank_R0',24)]]+[(research.ROUTE_ARM,'TT88_W0_RETRO146','bank_R0',146)]:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            aa,bb=coefs[a,mode,n,task],coefs[b,mode,n,task];assert set(aa)==set(bb)
            cc={e:r.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa}
            bound=r.score_bounds(r.combine([{k:v/len(cc) for k,v in c.items()} for c in cc.values()]),scores) if cc else [None,None]
            contrasts.append(dict(comparison=a+' minus '+b,mode=mode,prefix=n,task=task,paired_edits=len(cc),delta_bounds_pp=[v*100 if v is not None else None for v in bound],**ar.bootstrap(cc,scores,groups)))
    gradient=[]
    for slot in (0,1):
        for arm,old in zip(research.ARMS,('CE_ONLY','CE_U_MULTI')):
            curves=[]
            for t in small[:8]:
                p=(research.BASE/'private/edits'/t['anonymous_edit']/'s0'/old if slot==0 else RUN/'private/research/u'/t['anonymous_edit']/'s1'/arm)/'TRAINING.json'
                curves+=common.read(p)['curve']
            mean=lambda xs:sum(xs)/len(xs) if xs else None
            gradient.append(dict(arm=arm,slot=slot,updates=len(curves),mean_U_weighted_fraction=mean([x['U_weighted_fraction'] for x in curves if x['U_weighted_fraction'] is not None]),mean_CE_U_cosine=mean([x['CE_U_cosine'] for x in curves if x['CE_U_cosine'] is not None]),forwards=sum(x['forwards'] for x in curves),backwards=sum(x['backwards'] for x in curves)))
    missing=db.execute("SELECT count(*) FROM payload WHERE status!='FORMAT_VALID'").fetchone()[0]
    result=dict(status='COMPLETE_NO_MISSING' if not missing else 'TERMINAL_WITH_MISSING',panels=panels,contrasts=contrasts,U=r.summarize_u(records,'P2'),gradients=gradient,
        primary_U='CE+U minus CE T2G macro at bank8, two matched seed repeats',primary_routing='FITKEY5 minus R0 T1G macro at bank146',
        U_training_edits=8,U_seed_repeats=2,bank24_background16='unchanged W0',route_evaluation='Final frozen bank146; not a new sequential-order benchmark',
        independent_confirmation=False,scope_qualified_sources=0,medical_protection='NA',missing_payloads=missing,Judge_attempts=common.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts'])
    common.write(RUN/'public/RESEARCH_RESULTS.json',result)
    text='# 路由泛化与 U 净收益：两线开发实验\n\n原146和旧CAL/CHECK均已经暴露，结果仅用于开发；不声明独立确认或医学保护。U以同W0、同seed、同CE批次和160步的CE为对照；U多一次KL前反向，真实成本单独记录。24库只更新前8，另16固定W0。路由只增加native及已有4条FIT改写锚点，半径/专家权重/生成不变；仅评估最终146库。\n\n|比较|模式|库大小|任务|差值下界pp|差值上界pp|\n|---|---|---:|---|---:|---:|\n'
    for c in contrasts:
        text+='|'+'|'.join(map(str,[c['comparison'],c['mode'],c['prefix'],c['task'],*c['delta_bounds_pp']]))+'|\n'
    text+='\n差值界保留缺失分母；配对置信区间、完整面板、U强制启用/自然路由KL与梯度诊断见RESEARCH_RESULTS.json。不能将保护输入路由关闭产生的零KL称作U有效。所有正负结果均保留。\n'
    (RUN/'public/RESEARCH_REPORT_ZH.md').write_text(text)
    common.write(RUN/'public/RESEARCH_PROGRESS.json',dict(status='RESULTS_COMPLETE_PUBLICATION_PENDING',complete=False,missing=missing))
    common.write(root/'REPORT_COMPLETE.json',dict(status='AGGREGATES_COMPLETE_PUBLICATION_PENDING',epoch=time.time()))

if __name__=='__main__':main()
