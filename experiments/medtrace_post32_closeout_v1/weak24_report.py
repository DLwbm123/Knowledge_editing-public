"""All four cells and fixed contrasts for the exposed 24-edit interaction check."""
import json
import sqlite3
import time
import astra_report as ar
import weak24 as exp
import os
os.environ['JUDGE_PHASE']='WEAK24'
import combo24_queue as queue

RUN,r,common=ar.RUN,ar.r,ar.common


def main():
    root=queue.q.ROOT;assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')}
    records=[]
    for c in db.execute('SELECT * FROM consumer'):
        d=common.read(c['path']);assert common.digest(d)==c['output_binding'];records.append((d,c['payload_key']))
    assert len(records)==849
    panels=[];coefs={};groups={}
    for arm in exp.ARMS:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            m,c,g=r.panel(records,scores,'P2',arm,160,'bank_R0',task,exp.tasks(),24)
            m.update(ar.bootstrap(c,scores,g));panels.append(m);coefs[arm,task]=c;groups.update(g)
    contrasts=[]
    for a,b,label in [(2,0,'Weaker U versus CE'),(2,1,'Weaker U versus original U')]:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            aa,bb=coefs[exp.ARMS[a],task],coefs[exp.ARMS[b],task];assert set(aa)==set(bb)
            cc={e:r.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa}
            bound=r.score_bounds(r.combine([{k:v/len(cc) for k,v in c.items()} for c in cc.values()]),scores) if cc else [None,None]
            contrasts.append(dict(comparison=label,task=task,paired_edits=len(cc),delta_bounds_pp=[v*100 if v is not None else None for v in bound],**ar.bootstrap(cc,scores,groups)))
    u=r.summarize_u(records,'P2')
    for x in u:x['bank_contexts']=x.pop('edits');x['bank_source_relations']=x.pop('edit_source_relations')
    missing=sum(v is None for v in scores.values())
    result=dict(status='COMPLETE_NO_MISSING' if not missing else 'TERMINAL_WITH_MISSING',panels=panels,contrasts=contrasts,U=u,
        primary_U='T2G weaker U minus CE with R0',secondary='T2G weaker U minus original U',edits=24,seed_repeats=1,
        all24_weights_updated=True,new_weights=24,baseline_outputs_reused=566,new_outputs=283,training_required=True,U_weight=.001,
        fixed_final_bank_only=True,independent_confirmation=False,scope_qualified_sources=0,medical_protection='NA',
        scoring=common.read(root/'READY.json'),missing_payloads=missing,Judge_attempts=common.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts'])
    curves=[common.read(exp.point(t).parent/'TRAINING.json') for t in exp.tasks()]
    result['training']=dict(updates=sum(x['actual_updates'] for x in curves),seconds=sum(x['seconds'] for x in curves),forwards=sum(x['forwards'] for x in curves),backwards=sum(x['backwards'] for x in curves),mean_U_weighted_fraction=sum(c['U_weighted_fraction'] for x in curves for c in x['curve'])/3840,mean_CE_U_cosine=sum(c['CE_U_cosine'] for x in curves for c in x['curve'])/3840)
    common.write(RUN/'public/WEAK24_RESULTS.json',result)
    text='# 全部24编辑：较弱U系数开发比较\n\n新增U=0.001条件；CE与U=0.01复用原输出。相同W0、种子、160步及R0路由，最终完整24库。保留既有阴性结果。\n\n|比较|任务|差值下界pp|差值上界pp|\n|---|---|---:|---:|\n'
    for c in contrasts:text+='|'+'|'.join(map(str,[c['comparison'],c['task'],*c['delta_bounds_pp']]))+'|\n'
    text+='\n缺失保留分母；已有失败的完全相同payload继承缺失，不隐性重试。全部面板、编辑/来源区间及自然路由CHECK KL/精确Base-token一致性见WEAK24_RESULTS.json。仅检验一个预登记系数，不据结果选系数或种子；病例独立性未知、医学保护资格0。\n'
    (RUN/'public/WEAK24_REPORT_ZH.md').write_text(text)
    common.write(root/'REPORT_COMPLETE.json',dict(status='AGGREGATES_COMPLETE_PUBLICATION_PENDING',missing=missing,epoch=time.time()))
    common.write(RUN/'public/WEAK24_PROGRESS.json',dict(status='RESULTS_COMPLETE_PUBLICATION_PENDING',complete=False,missing=missing))

if __name__=='__main__':main()
