"""All four cells and fixed contrasts for the exposed 24-edit interaction check."""
import json
import sqlite3
import time
import astra_report as ar
import combo24 as exp
import combo24_queue as queue

RUN,r,common=ar.RUN,ar.r,ar.common


def main():
    root=queue.q.ROOT;assert (root/'ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(root/'queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    scores={x['key']:x['correct'] for x in db.execute('SELECT * FROM payload')}
    records=[]
    for c in db.execute('SELECT * FROM consumer'):
        d=common.read(c['path']);assert common.digest(d)==c['output_binding'];records.append((d,c['payload_key']))
    assert len(records)==1132
    panels=[];coefs={};groups={}
    for arm in exp.ARMS:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            m,c,g=r.panel(records,scores,'P2',arm,160,'bank_R0',task,exp.tasks(),24)
            m.update(ar.bootstrap(c,scores,g));panels.append(m);coefs[arm,task]=c;groups.update(g)
    contrasts=[]
    for a,b,label in [(1,0,'U at R0'),(3,2,'U at FITKEY5'),(2,0,'FITKEY5 with CE'),(3,1,'FITKEY5 with CEU'),(3,0,'Combined versus CE R0')]:
        for task in ('T0','T1G','T2G','T1L','T2L'):
            aa,bb=coefs[exp.ARMS[a],task],coefs[exp.ARMS[b],task];assert set(aa)==set(bb)
            cc={e:r.combine([aa[e],{k:-v for k,v in bb[e].items()}]) for e in aa}
            bound=r.score_bounds(r.combine([{k:v/len(cc) for k,v in c.items()} for c in cc.values()]),scores) if cc else [None,None]
            contrasts.append(dict(comparison=label,task=task,paired_edits=len(cc),delta_bounds_pp=[v*100 if v is not None else None for v in bound],**ar.bootstrap(cc,scores,groups)))
    u=r.summarize_u(records,'P2')
    for x in u:x['bank_contexts']=x.pop('edits');x['bank_source_relations']=x.pop('edit_source_relations')
    missing=sum(v is None for v in scores.values())
    result=dict(status='COMPLETE_NO_MISSING' if not missing else 'TERMINAL_WITH_MISSING',panels=panels,contrasts=contrasts,U=u,
        primary_U='T2G CEU minus CE with R0',primary_route='T1G FITKEY5 minus R0 with CEU',edits=24,seed_repeats=1,
        all24_weights_updated=True,weights_reused=48,baseline_outputs_reused=566,new_outputs=566,training_required=False,
        fixed_final_bank_only=True,independent_confirmation=False,scope_qualified_sources=0,medical_protection='NA',
        scoring=common.read(root/'READY.json'),missing_payloads=missing,Judge_attempts=common.read(RUN/'RESOURCE_LEDGER.json')['Judge_attempts'])
    common.write(RUN/'public/COMBO24_RESULTS.json',result)
    text='# 24编辑 U × 路由组合：冻结权重开发比较\n\n复用48份既有160步权重与566个原R0输出，仅新增566个FITKEY5输出。所有24专家均为对应CE/CEU条件，一个既定seed；不是此前仅前8更新的24库。旧CHECK只有4个来源，非独立确认。\n\n|比较|任务|差值下界pp|差值上界pp|\n|---|---|---:|---:|\n'
    for c in contrasts:text+='|'+'|'.join(map(str,[c['comparison'],c['task'],*c['delta_bounds_pp']]))+'|\n'
    text+='\n缺失保留分母；已有失败的完全相同payload继承缺失，不隐性重试。全部面板、编辑/来源区间及自然路由CHECK KL/精确Base-token一致性见COMBO24_RESULTS.json。组合效果不得归给单一因素；病例独立性未知、医学保护资格0。\n'
    (RUN/'public/COMBO24_REPORT_ZH.md').write_text(text)
    common.write(root/'REPORT_COMPLETE.json',dict(status='AGGREGATES_COMPLETE_PUBLICATION_PENDING',missing=missing,epoch=time.time()))
    common.write(RUN/'public/COMBO24_PROGRESS.json',dict(status='RESULTS_COMPLETE_PUBLICATION_PENDING',complete=False,missing=missing))

if __name__=='__main__':main()
