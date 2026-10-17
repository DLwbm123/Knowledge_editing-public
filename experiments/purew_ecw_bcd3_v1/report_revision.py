"""Four-method frozen cohort; exact shared missing bounds, no average protection gate."""
from collections import Counter
import json,sqlite3,time,os,sys
sys.path.insert(0,os.environ["RUN_ROOT"]+"/private/source")
sys.path.insert(0,os.environ["RUN_ROOT"]+"/private/tools")
import astra_report as ar
from panels import protection_ids
from logic import transitions
from report_fs import audit_requests
R,c,r=ar.RUN,ar.common,ar.r
ARMS=('A','B','C','D')
CONTRASTS=(('B','A'),('C','B'),('D','B'))

def main():
    marker=R/'private/REPORT_COMPLETE.json'
    if marker.exists():print(c.read(R/'public/RESULTS.json')['status']);return
    cost=c.read(R/'RESOURCE_LEDGER.json');assert all(x.get('ended_epoch') for x in cost['gpu_sessions'])
    full=(R/'private/GENERATION_COMPLETE.json').exists() and (R/'private/judge_purew146/ALL_WORKERS_COMPLETE.json').exists()
    assert full,'Partial needs explicit retained-state review; never label unattempted missing'
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    rows=list(db.execute('SELECT * FROM consumer'));assert len(rows)==644
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')};scores={k:x['correct'] for k,x in payloads.items()};audit_requests(payloads,cost)
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json');records=[];lookup={}
    for x in rows:
        d=c.read(x['path']);assert c.digest(d)==x['output_binding'];bound=json.loads(payloads[x['payload_key']]['binding']);query=ledger['queries'][x['query_id']]
        assert bound['output']==d['R0'] and all(bound[k]==query[k] for k in ['query_id','question','reference','image_sha256'])
        active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:x['prefix']]} if x['mode']=='bank_R0' else set()
        records.append((dict(binding=dict(phase=dict(arm=x['method'],node=0,prefix=x['prefix']),input=query,mode=x['mode'],owner_order=x['edit_order']),active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),x['payload_key']))
        lookup[x['method'],x['mode'],x['prefix'],x['edit_order'],x['query_id']]=x['payload_key']
    panels=[];coefs={};groups={}
    for arm in ARMS:
        for prefix in (1,3):
            for task in ('T0','T1G','T2G','T1L','T2L'):
                metric,co,g=r.panel(records,scores,'RETRO146',arm,0,'bank_R0',task,tasks[:prefix],prefix);metric.pop('route_ON',None);n=metric['observations']
                metric['micro_bounds']=[100*metric['known_correct']/n,100*(metric['known_correct']+metric['missing'])/n] if n else None
                panels.append(metric);coefs[arm,prefix,task]=co;groups.update(g)
    contrasts=[]
    # A is also the common mechanism reference for per-edit satisfaction.
    for a,b in dict.fromkeys(CONTRASTS+(('C','A'),('D','A'))):
        ca,cb=coefs[a,3,'T2G'],coefs[b,3,'T2G'];assert set(ca)==set(cb)
        delta={e:r.combine([ca[e],{k:-v for k,v in cb[e].items()}]) for e in ca}
        bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in x.items()} for x in delta.values()]),scores)
        contrasts.append(dict(comparison=a+'-'+b,primary=(a,b) in CONTRASTS,delta_macro_bounds_pp=[100*x for x in bounds],
            negative_edit_bounds=sum(r.score_bounds(x,scores)[0]<-1e-12 for x in delta.values()),**ar.bootstrap(delta,scores,groups)))
    protection=[];endpoints=[];gates={};diagnostics={}
    for arm in ARMS:
        protected=True
        for task in ('T1L','T2L'):
            ids=protection_ids(tasks,ledger,task);assert len(ids)==(2 if task=='T1L' else 34)
            keys=[lookup[arm,'protect100_R0',100,0,q] for q in ids];correct=sum(scores[k]==1 for k in keys);unknown=sum(scores[k] is None for k in keys)
            protection.append(dict(arm=arm,task=task,observations=len(ids),known_correct=correct,missing=unknown,Base_correct_to_wrong_bounds=[len(ids)-correct-unknown,len(ids)-correct]))
            protected &= correct==len(ids)
            for a,b in CONTRASTS:
                if arm==a:endpoints.append(dict(task=task,comparison=b+'-to-'+a,**transitions([(lookup[b,'protect100_R0',100,0,q],lookup[a,'protect100_R0',100,0,q]) for q in ids],scores)))
        t1=next(x for x in panels if x['arm']==arm and x['prefix']==3 and x['task']=='T1G')
        gates[arm]=dict(zero_individual_Base_damage=protected,T0_insertion_and_final=all(scores[lookup[arm,'insertion_R0',t['order'],t['order'],t['edit_id']]]==scores[lookup[arm,'bank_R0',3,0,t['edit_id']]]==1 for t in tasks[:3]),
            T1G_full_retention=t1['known_correct']==t1['observations'],full_scoring=all(scores[x['payload_key']] is not None for x in rows if x['method']==arm))
        if arm!='A':
            ls=[]
            for owner in (1,2,3):
                events=sorted([c.read(p) for p in (R/'private/steps'/arm/str(owner)).glob('*.json')],key=lambda e:e['slot']);assert len(events)==20
                source=c.read(R/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json'))
                reasons=Counter(e.get('rejection_reason',z) for e in events if not e['accepted'] for z in (set(a for candidate in e['candidates'] for a in candidate['reasons']) or {'NO_NATIVE_ACCEPT'}))
                ls.append(dict(source_position=owner,slots=20,accepted=sum(e['accepted'] for e in events),zero_fraction=sum(e['zero_native'] for e in events)/20,
                    rejection_flags=dict(reasons),measurements=source,final_block_risks=events[-1].get('risks',events[-1]['current_risk']),
                    numeric_domination_slots=sum(any(v['numeric_envelope_dominates'] for z in e['candidates'] for v in z['risks'].values()) for e in events),
                    final_path_and_net=c.read(R/'public'/('PATH_'+arm+'_'+str(owner)+'.json'))))
            diagnostics[arm]=ls
            gates[arm].update(all_source_CE_below_point1=all(x['measurements']['final']['weighted_all_FIT_CE']<.1 for x in ls),
                per_edit_T2G_not_worse=all(x['negative_edit_bounds']==0 for x in contrasts if x['comparison'].startswith(arm+'-')))
        for task in ('T0','T2G'):
            pairs=[]
            for t in tasks[:3]:
                ids=[t['edit_id']] if task=='T0' else [q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]
                pairs.extend((lookup[arm,'insertion_R0',t['order'],t['order'],q],lookup[arm,'bank_R0',3,0,q]) for q in dict.fromkeys(ids))
            endpoints.append(dict(arm=arm,task=task,comparison='actual-insertion-to-prefix3',**transitions(pairs,scores)))
    unknown=sum(scores[x['payload_key']] is None for x in rows)
    result=dict(status='COMPLETE_WITH_MISSING' if unknown else 'COMPLETE',scientific_success=not unknown and all(all(gates[a].values()) for a in ('B','C','D')),
        panels=panels,contrasts=contrasts,protection=protection,endpoints=endpoints,gates=gates,diagnostics=diagnostics,missing_consumers=unknown,
        new_Judge=cost['Judge_attempts'],GPU_process_hours=cost['gpu_seconds_used']/3600,
        actual_counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ['forwards','vision_forwards','backwards','committed_updates','attempt_slots','generation_attempts']},
        GPU_stage_failures=len(list((R/'private/failures').glob('*.json'))),all_consumers=644,A_immutable_reused_consumers=161,followup146_activated=False,
        calibration_coverage='CT/MRI eight training-source groups; KG/XRay absent; patient independence unknown',
        nonlinear_certificate=False,clinical_or_privacy_certificate=False,K_matched_first3='NA_NO_REBUILD')
    c.write(R/'public/RESULTS.json',result)
    text='# B/C/D前三源双锚有限对照\n\n'+result['status']+'。完整评分终态不等于逐题保护成功。\n\n|臂|任务|正确/分母|缺分|微平均上下界%|\n|---|---|---:|---:|---|\n'
    for x in panels:
        if x['prefix']==3:text+='|'+ '|'.join(map(str,[x['arm'],x['task'],str(x['known_correct'])+'/'+str(x['observations']),x['missing'],x['micro_bounds']]))+'|\n'
    text+='\n固定34 T2L+2 T1L的损伤与修复独立见RESULTS.json；缺分未知不通过零新增损伤门槛。原seed10000次edit/source配对区间、共享missing精确界、逐编辑退化、实际插入端点、源CE/拒绝/块风险和所有实际失败成本保留。\n\nA为旧Y只读复用。B加入一次Base校准，C再加逐视图硬KL和逐概念证据间隔，D只增历史answer probes。完整有限旋转因子无截断；FP64数值误差包络不覆盖采样及非线性误差。三源非医学/临床/146可扩展结论，无自动后续实验。原146/M暂停，O/L等待。\n'
    (R/'public/REPORT_ZH.md').write_text(text);c.write(marker,dict(status=result['status'],epoch=time.time()));print(result['status'])

if __name__=='__main__':main()
