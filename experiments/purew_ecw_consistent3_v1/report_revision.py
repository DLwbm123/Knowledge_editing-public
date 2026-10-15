"""Actual terminal cohorts only; historical B unknowns never become new Judge requests."""
from collections import Counter
import json,os,sqlite3,time
from pathlib import Path
import astra_report as ar
from panels import protection_ids
from logic import transitions
from report_fs import audit_requests
from historical_binding import binding_key
R,c,r=ar.RUN,ar.common,ar.r
ACTIVE=tuple(c.read(R/'private/FROZEN_PROTOCOL.json').get('active_arms',('N','J','NJ')))
ARMS=('B',)+ACTIVE;CONTRASTS=tuple((a,b) for a,b in (('N','B'),('J','B'),('NJ','J'),('NJ','N')) if a in ARMS and b in ARMS)

def main():
    marker=R/'private/REPORT_COMPLETE.json'
    if marker.exists():print(c.read(R/'public/RESULTS.json')['status']);return
    cost=c.read(R/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in cost['gpu_sessions'])
    gen=c.read(R/'private/GENERATION_COMPLETE.json');terminals=c.read(R/'private/ARM_TERMINALS.json')
    assert (R/'private/FINAL_CONSUMED.json').exists() and (R/'private/judge_purew146/ALL_WORKERS_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    rows=list(db.execute('SELECT * FROM consumer'));assert len(rows)==gen['new_consumers']
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')};scores={k:x['correct'] for k,x in payloads.items()};audit_requests(payloads,cost)
    tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json');epoch=c.read(R/'private/judge_purew146/EPOCH_MANIFEST.json')
    records=[];lookup={};coverage={a:Counter() for a in ARMS}
    old=Path(c.read(R/'private/FROZEN_PROTOCOL.json')['reference_run'])
    files=[('B',p,None) for p in (old/'private/outputs/B').glob('*/*/*/*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json')]
    assert len(files)==161
    files.extend((x['method'],Path(x['path']),x['output_binding']) for x in rows)
    for arm,path,locked in files:
        d=c.read(path)
        if locked:assert c.digest(d)==locked
        bb=d['binding'];phase=bb['phase'];query=ledger['queries'][bb['input']['query_id']];assert bb['input']==query
        key,full=binding_key(query,bb['judge_input'],d['R0'],epoch)
        if key in payloads:assert json.loads(payloads[key]['binding'])==full;status=payloads[key]['status']
        else:assert arm=='B';scores[key]=None;status='UNKNOWN_UNATTEMPTED'
        coverage[arm][status]+=1
        active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:phase['prefix']]} if phase['mode']=='bank_R0' else set()
        records.append((dict(binding=dict(phase=dict(arm=arm,node=0,prefix=phase['prefix']),input=query,mode=phase['mode'],owner_order=phase['owner']),active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),key))
        lookup[arm,phase['mode'],phase['prefix'],phase['owner'],query['query_id']]=key
    panels=[];coefs={};groups={}
    for arm in ARMS:
        for prefix in ((1,3) if arm=='B' or terminals[arm]['prefix']==3 else (1,)):
            for task in ('T0','T1G','T2G','T1L','T2L'):
                m,co,g=r.panel(records,scores,'RETRO146',arm,0,'bank_R0',task,tasks[:prefix],prefix);m.pop('route_ON',None);n=m['observations']
                m['micro_bounds']=[100*m['known_correct']/n,100*(m['known_correct']+m['missing'])/n] if n else None
                panels.append(m);coefs[arm,prefix,task]=co;groups.update(g)
    contrasts=[]
    for a,b in CONTRASTS:
        if (a,3,'T2G') not in coefs or (b,3,'T2G') not in coefs:
            contrasts.append(dict(comparison=a+'-'+b,status='NA_MATCHED_PREFIX3_NOT_GENERATED'));continue
        ca,cb=coefs[a,3,'T2G'],coefs[b,3,'T2G'];assert set(ca)==set(cb)
        delta={e:r.combine([ca[e],{k:-v for k,v in cb[e].items()}]) for e in ca}
        bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in x.items()} for x in delta.values()]),scores)
        contrasts.append(dict(comparison=a+'-'+b,status='ESTIMATED',delta_macro_bounds_pp=[100*x for x in bounds],negative_edit_bounds=sum(r.score_bounds(x,scores)[0]<-1e-12 for x in delta.values()),**ar.bootstrap(delta,scores,groups)))
    prefix1=[]
    for a,b in CONTRASTS:
        ca,cb=coefs[a,1,'T2G'],coefs[b,1,'T2G'];assert set(ca)==set(cb)
        delta={e:r.combine([ca[e],{k:-v for k,v in cb[e].items()}]) for e in ca}
        bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in x.items()} for x in delta.values()]),scores)
        prefix1.append(dict(comparison=a+'-'+b,prefix=1,secondary_only=True,delta_macro_bounds_pp=[100*x for x in bounds],**ar.bootstrap(delta,scores,groups)))
    protection=[];endpoints=[];gates={};diagnostics={}
    for arm in ARMS:
        terminal=3 if arm=='B' else terminals[arm]['prefix'];protected=True
        for task in ('T1L','T2L'):
            ids=protection_ids(tasks,ledger,task);keys=[lookup[arm,'protect100_R0',100,0,q] for q in ids]
            correct=sum(scores[k]==1 for k in keys);unknown=sum(scores[k] is None for k in keys)
            protection.append(dict(arm=arm,terminal_prefix=terminal,task=task,observations=len(ids),known_correct=correct,missing=unknown,Base_correct_to_wrong_bounds=[len(ids)-correct-unknown,len(ids)-correct]));protected &=correct==len(ids)
            for a,b in CONTRASTS:
                if arm==a and (3 if b=='B' else terminals[b]['prefix'])==terminal:
                    endpoints.append(dict(task=task,comparison=b+'-to-'+a,matched_terminal_prefix=terminal,**transitions([(lookup[b,'protect100_R0',100,0,q],lookup[a,'protect100_R0',100,0,q]) for q in ids],scores)))
        t1=next(x for x in panels if x['arm']==arm and x['prefix']==terminal and x['task']=='T1G')
        keys=[key for k,key in lookup.items() if k[0]==arm]
        gates[arm]=dict(zero_individual_Base_damage=protected,complete_three_sources=terminal==3,T0_insertion_and_terminal=all(scores[lookup[arm,'insertion_R0',t['order'],t['order'],t['edit_id']]]==scores[lookup[arm,'bank_R0',terminal,0,t['edit_id']]]==1 for t in tasks[:terminal]),T1G_full_retention=t1['known_correct']==t1['observations'],full_known_scoring=all(scores[k] is not None for k in keys))
        if arm!='B':
            ls=[]
            for owner in range(1,terminal+1):
                es=sorted([c.read(p) for p in (R/'private/steps'/arm/str(owner)).glob('*.json')],key=lambda e:e['slot']);assert len(es)==20
                source=c.read(R/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json'))
                flags=Counter(z for e in es if not e['accepted'] for z in set([e['rejection_reason']] if e.get('rejection_reason') else [v for a in e['candidates'] for v in a['reasons']]))
                ls.append(dict(source_position=owner,slots=20,accepted=sum(e['accepted'] for e in es),zero_fraction=sum(e['zero_native'] for e in es)/20,rejected_slot_flags=dict(flags),measurements=source,final_path_and_net=c.read(R/'public'/('PATH_'+arm+'_'+str(owner)+'.json')),last_solver={k:es[-1].get(k) for k in ('multipliers','predicted_risks','current_actual_KLs','current_linear_predicted_KLs','projected_KKT_residual')},last_candidate_KLs=[a.get('current_removed_KLs') for a in es[-1]['candidates']]))
            diagnostics[arm]=ls
            m=R/'public'/('MARGIN_DIAGNOSTIC_'+arm+'.json')
            if m.exists():diagnostics[arm+'_margin']=c.read(m)
            matched=[x for x in contrasts if x['comparison'].startswith(arm+'-')]
            gates[arm].update(all_source_CE_below_point1=terminal==3 and all(x['measurements']['final']['weighted_all_FIT_CE']<.1 for x in ls),per_edit_T2G_not_worse=bool(matched) and all(x['status']=='ESTIMATED' and x['negative_edit_bounds']==0 for x in matched))
        for task in ('T0','T2G'):
            pairs=[]
            for t in tasks[:terminal]:
                ids=[t['edit_id']] if task=='T0' else [q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if not ledger['Base_correctness'][q]]
                pairs.extend((lookup[arm,'insertion_R0',t['order'],t['order'],q],lookup[arm,'bank_R0',terminal,0,q]) for q in dict.fromkeys(ids))
            endpoints.append(dict(arm=arm,task=task,comparison='actual-insertion-to-actual-terminal',terminal_prefix=terminal,**transitions(pairs,scores)))
    unknown=sum(sum(v for k,v in x.items() if k in ('MISSING','UNKNOWN_UNATTEMPTED')) for x in coverage.values())
    status='PARTIAL_SCIENTIFIC_STOP' if not gen['full_protocol_complete'] else 'COMPLETE_WITH_UNKNOWN' if unknown else 'COMPLETE'
    result=dict(status=status,scientific_success=not unknown and gen['full_protocol_complete'] and all(all(gates[a].values()) for a in ARMS[1:]),full_protocol_complete=gen['full_protocol_complete'],actual_new_consumer_count=len(rows),historical_B_external_consumers=161,all_consumers=len(files),coverage={a:dict(v) for a,v in coverage.items()},terminals=terminals,panels=panels,contrasts=contrasts,prefix1_secondary=prefix1,deferred_arms={a:'NA_NOT_STARTED_ZERO_GAP_THEORY_PENDING' for a in ('N','NJ') if a not in ACTIVE},protection=protection,endpoints=endpoints,gates=gates,diagnostics=diagnostics,unknown_consumers=unknown,new_Judge=cost['Judge_attempts'],GPU_process_hours=cost['gpu_seconds_used']/3600,actual_counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ('forwards','vision_forwards','backwards','committed_updates','attempt_slots','generation_attempts')},owned_matrix_peak_bytes=cost.get('weights_observed_peak_bytes'),owned_GPU_peak_MiB=cost.get('GPU_memory_observed_peak_MiB'),failures=[dict(kind=p.stem,epoch=c.read(p).get('epoch')) for p in (R/'private/failures').glob('*.json')],old_B_not_rejudged=True,automatic146=False,K_matched_first3='NA_NO_REBUILD',nonlinear_clinical_privacy_certificate=False,calibration_coverage='Eight CT/MRI source groups; no KG/XRay; patient independence unknown')
    if ACTIVE==('J',):
        oldC=Path(c.read(R/'private/FROZEN_PROTOCOL.json')['reference_run'])/'private/steps/C/1'
        es=sorted([c.read(p) for p in oldC.glob('*.json')],key=lambda e:e['slot']);assert len(es)==20
        result['old_C_source1_training_only']=dict(slots=20,accepted=sum(e['accepted'] for e in es),zero_fraction=sum(e['zero_native'] for e in es)/20,all_dual_multipliers_zero=all(all(v==0 for v in e['multipliers'].values()) for e in es),evaluation='NA_NOT_GENERATED_NO_MODEL_REPLAY')
        result['execution_revision']='J_ONLY_DECOUPLED_REVISION1'
        result['completed_scope']='J_ONLY_OR_ITS_ACTUAL_SCIENTIFIC_TERMINAL'
        result['full_original_N_J_NJ_protocol_complete']=False
    c.write(R/'public/RESULTS.json',result)
    text='# '+('/'.join(ACTIVE))+'前三源有限对照\n\n'+status+'。完整实际消费者评分不代表完整三源或科学成功。\n\n|臂|实际末端|任务|正确/分母|未知|微平均界%|\n|---|---:|---|---:|---:|---|\n'
    for x in panels:
        if x['prefix']==(3 if x['arm']=='B' else terminals[x['arm']]['prefix']):text+='|'+ '|'.join(map(str,[x['arm'],x['prefix'],x['task'],str(x['known_correct'])+'/'+str(x['observations']),x['missing'],x['micro_bounds']]))+'|\n'
    text+='\n固定34T2L+2T1L逐题保护、共享缺分精确界、10000次原seed配对edit/source区间及实际插入端点见RESULTS.json。科学止步的prefix1不与prefix3冒充配对。B为历史外部只读161输出，未尝试分不单独请求Judge，不混为永久missing。\n\nJ使当前真实KL函数参与方向求解，最终原生逐视图.01硬门不变。N/NJ若待定不生成也不计永久missing；不将其归一化校准门施加于J。医学块仅首源既有粗框，不是临床反事实；无旧病例重放。有限Fisher/线性化没有全局非线性证书；八源之外、KG/XRay无覆盖。三源不等146/临床/SOTA/隐私证据。接受拒绝、完整CE、间隔失败计数、路径/量化及全部失败成本保留；未知不通过零损伤。原146/M暂停、O/L等待，无自动后续实验。\n'
    (R/'public/REPORT_ZH.md').write_text(text);c.write(marker,dict(status=status,epoch=time.time()));print(status)

if __name__=='__main__':main()
