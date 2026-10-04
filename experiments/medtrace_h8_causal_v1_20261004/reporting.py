"""Evidence-bound four-arm closeout; no generation, requests or parameter choice."""
from collections import Counter,defaultdict
import csv
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
from audit import read,write,digest,OLD,payload
from legacy_metrics import metric,paired,micro_delta,trajectory,bounds,combine
import legacy_metrics
from scripts.medtrace.stage17_campaign import role_map,query_ids
from scripts.medtrace.astra_judge_bundle import validate
ARMS=('A_NO_H','B_H1','C_H025','D_EXTRA_FIT');PAIRS=(('B_H1','A_NO_H'),('C_H025','A_NO_H'),('C_H025','B_H1'),('D_EXTRA_FIT','A_NO_H'),('B_H1','D_EXTRA_FIT'))

original_bootstrap=legacy_metrics.bootstrap
BOOTSTRAP_ENABLED=True
def scoped_bootstrap(units,scores,groups,repetitions=10000):
    if not BOOTSTRAP_ENABLED:return dict(edit_ci95_envelope=None,source_cluster_ci95_envelope=None,bootstrap_status='Exact point bounds only for sensitivity/route strata/fixed panels; registered main and H8 panels receive10000 draws')
    if units and all(not u for u in units.values()):return dict(edit_ci95_envelope=[0.,0.],source_cluster_ci95_envelope=[0.,0.],source_groups=len({groups[e] for e in units}),bootstrap_status='All10000 draws identically zero')
    return original_bootstrap(units,scores,groups,repetitions)
legacy_metrics.bootstrap=scoped_bootstrap

def components(tasks,H):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def join(a,b):parent[find(a)]=find(b)
    for t in tasks:
        eid=t['edit_id'];join(eid,'source:'+t['native']['source_group']);join(eid,'image:'+t['native']['image_sha256'])
        for h in H.get(eid,[]):join(eid,'source:'+h['source_group'])
    return {t['edit_id']:find(t['edit_id']) for t in tasks}

def load():
    assert read(RUN/'private/CPU_ADMISSION.json')['status']=='PASS'
    assert read(RUN/'private/GPU_MECHANICAL.json')['status']=='PASS'
    complete=read(RUN/'private/GENERATION_COMPLETE.json');assert 'P1' in complete['completed_phases']
    assert read(RUN/'private/judge_common/SCORER_DONE.json')['status']=='COMMON_SCORING_COMPLETE_WITH_MISSING'
    resources=read(RUN/'RESOURCE_LEDGER.json');assert all(x.get('ended_epoch') for x in resources['gpu_sessions'])
    l=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');by={t['edit_id']:t for t in l['tasks']};tasks=[by[e] for e in l['main_T0']];H=read(RUN/'private/H_AVAILABLE.json')
    db=sqlite3.connect('file:'+str(RUN/'private/judge_common/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    ps={r['key']:dict(r) for r in db.execute('SELECT * FROM payload')};scores={k:v['correct'] for k,v in ps.items()}
    assert all(v['status'] in ('FORMAT_VALID','MISSING') for v in ps.values())
    for k,v in ps.items():assert digest(json.loads(v['binding']))==k and (v['correct'] in (0,1) if v['status']=='FORMAT_VALID' else v['correct'] is None)
    attempted=[k for batch in resources.get('Judge_batches',[]) for k in batch['keys']]
    assert len(attempted)==len(set(attempted))==resources['Judge_attempts']
    inherited={k for k,v in ps.items() if v['status']=='MISSING' and str(v['batch']).startswith('INHERITED_PR22_')}
    assert set(attempted)|inherited==set(ps) and not set(attempted)&inherited
    for batch in resources.get('Judge_batches',[]):
        saved=read(RUN/'private/judge_common/evidence'/(batch['id']+'.json'))
        assert saved['batch']['records']==[json.loads(ps[k]['record']) for k in batch['keys']]
        if batch['status']=='FORMAT_VALID':
            ev=saved['evidence'];assert ev['actual_model']=='gpt-6-astra' and ev['reasoning_effort']=='high' and ev['input_binding']==digest(saved['batch']) and ev['exit_code']==0 and not ev.get('errors') and not ev['tool_event_types'] and all(ev['isolation_checks'].values())
            assert [int(x['is_correct']) for x in validate(saved['batch'],saved['response'])]==[scores[k] for k in batch['keys']]
        else:assert batch['status']=='FAILED_NO_RETRY' and all(scores[k] is None for k in batch['keys'])
    lookup={};raw={}
    for c in db.execute('SELECT * FROM consumer'):
        key=(c['method'],c['mode'],c['edit_order'],c['query_id']);assert key not in lookup
        lookup[key]=c['payload_key']
        if c['method']!='Base':
            d=read(c['path']);assert digest(d)==c['output_binding'];raw[key]=d
            assert d['binding']['arm']==c['method']
    db.close()
    # Each four-consumer unit has one identical normal routing decision. Common
    # Base/nonH8 execution must have identical full raw outputs, not just scores.
    for (arm,mode,order,qid),d in raw.items():
        if arm!='A_NO_H':continue
        group=[raw[a,mode,order,qid] for a in ARMS]
        assert all(x.get('route')==d.get('route') for x in group)
        if not d.get('effective_expert') or d['effective_expert'] not in H:assert all(x['R0']==d['R0'] for x in group)
    if complete['full146']:
        for a in ARMS:
            assert sum(k[0]==a and k[1]=='single' for k in raw)==sum(len(query_ids(t)) for t in tasks)
            assert sum(k[0]==a and k[1]=='native' for k in raw)==146
            for p in (1,50,100,146):assert sum(k[0]==a and k[1]=='sequential' and k[2]==p for k in raw)==len({q for t in tasks[:p] for q in query_ids(t)})
    return l,tasks,H,complete,resources,lookup,raw,scores,inherited

def transitions(a,b,scores):
    counts=Counter();unknown=0;improved=defaultdict(int);worsened=defaultdict(int)
    for x,y in zip(a,b):
        assert (x['edit'],x['query_id'])==(y['edit'],y['query_id'])
        sx,sy=scores[x['key']],scores[y['key']]
        if sx is None or sy is None:
            if x['key']==y['key']:counts['unchanged_shared_missing']+=1
            else:unknown+=1
        else:
            counts[f'{sy}_to_{sx}']+=1
            if sx>sy:improved[x['edit']]+=1
            if sx<sy:worsened[x['edit']]+=1
    return dict(known_transitions=dict(counts),unknown_pairs=unknown,known_net_correct=counts['0_to_1']-counts['1_to_0'],known_improved_edits=len(improved),known_worsened_edits=len(worsened))

def route_diagnosis(tasks,H,raw,lookup,scores,full):
    if not full:return dict(status='P2_NOT_COMPLETE',historical_T1G_denominator=76)
    b=read(RUN/'private/legacy_stage17/BINDINGS.json');count=Counter();rows=[]
    priorities=read(RUN/'private/FROZEN_ORACLE_PRIORITY.json')['items'];by={t['edit_id']:t for t in tasks}
    for item in priorities:
        eid,qid=item['edit_id'],item['query_id'];t=by[eid];key=('B_H1','oracle',t['order'],qid)
        if key not in raw:count['oracle_not_executed']+=1;continue
        normal=raw['B_H1','sequential',146,qid];single=raw['B_H1','single',t['order'],qid];oracle=raw[key]
        oldsingle=read(OLD/'private/edits'/f"e{t['order']:03d}"/'single'/(digest(qid)+'.json'))
        oldnormal=read(OLD/'private/sequential/e146/panel'/(digest(qid)+'.json'))
        reproduced=single['R0']==oldsingle['R0'] and normal['R0']==oldnormal['R0']
        nk=lookup['B_H1','sequential',146,qid];ok=lookup[key];ns,oscore=scores[nk],scores[ok]
        category='INSUFFICIENT_MISSING' if ns is None or oscore is None else 'OWN_EXPERT_CAN_ANSWER_R0_REJECTED_OR_OTHER' if oscore==1 and ns==0 and normal['effective_expert']!=eid else 'OWN_EXPERT_CANNOT_ANSWER' if oscore==0 else 'CURRENT_NORMAL_CORRECT_OR_NO_RESCUE'
        count[category]+=1;count['historical_exact_reproduced']+=int(reproduced)
        if reproduced and category=='OWN_EXPERT_CAN_ANSWER_R0_REJECTED_OR_OTHER':count['historical_direct_route_explanation']+=1
        rows.append(dict(edit_id=eid,query_id=qid,reproduced_both_historical_paths=reproduced,category=category,historical_explanation='DIRECTLY_BOUND' if reproduced else 'INCONCLUSIVE_RECONSTRUCTION_DRIFT'))
    write(RUN/'private/ORACLE_ATTRIBUTION.json',dict(rows=rows))
    # The historical-priority table also includes H8 T2G; report the T1G fixed
    # 76 denominator separately by membership in the frozen historical record.
    hist=[json.loads(s) for s in (RUN/'private/HISTORICAL_ATTRIBUTION.jsonl').read_text().splitlines()]
    damaged={(r['edit_id'],r['query_id']) for r in hist if r.get('task')=='T1G' and r.get('mode')=='sequential' and r.get('prefix')==146 and r.get('new_correct')==0}
    # Use the exact original single/final intersection, rather than all final errors.
    hist_t1g=read(RUN/'public/PHASE_A_ATTRIBUTION.json')['T1G']['damage_occurrences']
    direct=sum(r['historical_explanation']=='DIRECTLY_BOUND' and r['category']=='OWN_EXPERT_CAN_ANSWER_R0_REJECTED_OR_OTHER' and (r['edit_id'],r['query_id']) in damaged for r in rows)
    offs=[d for (a,m,p,q),d in raw.items() if a=='A_NO_H' and m=='sequential' and p==146 and not d.get('effective_expert')]
    return dict(status='ORACLE_DIAGNOSTIC_ONLY',all_priority_counts=dict(count),historical_T1G_denominator=hist_t1g,historical_direct_explainable_occurrences=direct,historical_direct_explainable_fraction=direct/hist_t1g if hist_t1g else None,nearest_OFF_unique_queries=len(offs),nearest_OFF_with_other_own_radius_candidate=sum(bool(x.get('other_own_radius_candidates')) for x in offs),patient_independence='UNKNOWN',no_fallback_deployed=True)

def report():
    global BOOTSTRAP_ENABLED
    began=time.time();l,tasks,H,complete,resources,lookup,raw,scores,inherited=load();roles=role_map(l);groups=components(tasks,H)
    allrows=[];panels=[];contrasts=[];metrics={};units={}
    def add_rows(mode,prefix,source_tasks,fixed=None):
        for a in ARMS:
            for t in source_tasks:
                for ev in t['events']:
                    for qid in ev['all_probe_query_ids']:
                        order=t['order'] if mode=='single' else prefix;slot=(a,mode,order,qid)
                        if slot not in lookup:continue
                        d=raw[slot];allrows.append(dict(arm=a,mode=mode if fixed is None else 'fixed_panel_'+str(fixed),prefix=prefix,task=ev['task'],edit=t['edit_id'],query_id=qid,key=lookup[slot],original_base=l['Base_correctness'][qid],active=mode=='sequential' and qid in roles[str(prefix)],owner_H8=t['edit_id'] in H,selected='H8' if d.get('effective_expert') in H else 'Base' if not d.get('effective_expert') else 'nonH8',base_key=lookup['Base','base',0,qid]))
    chosen=tasks if complete['full146'] else [t for t in tasks if t['edit_id'] in H]
    add_rows('single',1,chosen)
    if complete['full146']:
        for p in (1,50,100,146):
            add_rows('sequential',p,tasks[:p])
            for fixed in (1,50,100,146):
                if fixed<=p:add_rows('sequential',p,tasks[:fixed],fixed=fixed)
    # Fit diagnostic retains 12 relationship occurrences, distinct inputs separately.
    for a in ARMS:
        for t in tasks:
            if t['edit_id'] not in H:continue
            entries=sorted((k,d) for k,d in raw.items() if k[0]==a and k[1]=='H_fit' and k[2]==t['order'])
            assert len(entries)==len(H[t['edit_id']])
            for slot,d in entries:
                qid=slot[-1]
                allrows.append(dict(arm=a,mode='H_fit',prefix=1,task='H_fit',edit=t['edit_id'],query_id=qid,key=lookup[slot],original_base=False,active=False,owner_H8=True,selected='H8' if d.get('effective_expert') else 'Base',base_key=None))
    keys=list(dict.fromkeys((r['mode'],r['prefix'],r['task']) for r in allrows))
    base_missing=any(scores[r['base_key']] is None for r in allrows if r['base_key'])
    for mode,p,task in keys:
        for stratum in ('all','owner_H8','owner_nonH8','selected_H8','selected_nonH8','selected_Base'):
            for mask in ('original','new_Base_sensitivity') if task!='H_fit' and not base_missing else ('original',):
                for a in ARMS:
                    rows=[r for r in allrows if (r['arm'],r['mode'],r['prefix'],r['task'])==(a,mode,p,task)]
                    rows=[r for r in rows if stratum=='all' or (r['owner_H8'] if stratum=='owner_H8' else not r['owner_H8'] if stratum=='owner_nonH8' else r['selected']==stratum[9:])]
                    eligible=[r for r in rows if not (task.endswith('L') and r['active']) and (task in ('T0','H_fit') or (r['original_base'] if mask=='original' else bool(scores[r['base_key']]))==task.endswith('L'))]
                    m,u=metric(eligible,scores);m.update(unique_queries=len({r['query_id'] for r in eligible}));slot=(mask,stratum,a,mode,p,task);metrics[slot]=eligible;units[slot]=u
                    panels.append(dict(mask=mask,stratum=stratum,arm=a,mode=mode,prefix=p,task=task,**m))
                for a,b in PAIRS:
                    BOOTSTRAP_ENABLED=mask=='original' and stratum in ('all','owner_H8') and not mode.startswith('fixed_panel_')
                    sa=(mask,stratum,a,mode,p,task);sb=(mask,stratum,b,mode,p,task);ra,rb=metrics[sa],metrics[sb]
                    c=dict(mask=mask,stratum=stratum,comparison=a+'-'+b,mode=mode,prefix=p,task=task,micro_delta_bounds=micro_delta(ra,rb,scores),**paired(units[sa],units[sb],scores,groups),**transitions(ra,rb,scores))
                    if ra:c['net_correct_bounds']=[v*len(ra) for v in c['micro_delta_bounds']]
                    c['guaranteed_net_improved_edits']=sum(bounds(combine([units[sa][e],{k:-v for k,v in units[sb][e].items()}]),scores)[0]>0 for e in units[sa]);contrasts.append(c)
    gradients=[]
    for t in tasks:
        if t['edit_id'] not in H:continue
        for a in ARMS:
            d=read(RUN/'private/edits'/f"e{t['order']:03d}"/a/'TRAINING.json')
            gradients.append(dict(edit_index=t['order'],arm=a,seconds=d['seconds'],tokens=d['tokens'],steps=d['steps'],clip_steps=sum(x['clip_applied'] for x in d['curve']),H_nonzero_steps=None if a not in ('B_H1','C_H025') else sum(x['H_gradient']>0 for x in d['curve']),diagnostic_steps=[dict(step=x['step'],gradient=x['diagnostic']) for x in d['curve'] if x['diagnostic'] is not None]))
    route=route_diagnosis(tasks,H,raw,lookup,scores,complete['full146'])
    insertion={a:trajectory([(lookup[a,'native',i,t['edit_id']],lookup[a,'sequential',146,t['edit_id']]) for i,t in enumerate(tasks,1)],scores) for a in ARMS} if complete['full146'] else None
    criteria={}
    for a in ('B_H1','C_H025','D_EXTRA_FIT'):
        selected={c['task']:c for c in contrasts if c['mask']=='original' and c['stratum']=='all' and c['mode']=='sequential' and c['prefix']==146 and c['comparison']==a+'-A_NO_H'}
        t0=next((m for m in panels if (m['mask'],m['stratum'],m['arm'],m['mode'],m['prefix'],m['task'])==('original','all',a,'sequential',146,'T0')),None)
        if not complete['full146']:criteria[a]='P2_NOT_COMPLETE';continue
        ok=bool(t0 and t0['probes']==146 and t0['micro_bounds'][0]==1 and all(selected[t]['macro_delta_bounds'][0]>=-.01-1e-12 for t in ('T1G','T2G')) and selected['T2L']['macro_delta_bounds'][0]>0 and selected['T2L']['micro_delta_bounds'][0]>=0 and selected['T2L']['net_correct_bounds'][0]>=3-1e-12 and selected['T2L']['guaranteed_net_improved_edits']>=3)
        uncertain=any(c['unknown_pairs'] for c in selected.values())
        criteria[a]='PASS_DEVELOPMENT_REFERENCE_ONLY' if ok else 'INCONCLUSIVE_SHARED_MISSING' if uncertain else 'FAIL_DEVELOPMENT_REFERENCE'
    retained=[];deleted=[]
    for t in tasks:
        if t['edit_id'] in H:continue
        p=RUN/'private/edits'/f"e{t['order']:03d}"/'W_init.pt'
        if p.exists():assert p.resolve().is_relative_to(RUN.resolve()) and not p.is_symlink();deleted.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
    for p in (RUN/'private').rglob('*.pt'):
        if not p.is_symlink():retained.append(dict(path=str(p),bytes=p.stat().st_size))
    write(RUN/'private/RETAINED_WEIGHTS.json',dict(items=retained,owned_temporary_deleted=deleted,final_compact_weights_retained=True,all_consumers_ended=True))
    result=dict(status='REPORTED_PUBLICATION_PENDING',N=146,completed_phases=complete['completed_phases'],full146=complete['full146'],H8=8,branches=32,H_relations=12,H_eval=None,panels=panels,paired=contrasts,route_diagnosis=route,training=gradients,development_reference=criteria,insertion_to_final=insertion,judge=dict(payloads=len(scores),valid=sum(v is not None for v in scores.values()),permanent_missing=sum(v is None for v in scores.values()),inherited_PR22_missing=len(inherited),new_attempts=resources['Judge_attempts'],immutable_snapshot=None,old_scores_in_main=0),cost=dict(GPU_hours=resources['gpu_seconds_used']/3600,generated_weights_observed_peak_bytes=resources.get('weights_observed_peak_bytes'),exact_storage_peak_known=False,retained_weight_files=len(retained),retained_weight_bytes=sum(x['bytes'] for x in retained),first_clock_wall_seconds=time.time()-read(RUN/'RUN_MANIFEST.json')['starting_epoch'],CPU_reporting_seconds=time.time()-began),limitations=['Exposed original146/one order, development diagnosis only','Only8 H-supported edits and12 relationships share7 source images','H_fit is training fitting; H_eval NA','Patient independence UNKNOWN; source connectivity sensitivity is not patient isolation','Original shared U/future metadata overlap unchanged','D extra-fit is not token/FLOP/information matched','No immutable Judge snapshot','Oracle uses ownership labels and is diagnostic only','Historical attribution only after both old paths reproduce'])
    result['runtime_reproducibility_amendment']=read(RUN/'RUNTIME_REPRODUCIBILITY_AMENDMENT.json')
    result['execution_source']=read(RUN/'private/GPU_SOURCE_VERSION.json')
    result['limitations'].append('Common deterministic execution repair differs from the original nondeterministic runtime mode; two prior-mode NO_H branches are retained outside the matched main comparison')
    dest=RUN/'public/final_metrics';dest.mkdir(exist_ok=False);write(dest/'PAIRED_RESULTS.json',result)
    fields=['mask','stratum','arm','mode','prefix','task','edits','probes','unique_queries','known_correct','missing_occurrences','macro_bounds','micro_bounds']
    for name,select in [('H8_PAIRED_RESULTS.csv',lambda x:x['mode']=='H_fit' or x['mode']=='single' and x['stratum']=='owner_H8'),('FULLBANK_PAIRED_RESULTS.csv',lambda x:x['mode']=='sequential')]:
        selected=[m for m in panels if select(m)]
        if not selected:continue
        with (dest/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(selected)
    text='# H8配对实验审阅\n\n主比较固定 B_H1−A_NO_H；所有完整分母、配对转移、共享missing差界、编辑bootstrap10000次和来源连通组敏感性见PAIRED_RESULTS.json。\n\n'
    text+='完成阶段：'+','.join(complete['completed_phases'])+'。完整146状态：'+str(complete['full146'])+'。独立H_eval：NA。\n\n'
    text+='|臂|任务|编辑|probe|已知正确|missing|宏平均界|微平均界|\n|---|---|---:|---:|---:|---:|---|---|\n'
    for m in panels:
        if m['mask']=='original' and m['stratum']=='all' and (m['mode']=='sequential' and m['prefix']==146 if complete['full146'] else m['mode']=='single'):
            text+=f"|{m['arm']}|{m['task']}|{m['edits']}|{m['probes']}|{m['known_correct']}|{m['missing_occurrences']}|{m['macro_bounds']}|{m['micro_bounds']}|\n"
    text+='\n开发参考：'+json.dumps(criteria,ensure_ascii=False)+'。未达到门槛的结果照留；不自动追加实验。\n\n'+'\n'.join('- '+x for x in result['limitations'])+'\n'
    (dest/'FINAL_REVIEW_ZH.md').write_text(text)
    (dest/'ROUTE_DIAGNOSTICS_ZH.md').write_text('# 路由诊断\n\n'+json.dumps(route,ensure_ascii=False,indent=2)+'\n\nOracle不进入主成绩；历史漂移项为INCONCLUSIVE。\n')
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text('# 训练诊断\n\n'+json.dumps(gradients,ensure_ascii=False,indent=2)+'\n\nH缺失臂记NA；梯度冲突仅作机制线索；D不严格等算力。\n')
    write(dest/'EXECUTION_AUDIT_SUMMARY.json',dict(status='REPORTED_PUBLICATION_PENDING',complete_phase_counts=complete['completed_phases'],GPU_mechanical='PASS',H8=8,branches=32,shared_experts=138 if complete['full146'] else None,judge=result['judge'],cost=result['cost'],execution_source=result['execution_source'],runtime_reproducibility_amendment=result['runtime_reproducibility_amendment'],public_delivery_pending=True))
    write(RUN/'private/REPORT_COMPLETE.json',dict(status='REPORTED_PUBLICATION_PENDING',epoch=time.time(),required_next=['scientific review','privacy scan','proxy push/new draft PR/remote and anonymous verification/attach'],no_automatic_followup_experiment=True))
    print(json.dumps(dict(status='REPORTED_PUBLICATION_PENDING',completed_phases=complete['completed_phases'])),flush=True)

if __name__=='__main__':report()
