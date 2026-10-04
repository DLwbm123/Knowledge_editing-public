"""Evidence-bound six-arm structured closeout; no generation, requests or parameter choice."""
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
ARMS=('CP_NO_H','CP_H1','TK_NO_H','TK_H1','FREE_NO_H','FREE_H1')
PAIRS=(('TK_H1','FREE_H1'),('TK_H1','CP_H1'),('TK_H1','TK_NO_H'),('CP_H1','CP_NO_H'),('FREE_H1','FREE_NO_H'),('TK_NO_H','CP_NO_H'),('CP_H1','FREE_H1'))

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
    complete=read(RUN/'private/GENERATION_COMPLETE.json');assert complete['full146'] and 'P2' in complete['completed_phases']
    assert read(RUN/'private/CONTROLLER_COMPLETE.json')['status']=='GENERATION_COMPLETE'
    assert read(RUN/'private/PARENT_BANK_ADMISSION.json')['status']=='PASS'
    assert read(RUN/'private/judge_common/SCORER_DONE.json')['status']=='COMMON_SCORING_COMPLETE_WITH_MISSING'
    resources=read(RUN/'RESOURCE_LEDGER.json');assert all(x.get('ended_epoch') for x in resources['gpu_sessions'])
    l=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');by={t['edit_id']:t for t in l['tasks']};tasks=[by[e] for e in l['main_T0']];H=read(RUN/'private/H_AVAILABLE.json')
    db=sqlite3.connect('file:'+str(RUN/'private/judge_common/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    ps={r['key']:dict(r) for r in db.execute('SELECT * FROM payload')};scores={k:v['correct'] for k,v in ps.items()}
    assert all(v['status'] in ('FORMAT_VALID','MISSING') for v in ps.values())
    for k,v in ps.items():assert digest(json.loads(v['binding']))==k and (v['correct'] in (0,1) if v['status']=='FORMAT_VALID' else v['correct'] is None)
    attempted=[k for batch in resources.get('Judge_batches',[]) for k in batch['keys']]
    assert len(attempted)==len(set(attempted))==resources['Judge_attempts']
    from qwen_queue import validate_inherited
    inherited_rows={r['key']:dict(r) for r in db.execute('SELECT * FROM inherited')}
    inherited=set(inherited_rows)
    assert set(attempted)|inherited==set(ps) and not set(attempted)&inherited
    for k,item in inherited_rows.items():
        row=json.loads(item['parent_row']);saved=read(item['batch_receipt']);full=json.loads(ps[k]['binding'])
        assert ps[k]['status']==row['status'] and ps[k]['correct']==row['correct'] and ps[k]['record']==row['record'] and ps[k]['batch']=='INHERITED_PARENT_'+row['batch']
        validate_inherited(full,row,saved,saved['parent_attempt'],read(RUN/'private/judge_common/EPOCH_MANIFEST.json')['judge_identity'])
    for batch in resources.get('Judge_batches',[]):
        saved=read(RUN/'private/judge_common/evidence'/(batch['id']+'.json'))
        assert saved['batch']['records']==[json.loads(ps[k]['record']) for k in batch['keys']]
        if batch['status']=='FORMAT_VALID':
            ev=saved['evidence'];judge_lock=read(RUN/'private/judge_common/EPOCH_MANIFEST.json');assert ev['actual_model']==judge_lock['model'] and ev['reasoning_effort']==judge_lock['reasoning_effort'] and ev['snapshot']==judge_lock['snapshot'] and ev['judge_identity']==judge_lock['judge_identity'] and ev['input_binding']==digest(saved['batch']) and ev['exit_code']==0 and not ev.get('errors') and not ev['tool_event_types'] and all(ev['isolation_checks'].values())
            assert [int(x['is_correct']) for x in validate(saved['batch'],saved['response'])]==[scores[k] for k in batch['keys']]
        else:assert batch['status']=='FAILED_NO_RETRY' and all(scores[k] is None for k in batch['keys'])
    lookup={};raw={};phase_cache={}
    bases=read(RUN/"private/legacy_stage17/BINDINGS.json");judge_identity=read(RUN/"private/judge_common/EPOCH_MANIFEST.json")["judge_identity"]
    for c in db.execute('SELECT * FROM consumer'):
        key=(c['method'],c['mode'],c['edit_order'],c['query_id']);assert key not in lookup
        lookup[key]=c['payload_key']
        if c['method']!='Base':
            d=read(c['path']);assert digest(d)==c['output_binding'];raw[key]=d
            assert d['binding']['arm']==c['method']
            row=d['binding']['input'];b=d['binding']['judge_input'] if c['mode']=='H_fit' else bases[row['opaque_Base_id']]
            assert c['mode']=='H_fit' or row==l['queries'][row['query_id']]
            full=dict(query_id=row['query_id'],question=row['question'],reference=row['reference'],image_sha256=row['image_sha256'],image_path=b['image_path'],prompt_ids=b['prompt_ids'],attention_mask=b['attention_mask'],runtime=b['runtime'],generation=b['generation'],output=d['R0'],judge=judge_identity)
            assert json.loads(ps[c['payload_key']]['binding'])==full
            if c['mode']!='H_fit':
                phase=d['binding']['phase'];pk=digest(phase)
                if pk not in phase_cache:
                    matches=[]
                    for path in (RUN/'private/bank_bindings').glob(c['mode']+'_'+f"{c['edit_order']:03d}"+'_*.json'):
                        item=read(path)
                        if item['phase']==phase:matches.append(item)
                    assert len(matches)==1
                    item=matches[0];assert digest(item['weights'])==phase['weight_ancestry']
                    for eid,arms in item['points'].items():
                        for arm,path in arms.items():assert read(Path(path).parent/'TRAINING.json')['final_state_hash']==item['weights'][eid][arm]
                    phase_cache[pk]=item
                eid=d.get('effective_expert')
                if eid:
                    weights=phase_cache[pk]['weights'][eid]
                    assert d['binding']['weight']==weights.get(c['method'],weights.get('SHARED_NO_H'))
    db.close()
    # Each four-consumer unit has one identical normal routing decision. Common
    # Base/nonH8 execution must have identical full raw outputs, not just scores.
    for (arm,mode,order,qid),d in raw.items():
        if arm!=ARMS[0]:continue
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
    rows=[d for (a,m,p,q),d in raw.items() if a==ARMS[0] and m=='sequential' and p==146]
    return dict(status='FROZEN_R0_ONLY_NO_ORACLE',final_unique_queries=len(rows),selected=dict(Counter('H8' if d.get('effective_expert') in H else 'nonH8' if d.get('effective_expert') else 'Base' for d in rows)),nearest_OFF_with_other_own_radius_candidate=sum(not d.get('effective_expert') and bool(d.get('other_own_radius_candidates')) for d in rows),patient_independence='UNKNOWN',no_fallback_deployed=True)


def gate_status(checks):
    if any(v is not None and v[1]<threshold-1e-12 for v,threshold in checks):return 'NOT_SUPPORTED'
    if any(v is None or v[0]<threshold-1e-12 for v,threshold in checks):return 'INCONCLUSIVE'
    return 'SUPPORTED_ON_EXPOSED_PANEL'


def report():
    global BOOTSTRAP_ENABLED
    began=time.time();l,tasks,H,complete,resources,lookup,raw,scores,inherited=load();roles=role_map(l);groups=components(tasks,H)
    allrows=[];panels=[];contrasts=[];interactions=[];metrics={};units={}
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
    for mask,stratum,arm,mode,p,task in list(units):
        if arm!=ARMS[0]:continue
        slots={a:(mask,stratum,a,mode,p,task) for a in ARMS}
        u={a:units[slots[a]] for a in ARMS}
        assert all(list(u[a])==list(u['TK_H1']) for a in ARMS)
        lhs={e:combine([u['TK_H1'][e],u['CP_NO_H'][e]]) for e in u['TK_H1']}
        rhs={e:combine([u['TK_NO_H'][e],u['CP_H1'][e]]) for e in u['TK_H1']}
        BOOTSTRAP_ENABLED=mask=='original' and stratum in ('all','owner_H8') and not mode.startswith('fixed_panel_')
        from fractions import Fraction
        rs={a:metrics[slots[a]] for a in ARMS};n=len(rs['TK_H1'])
        coeff=combine([{k:Fraction(v*sign,n) for k,v in Counter(r['key'] for r in rs[a]).items()} for a,sign in [('TK_H1',1),('TK_NO_H',-1),('CP_H1',-1),('CP_NO_H',1)]]) if n else {}
        interactions.append(dict(mask=mask,stratum=stratum,mode=mode,prefix=p,task=task,formula='(TK_H1-TK_NO_H)-(CP_H1-CP_NO_H)',micro_delta_bounds=bounds(coeff,scores) if n else None,**paired(lhs,rhs,scores,groups)))
    gradients=[]
    for t in tasks:
        if t['edit_id'] not in H:continue
        for a in ARMS:
            d=read(RUN/'private/edits'/f"e{t['order']:03d}"/a/'TRAINING.json')
            gradients.append(dict(edit_index=t['order'],arm=a,seconds=d['seconds'],tokens=d['tokens'],steps=d['steps'],clip_steps=sum(x['clip_applied'] for x in d['curve']),H_nonzero_steps=None if not a.endswith('_H1') else sum(x['H_gradient']>0 for x in d['curve']),diagnostic_steps=[dict(step=x['step'],gradient=x['diagnostic']) for x in d['curve'] if x['diagnostic'] is not None]))
    route=route_diagnosis(tasks,H,raw,lookup,scores,complete['full146'])
    insertion={a:trajectory([(lookup[a,'native',i,t['edit_id']],lookup[a,'sequential',146,t['edit_id']]) for i,t in enumerate(tasks,1)],scores) for a in ARMS} if complete['full146'] else None
    storage=[]
    import torch,importlib.util
    spec=importlib.util.spec_from_file_location('structured_reporting_worker',RUN/'private/tools/worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    for t in tasks:
        if t['edit_id'] not in H:continue
        for a in ARMS:
            d=RUN/'private/edits'/f"e{t['order']:03d}"/a;check=read(d/'STRUCTURE_CHECK.json');assert check['status']=='PASS'
            start=time.perf_counter();saved=torch.load(d/'final.pt',map_location='cpu',weights_only=True);loaded=time.perf_counter()
            free=w.inference_state(saved['expert']);expanded=time.perf_counter()
            storage.append(dict(edit_index=t['order'],arm=a,parameters=sum(v.numel() for v in saved['expert'].values()),parameter_payload_bytes=sum(v.numel()*v.element_size() for v in saved['expert'].values()),serialized_checkpoint_bytes=(d/'final.pt').stat().st_size(),expanded_parameters=sum(v.numel() for v in free.values()),CPU_load_seconds=loaded-start,CPU_expansion_seconds=expanded-loaded,measurement='read-only CPU closeout measurement, not GPU peak or training latency'))
    ratios=[]
    for t in tasks:
        if t['edit_id'] not in H:continue
        rows={x['arm']:x for x in storage if x['edit_index']==t['order']}
        ratios.append(rows['FREE_H1']['parameter_payload_bytes']/rows['TK_H1']['parameter_payload_bytes'])
    def panel(a,mode,stratum,task):
        return next(x for x in panels if (x['mask'],x['arm'],x['mode'],x['prefix'],x['stratum'],x['task'])==('original',a,mode,1 if mode=='single' else 146,stratum,task))
    def contrast(mode,stratum,task):
        return next(x for x in contrasts if (x['mask'],x['comparison'],x['mode'],x['prefix'],x['stratum'],x['task'])==('original','TK_H1-FREE_H1',mode,1 if mode=='single' else 146,stratum,task))
    t0s=panel('TK_H1','single','owner_H8','T0');t0b=panel('TK_H1','sequential','all','T0')
    checks=[(t0s['micro_bounds'] if t0s['probes']==8 else None,1),(t0b['micro_bounds'] if t0b['probes']==146 else None,1),([min(ratios),min(ratios)],10)]
    for mode,stratum in [('single','owner_H8'),('sequential','all')]:
        checks.extend((contrast(mode,stratum,t)['macro_delta_bounds'],-.01) for t in ('T1G','T2G'))
        checks.extend((contrast(mode,stratum,'T2L')[k],0) for k in ('macro_delta_bounds','micro_delta_bounds'))
    criteria=dict(TK_H1_vs_FREE_H1=gate_status(checks),requirements=[dict(interval=v,minimum=t) for v,t in checks],promotion=False)
    retained=[dict(path=str(p),bytes=p.stat().st_size) for p in (RUN/'private/edits').rglob('*.pt') if not p.is_symlink()]
    write(RUN/'private/RETAINED_WEIGHTS.json',dict(items=retained,final_compact_weights_retained=True,cleanup_pending_dependency_review=True,parent_weights_untouched=True))
    result=dict(status='REPORTED_PUBLICATION_PENDING',N=146,completed_phases=complete['completed_phases'],full146=complete['full146'],H8=8,branches=48,H_relations=12,H_eval=None,panels=panels,paired=contrasts,interaction=interactions,storage=storage,route_diagnosis=route,training=gradients,development_reference=criteria,insertion_to_final=insertion,judge=dict(payloads=len(scores),valid=sum(v is not None for v in scores.values()),permanent_missing=sum(v is None for v in scores.values()),inherited_parent_Qwen_payloads=len(inherited),inherited_parent_Qwen_missing=sum(scores[k] is None for k in inherited),new_attempts=resources['Judge_attempts'],model=read(RUN/'private/judge_common/EPOCH_MANIFEST.json')['model'],immutable_snapshot=read(RUN/'private/judge_common/EPOCH_MANIFEST.json')['snapshot'],concurrency=32,records_per_context=1,old_scores_in_main="exact-parent-Qwen-full-key-only"),cost=dict(GPU_hours=resources['gpu_seconds_used']/3600,generated_weights_observed_peak_bytes=resources.get('weights_observed_peak_bytes'),exact_storage_peak_known=False,retained_weight_files=len(retained),retained_weight_bytes=sum(x['bytes'] for x in retained),first_clock_wall_seconds=time.time()-read(RUN/'RUN_MANIFEST.json')['starting_epoch'],CPU_reporting_seconds=time.time()-began),limitations=['Exposed original146/one order, development diagnosis only','Only8 H-supported edits and12 relationships share7 source images','H_fit is training fitting; H_eval NA','Patient independence UNKNOWN; source connectivity sensitivity is not patient isolation','Original shared U/future metadata overlap unchanged','CP/FREE normalization and Tucker no-normalization are different coordinate optimizer recipes, not isolated capacity','Qwen fixed snapshot; single-record context differs from historical Astra; no cross-Judge score mixing','Only8 experts use each new structure,138 shared parent FREE experts remain unchanged','Matched pressure panel unavailable; no mainline promotion','Free expansion is inference-only; actual CP/Tucker factors persist'])
    result['judge_amendment']=read(RUN/'QWEN_JUDGE_AMENDMENT.json')
    result['runtime_reproducibility_amendment']=read(RUN/'RUNTIME_REPRODUCIBILITY_AMENDMENT.json')
    result['execution_source']=read(RUN/'private/GPU_SOURCE_VERSION.json')
    result['limitations'].append('All six arms use frozen deterministic parent runtime; parent historical free arms are not the primary matched controls')
    dest=RUN/'public/final_metrics';dest.mkdir(exist_ok=False);write(dest/'PAIRED_RESULTS.json',result)
    fields=['mask','stratum','arm','mode','prefix','task','edits','probes','unique_queries','known_correct','missing_occurrences','macro_bounds','micro_bounds']
    for name,select in [('H8_PAIRED_RESULTS.csv',lambda x:x['mode']=='H_fit' or x['mode']=='single' and x['stratum']=='owner_H8'),('FULLBANK_PAIRED_RESULTS.csv',lambda x:x['mode']=='sequential')]:
        selected=[m for m in panels if select(m)]
        if not selected:continue
        with (dest/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(selected)
    text='# H8配对实验审阅\n\n主比较固定 TK_H1−FREE_H1、TK_H1−CP_H1、TK_H1−TK_NO_H；另报告预定H交互效应；所有完整分母、配对转移、共享missing差界、编辑bootstrap10000次和来源连通组敏感性见PAIRED_RESULTS.json。\n\n'
    text+='本轮统一使用 Qwen3-32B-AWQ 固定快照、单条独立上下文和最多32并发；不混用历史Astra或测速分数。\n\n'
    text+='完成阶段：'+','.join(complete['completed_phases'])+'。完整146状态：'+str(complete['full146'])+'。独立H_eval：NA。\n\n'
    text+='|臂|任务|编辑|probe|已知正确|missing|宏平均界|微平均界|\n|---|---|---:|---:|---:|---:|---|---|\n'
    for m in panels:
        if m['mask']=='original' and m['stratum']=='all' and (m['mode']=='sequential' and m['prefix']==146 if complete['full146'] else m['mode']=='single'):
            text+=f"|{m['arm']}|{m['task']}|{m['edits']}|{m['probes']}|{m['known_correct']}|{m['missing_occurrences']}|{m['macro_bounds']}|{m['micro_bounds']}|\n"
    text+='\n开发参考：'+json.dumps(criteria,ensure_ascii=False)+'。未达到门槛的结果照留；不自动追加实验。\n\n'+'\n'.join('- '+x for x in result['limitations'])+'\n'
    (dest/'FINAL_REVIEW_ZH.md').write_text(text)
    (dest/'ROUTE_DIAGNOSTICS_ZH.md').write_text('# 路由诊断\n\n'+json.dumps(route,ensure_ascii=False,indent=2)+'\n\nR0未改、无额外Oracle；新结构仅覆盖8个专家。\n')
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text('# 训练诊断\n\n'+json.dumps(gradients,ensure_ascii=False,indent=2)+'\n\nH缺失臂记NA；梯度冲突仅作机制线索；结构坐标和归一化配方差异照报。\n')
    write(dest/'EXECUTION_AUDIT_SUMMARY.json',dict(status='REPORTED_PUBLICATION_PENDING',complete_phase_counts=complete['completed_phases'],GPU_mechanical='PASS',H8=8,branches=48,shared_experts=138 if complete['full146'] else None,judge=result['judge'],cost=result['cost'],execution_source=result['execution_source'],runtime_reproducibility_amendment=result['runtime_reproducibility_amendment'],public_delivery_pending=True))
    write(RUN/'private/REPORT_COMPLETE.json',dict(status='REPORTED_PUBLICATION_PENDING',epoch=time.time(),required_next=['scientific review','privacy scan','proxy push/new draft PR/remote and anonymous verification/attach'],no_automatic_followup_experiment=True))
    print(json.dumps(dict(status='REPORTED_PUBLICATION_PENDING',completed_phases=complete['completed_phases'])),flush=True)

if __name__=='__main__':report()
