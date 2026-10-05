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
from protocol import ARMS,BASE_ARMS,PAIRS,PRIMARY,decode


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
    complete=read(RUN/'private/GENERATION_COMPLETE.json');assert not complete['full146'] and complete['H8']==8 and complete['branches']==240 and 'P1' in complete['completed_phases']
    assert read(RUN/'private/CONTROLLER_COMPLETE.json')['status']=='GENERATION_COMPLETE'
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
        if c['method']=='Base':
            row=l['queries'][c['query_id']];b=bases[row['opaque_Base_id']]
            assert c['output_binding']==digest(b)
            full=dict(query_id=row['query_id'],question=row['question'],reference=row['reference'],image_sha256=row['image_sha256'],image_path=b['image_path'],prompt_ids=b['prompt_ids'],attention_mask=b['attention_mask'],runtime=b['runtime'],generation=b['generation'],output=dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids']),judge=judge_identity)
            assert json.loads(ps[c['payload_key']]['binding'])==full
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
    selected=[t for t in tasks if t['edit_id'] in H]
    assert len(selected)==8
    for a in ARMS:
        expected={(a,'single',t['order'],qid) for t in selected for qid in query_ids(t)}
        assert {k for k in raw if k[0]==a and k[1]=='single'}==expected
        for t in selected:
            assert sum(k[0]==a and k[1]=='H_fit' and k[2]==t['order'] for k in raw)==len(H[t['edit_id']])
            td=RUN/'private/edits'/f"e{t['order']:03d}"/a
            tr=read(td/'TRAINING.json');sc=read(td/'STRUCTURE_CHECK.json');structure,condition,slot=decode(a)
            assert tr['status']=='COMPLETE' and tr['steps']==320 and len(tr['curve'])==320 and sc['status']=='PASS'
            assert tr['binding']['structure']==structure and tr['binding']['seed_slot']==slot and tr['binding']['condition']==condition
            assert sc['state_hash']==tr['final_state_hash'] and set(sc['saved_keys'])=={'G1','G2','G3','G4'}
            assert sc['parameters']==read(RUN/'PLAN_CONFIG.json')['structures'][structure]['parameters']
            warm=read(td.parent/f'{structure}_s{slot}_WARMUP/W0_TRAINING.json')
            assert warm['final_state_hash']==tr['binding']['W_init'] and warm['steps']==320
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


def seed_mean(units):
    from fractions import Fraction
    assert len(units)==3 and all(list(u)==list(units[0]) for u in units)
    return {e:{k:v/3 for k,v in combine([u[e] for u in units]).items()} for e in units[0]}


def report():
    from fractions import Fraction
    import torch
    began=time.time();l,tasks,H,complete,resources,lookup,raw,scores,inherited=load();selected=[t for t in tasks if t['edit_id'] in H];groups=components(tasks,H)
    rows=[];panels=[];contrasts=[];interaction=[];units={};eligible_map={}
    for a in ARMS:
        for t in selected:
            for ev in t['events']:
                for qid in ev['all_probe_query_ids']:
                    rows.append(dict(arm=a,task=ev['task'],edit=t['edit_id'],query_id=qid,key=lookup[a,'single',t['order'],qid],original_base=l['Base_correctness'][qid],base_key=lookup['Base','base',0,qid]))
            for k,d in raw.items():
                if k[0]==a and k[1]=='H_fit' and k[2]==t['order']:rows.append(dict(arm=a,task='H_fit',edit=t['edit_id'],query_id=k[-1],key=lookup[k],original_base=False,base_key=None))
    base_missing=any(scores[r['base_key']] is None for r in rows if r['base_key'])
    for mask in ['original']+([] if base_missing else ['new_Base_sensitivity']):
        for task in ('T0','T1G','T2G','T1L','T2L','H_fit'):
            if task=='H_fit' and mask!='original':continue
            for a in ARMS:
                chosen=[r for r in rows if r['arm']==a and r['task']==task]
                eligible=[r for r in chosen if task in ('T0','H_fit') or (r['original_base'] if mask=='original' else bool(scores[r['base_key']]))==task.endswith('L')]
                m,u=metric(eligible,scores);units[a]=u;eligible_map[a]=eligible
                panels.append(dict(mask=mask,task=task,arm=a,seed_slot=decode(a)[2],unique_queries=len({r['query_id'] for r in eligible}),**m))
            for a in BASE_ARMS:
                pooled=[r for k in range(3) for r in eligible_map[a+'_s'+str(k)]];m,auto=metric(pooled,scores)
                u=seed_mean([units[a+'_s'+str(k)] for k in range(3)]);assert u==auto
                units[a]=u;eligible_map[a]=pooled;panels.append(dict(mask=mask,task=task,arm=a,seed_slot='within_edit_mean',unique_queries=len({r['query_id'] for r in pooled}),**m))
            for slot in ('mean',0,1,2):
                for aa,bb in PAIRS:
                    a,b=(aa,bb) if slot=='mean' else (aa+'_s'+str(slot),bb+'_s'+str(slot))
                    ra,rb=eligible_map[a],eligible_map[b]
                    contrasts.append(dict(mask=mask,task=task,comparison=aa+'-'+bb,seed_slot=slot,primary=(aa,bb) in PRIMARY,micro_delta_bounds=micro_delta(ra,rb,scores),**paired(units[a],units[b],scores,groups),**transitions(ra,rb,scores)))
            for structure in ('TT88','TT84','TT48'):
                h,n=structure+'_H1',structure+'_NO_H';ch,cn='TT44_H1','TT44_NO_H'
                lhs={e:combine([units[h][e],units[cn][e]]) for e in units[h]};rhs={e:combine([units[n][e],units[ch][e]]) for e in units[h]}
                N=len(eligible_map[h]);coef=combine([{k:Fraction(v*sg,N) for k,v in Counter(r['key'] for r in eligible_map[a]).items()} for a,sg in [(h,1),(n,-1),(ch,-1),(cn,1)]]) if N else {}
                interaction.append(dict(mask=mask,task=task,formula=f'({h}-{n})-({ch}-{cn})',seed_slot='within_edit_mean',micro_delta_bounds=bounds(coef,scores) if N else None,**paired(lhs,rhs,scores,groups)))
    storage=[];training=[];inventory=[];mechanism=[]
    for t in selected:
        d=RUN/'private/edits'/f"e{t['order']:03d}"
        for a in ARMS:
            p=d/a/'final.pt';state=torch.load(p,map_location='cpu',weights_only=True);tr=read(d/a/'TRAINING.json')
            assert state['state_hash']==tr['final_state_hash'] and set(state['expert'])=={'G1','G2','G3','G4'}
            storage.append(dict(arm=a,parameters=sum(v.numel() for v in state['expert'].values()),bytes=p.stat().st_size))
            training.append(dict(arm=a,seconds=tr['seconds'],tokens=tr['tokens'],accepted=tr['accepted_updates'],guard_extra_forwards=tr['guard_extra_forwards'],peak_allocated=tr['training_peak_allocated_bytes'],peak_reserved=tr['training_peak_reserved_bytes'],TT_residual_seconds=tr['TT_contraction_and_residual_seconds'],timing_activation_shape=tr['timing_training_activation_shape']))
    for a in ARMS:
        trs=[read(RUN/'private/edits'/f"e{t['order']:03d}"/a/'TRAINING.json') for t in selected]
        curves=[x for tr in trs for x in tr['curve']];alphas=Counter(str(x.get('guard',{}).get('alpha','unguarded')) for x in curves)
        fail=Counter(str(i) for x in curves for trial in x.get('guard',{}).get('trials',[]) for i in trial['failed_indices'])
        mechanism.append(dict(arm=a,outer_attempts=len(curves),accepted=sum(tr['accepted_updates'] for tr in trs),alpha_counts=dict(alphas),failed_constraint_index_counts=dict(fail),cumulative_parameter_displacement=sum(tr['curve'][-1]['cumulative_parameter_displacement'] for tr in trs),cumulative_function_displacement=sum(tr['curve'][-1]['cumulative_function_displacement'] for tr in trs),extra_guard_forwards=sum(tr['guard_extra_forwards'] for tr in trs)))
    diagnostics=[]
    def mean(values):return sum(values)/len(values) if values else None
    # Publish aggregate losses/norms, never clinical inputs or per-question scores.
    for a in ARMS:
        ds=[RUN/'private/edits'/f"e{t['order']:03d}"/a for t in selected]
        for step in (0,1,20,80,160,320):
            items=[read(d/'W0_DIAGNOSTIC.json')['diagnostic'] if step==0 else read(d/'TRAINING.json')['curve'][step-1]['post_update_diagnostic'] for d in ds]
            diagnostics.append(dict(arm=a,step=step,N=len(items),terms={term:{field:mean([x['terms'][term][field] for x in items]) for field in ('raw_loss','weighted_loss','raw_gradient_norm','weighted_gradient_norm')} for term in ('native','fit','H','U')},cosines={field:mean([x[field] for x in items if x[field] is not None]) for field in ('H_edit_cosine','H_U_cosine','H_protection_cosine')}))
            if step:
                curves=[read(d/'TRAINING.json')['curve'][step-1] for d in ds]
                diagnostics[-1]['mean_preclip_norm']=mean([c['preclip_total_norm'] for c in curves])
                diagnostics[-1]['mean_clip_scale']=mean([c['clip_scale'] for c in curves])
                diagnostics[-1]['cores']={k:{field:mean([c['cores'][k][field] for c in curves if c['cores'][k][field] is not None]) for field in ('parameter_norm','gradient_norm','actual_update_norm','relative_update_norm')} for k in ('G1','G2','G3','G4')}
                diagnostics[-1]['function_update_norm']=mean([c['function_update_norm'] for c in curves])
    for p in (RUN/'private').rglob('*.pt'):
        if p.is_symlink():continue
        assert p.resolve().is_relative_to(RUN.resolve());inventory.append(dict(path=str(p),bytes=p.stat().st_size,retain=p.name in ('W0.pt','final.pt','ROUTER.pt'),reason='explicit user retention of sharedW0/TTfinal/router' if p.name in ('W0.pt','final.pt','ROUTER.pt') else 'review unused temporary; preserve active resume'))
    write(RUN/'private/RETAINED_WEIGHTS.json',dict(items=inventory,last_consumers_complete=True,cleanup_pending_review=True))
    warmups=list((RUN/'private/edits').glob('e*/*_WARMUP/W0_TRAINING.json'));assert len(warmups)==96
    primary=[c for c in contrasts if c['mask']=='original' and c['task']=='T2G' and c['primary'] and c['seed_slot']=='mean'];assert len(primary)==2
    stalled=[]
    for slot in range(3):
        a='TT88_GUARDED_H_s'+str(slot);m=next(x for x in mechanism if x['arm']==a);hf=next(p for p in panels if p['mask']=='original' and p['task']=='H_fit' and p['arm']==a)
        if m['accepted']==0 and hf['known_correct']==0 and hf['missing_occurrences']==0:stalled.append(a)
    result=dict(status='REPORTED_PUBLICATION_PENDING',full_method='NOT_CONFIRMED',decision='DESCRIPTIVE_PARETO_OR_INCONCLUSIVE_NO_AUTOMATIC_PROMOTION',stalled_arms=stalled,original_cohort_N=146,evaluated_edit_N=8,seeds=3,independent_edit_N=8,branches=240,warmups=96,panels=panels,paired=contrasts,interaction=interaction,storage=storage,training=training,mechanism=mechanism,diagnostics=diagnostics,judge=dict(payloads=len(scores),valid=sum(v is not None for v in scores.values()),permanent_missing=sum(v is None for v in scores.values()),inherited=len(inherited),new_attempts=resources['Judge_attempts'],consumers=len(lookup)),cost=dict(GPU_hours=resources['gpu_seconds_used']/3600,first_clock_wall_seconds=time.time()-read(RUN/'RUN_MANIFEST.json')['starting_epoch'],CPU_report_seconds=time.time()-began,observed_weight_peak_bytes=resources.get('weights_observed_peak_bytes'),device_continuous_peak_unknown=True,warmup_steps=sum(read(p)['steps']+read(p.parent/'native_TRAINING.json')['steps']+80 for p in warmups)),deployment=dict(stored='G1/G2/G3/G4 only',temporary_AB_elements=73728,temporary_AB_FP32_bytes=294912,trainable_AB=0,expert_counts=read(RUN/'PLAN_CONFIG.json')['structures'],router_bytes=sum(p.stat().st_size for p in (RUN/'private/edits').glob('e*/ROUTER.pt')),owned_total_bytes=sum(x['bytes'] for x in inventory),temporary_caches_bytes=sum(x['bytes'] for x in inventory if not x['retain']),continuous_device_peak_unknown=True),limitations=read(RUN/'PLAN_CONFIG.json')['limitations'])
    dest=RUN/'public/final_metrics';dest.mkdir(exist_ok=False);write(dest/'PAIRED_RESULTS.json',result)
    fields=['mask','arm','seed_slot','task','edits','probes','known_correct','missing_occurrences','macro_bounds','micro_bounds']
    with (dest/'RESULTS.csv').open('x',newline='') as f:
        wr=csv.DictWriter(f,fields,extrasaction='ignore');wr.writeheader();wr.writerows(panels)
    text='# TT-only预算与H更新结果\n\n96 TT暖启动，240条续训，8编辑×3seed。每edit先对seed平均后bootstrap；不是24独立病例。全部部署专家仅TT核。\n\n|方式|任务|编辑|seed重复分母|正确|missing|宏界|微界|\n|---|---|---:|---:|---:|---:|---|---|\n'
    for p in panels:
        if p['mask']=='original' and p['seed_slot']=='within_edit_mean':text+=f"|{p['arm']}|{p['task']}|{p['edits']}|{p['probes']}|{p['known_correct']}|{p['missing_occurrences']}|{p['macro_bounds']}|{p['micro_bounds']}|\n"
    text+='\n主对比与CI：'+json.dumps(primary,ensure_ascii=False)+'\n\n保护更新与小学习率的机制表：'+json.dumps(mechanism,ensure_ascii=False)+'\n\nH_fit仅训练诊断。没有独立H_eval/locality支持，完整方法NOT_CONFIRMED。保护与小学习率相近时不能排除更小步长解释；高拒绝且不学习H不是成功。不自动选主线、扩秩或追加146。\n\n'+'\n'.join('- '+v for v in result['limitations'])+'\n'
    (dest/'REPORT_ZH.md').write_text(text)
    write(RUN/'private/REPORT_COMPLETE.json',dict(status='REPORTED_PUBLICATION_PENDING',epoch=time.time(),required_next=['scientific/privacy review','retain TT finals/W0/router; only owned unused temporary cleanup','proxy new PR delivery'],no_automatic_next_experiment=True))
    print('REPORT_COMPLETE_PUBLICATION_PENDING',flush=True)

if __name__=='__main__':report()
