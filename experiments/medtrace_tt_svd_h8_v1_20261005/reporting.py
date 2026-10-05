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
ARMS=('TTKEEP_NO_H','TTKEEP_H1','TTFREE_NO_H','TTFREE_H1','SVDKEEP_NO_H','SVDKEEP_H1','SVDFREE_NO_H','SVDFREE_H1')
PAIRS=tuple((r+'FREE'+h,r+'KEEP'+h) for r in ('TT','SVD') for h in ('_NO_H','_H1'))+tuple(('SVD'+form+h,'TT'+form+h) for form in ('KEEP','FREE') for h in ('_NO_H','_H1'))+tuple((r+'_H1',r+'_NO_H') for r in ('TTKEEP','TTFREE','SVDKEEP','SVDFREE'))

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
    complete=read(RUN/'private/GENERATION_COMPLETE.json');assert not complete['full146'] and complete['H8']==8 and complete['branches']==64 and 'P1' in complete['completed_phases']
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
            tr=read(td/'TRAINING.json');sc=read(td/'STRUCTURE_CHECK.json');init=read(td.parent/'INITIALIZATION.json')
            assert tr['status']=='COMPLETE' and tr['steps']==320 and len(tr['curve'])==320 and sc['status']=='PASS'
            assert tr['binding']['initializer_binding']==init['routes'][a.split('_')[0]]
            assert tr['binding']['initializer_route']==a.split('_')[0] and sc['state_hash']==tr['final_state_hash']
            assert len({x['native_steps'] for x in init['routes'].values()})==1
            assert {k:x['parameters'] for k,x in init['routes'].items()}=={'TTKEEP':3584,'TTFREE':73728,'SVDKEEP':73732,'SVDFREE':73728}
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
                m,u=metric(eligible,scores);units[mask,task,a]=u;eligible_map[mask,task,a]=eligible
                panels.append(dict(mask=mask,task=task,arm=a,unique_queries=len({r['query_id'] for r in eligible}),**m))
            for a,b in PAIRS:
                ra,rb=eligible_map[mask,task,a],eligible_map[mask,task,b]
                contrasts.append(dict(mask=mask,task=task,comparison=a+'-'+b,micro_delta_bounds=micro_delta(ra,rb,scores),**paired(units[mask,task,a],units[mask,task,b],scores,groups),**transitions(ra,rb,scores)))
            us={a:units[mask,task,a] for a in ARMS}
            for family in ('TT','SVD'):
                fh,fn,kh,kn=[family+x for x in ('FREE_H1','FREE_NO_H','KEEP_H1','KEEP_NO_H')]
                lhs={e:combine([us[fh][e],us[kn][e]]) for e in us[fh]};rhs={e:combine([us[fn][e],us[kh][e]]) for e in us[fh]}
                n=len(eligible_map[mask,task,ARMS[0]])
                coeff=combine([{k:Fraction(v*sg,n) for k,v in Counter(r['key'] for r in eligible_map[mask,task,a]).items()} for a,sg in [(fh,1),(fn,-1),(kh,-1),(kn,1)]]) if n else {}
                interaction.append(dict(mask=mask,task=task,formula=f'({fh}-{fn})-({kh}-{kn})',micro_delta_bounds=bounds(coeff,scores) if n else None,**paired(lhs,rhs,scores,groups)))

    storage=[];training=[];inventory=[]
    for t in selected:
        d=RUN/'private/edits'/f"e{t['order']:03d}";init=read(d/'INITIALIZATION.json')
        training.append(dict(edit_index=t['order'],native_steps_per_route=init['routes']['TTKEEP']['native_steps'],A2_steps=80,W0_steps=320,continuation_steps=320,initialization_seconds=init['seconds'],continuations=[dict(arm=a,seconds=read(d/a/'TRAINING.json')['seconds'],tokens=read(d/a/'TRAINING.json')['tokens']) for a in ARMS]))
        for a in ARMS:
            p=d/a/'final.pt';state=torch.load(p,map_location='cpu',weights_only=True)
            assert state['state_hash']==read(d/a/'TRAINING.json')['final_state_hash']
            expected={'TTKEEP':{'G1','G2','G3','G4'},'TTFREE':{'A','B'},'SVDKEEP':{'U','V','s'},'SVDFREE':{'A','B'}}
            assert set(state['expert'])==expected[a.split('_')[0]]
            storage.append(dict(edit_index=t['order'],arm=a,parameters=sum(v.numel() for v in state['expert'].values()),bytes=p.stat().st_size))
    for p in (RUN/'private').rglob('*.pt'):
        if p.is_symlink():continue
        assert p.resolve().is_relative_to(RUN.resolve());inventory.append(dict(path=str(p),bytes=p.stat().st_size))
    write(RUN/'private/RETAINED_WEIGHTS.json',dict(items=inventory,last_consumers_complete=True,cleanup_pending_review=True))
    primary=[c for c in contrasts if c['mask']=='original' and c['task']=='T2G' and c['comparison'] in ('TTFREE_NO_H-TTKEEP_NO_H','SVDFREE_NO_H-SVDKEEP_NO_H')]
    assert len(primary)==2
    decision='DESCRIPTIVE_EXPOSED_H8_ONLY_NO_AUTOMATIC_PROMOTION'
    result=dict(status='REPORTED_PUBLICATION_PENDING',original_cohort_N=146,evaluated_edit_N=8,full146=False,arms=list(ARMS),branches=64,panels=panels,paired=contrasts,interaction=interaction,storage=storage,training=training,decision=decision,judge=dict(payloads=len(scores),valid=sum(v is not None for v in scores.values()),permanent_missing=sum(v is None for v in scores.values()),inherited=len(inherited),new_attempts=resources['Judge_attempts'],consumers=len(lookup)),cost=dict(GPU_hours=resources['gpu_seconds_used']/3600,first_clock_wall_seconds=time.time()-read(RUN/'RUN_MANIFEST.json')['starting_epoch'],CPU_report_seconds=time.time()-began,observed_weight_peak_bytes=resources.get('weights_observed_peak_bytes'),exact_peak_unknown=True),limitations=read(RUN/'PLAN_CONFIG.json')['limitations'],execution_source=read(RUN/'private/GPU_SOURCE_VERSION.json'))
    dest=RUN/'public/final_metrics';dest.mkdir(exist_ok=False);write(dest/'PAIRED_RESULTS.json',result)
    fields=['mask','arm','task','edits','probes','known_correct','missing_occurrences','macro_bounds','micro_bounds']
    with (dest/'RESULTS.csv').open('x',newline='') as f:
        wr=csv.DictWriter(f,fields,extrasaction='ignore');wr.writeheader();wr.writerows(panels)
    text='# TT/SVD结构与FREE配对消融结果\n\n原146登记中的固定H8，64续训，非146全库。TT/SVD独立初始化；同路线KEEP/FREE共享warm函数，匹配更新/样本但不匹配坐标或墙钟。SVD不是比FREE更紧凑的参数化。\n\n|臂|任务|编辑|分母|已知正确|missing|宏界|微界|\n|---|---|---:|---:|---:|---:|---|---|\n'
    for p in panels:
        if p['mask']=='original':text+=f"|{p['arm']}|{p['task']}|{p['edits']}|{p['probes']}|{p['known_correct']}|{p['missing_occurrences']}|{p['macro_bounds']}|{p['micro_bounds']}|\n"
    text+='\n主比较与CI：'+json.dumps(primary,ensure_ascii=False)+'\n\n结论标签：'+decision+'；不自动更换主线或追加实验。Hfit仅训练诊断，H_eval NA，空locality不能证明安全。\n\n'+'\n'.join('- '+v for v in result['limitations'])+'\n'
    (dest/'REPORT_ZH.md').write_text(text)
    write(RUN/'private/REPORT_COMPLETE.json',dict(status='REPORTED_PUBLICATION_PENDING',epoch=time.time(),required_next=['scientific/privacy review','owned last-consumer cleanup','proxy new PR delivery'],no_automatic_next_experiment=True))
    print('REPORT_COMPLETE_PUBLICATION_PENDING',flush=True)

if __name__=='__main__':report()
