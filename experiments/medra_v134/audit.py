"""CPU-only descriptive stage-0 audit. Missing curvature never becomes a proxy F."""
import collections
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def table(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def run(workspace,private):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    workspace,private=Path(workspace),Path(private)
    public=workspace/'reports/medra_v134_20261027';public.mkdir(exist_ok=True)
    sys.path.insert(0,str(workspace/'experiments/medtrace_module_transfer_v1'))
    from module_math import factors
    old=workspace/'.private/med146_single_sequential_v133_20261027'
    recovered=old/'scoring_recovery1'
    read=lambda p:json.loads(p.read_text())
    metadata=read(private/'V133_METADATA.private.json')
    packet=read(recovered/'PACKET_METADATA_B.json');tasks=packet['tasks'];mask=packet['base_mask']
    consumers=read(recovered/'CONSUMERS_B_FROZEN.json')['consumers'];decisions=read(recovered/'MERGED_DECISIONS.json')['decisions']
    records={(v['arm'],v['mode'],v['prefix'],v['owner'],v['query_id']):v for v in consumers}
    routes={(v['mode'],v['prefix'],v['owner'],v['query_id']):v for v in metadata['routes']}
    assert len(records)==22482 and len(routes)==7494
    package=torch.load(old/'backup/MED146.pt',map_location='cpu',weights_only=True)
    assert len(package['states'])==len(package['R0_entries'])==146
    byid={t['edit_id']:t['order'] for t in tasks}
    groups=list(dict.fromkeys(t['native']['source_group'] for t in tasks));groupindex={g:i+1 for i,g in enumerate(groups)}
    training={v['owner']:v for v in metadata['training']}
    expert_rows=[]
    for task in tasks:
        owner=task['order'];state={k:v.double() for k,v in package['states'][task['edit_id']].items()}
        b,a=factors(state);norm2=float(((b.T@b)*(a@a.T).T).sum())
        assert math.isfinite(norm2) and norm2>=0
        t=training[owner]
        expert_rows.append(dict(owner=owner,edit_id=task['edit_id'],source_group=task['native']['source_group'],
            source_group_ordinal=groupindex[task['native']['source_group']],raw_MED_lambda=t['coefficient'],
            raw_MED_operator_norm=math.sqrt(norm2),actual_operator_step_path_length=t['actual_operator_step_sum'],
            protected_penalty_pre_final_step=t['last_logged_penalty_before'],
            penalty_state_step=159,C_source_gain=t['C_gain'],C_FULL_CTX_penalty=t['C_FULL_CTX_penalty'],
            native_NLL_drop_pre_final_step_proxy=t['first_logged_source_NLL'][0]-t['last_logged_source_NLL'][0],
            actual_final_source_gain=None,actual_final_protected_risk=None,q_i=None,
            risk_missing_reason='Required MED low-rank directions were retired; pre-step160 scalar is not final risk',
            mapped_alias_mask_positions=metadata['mapped_alias_mask_positions'],
            all_fallback_mask_positions=None,state_hash=package['state_hashes'][task['edit_id']]))
    table(private/'EXPERT_RISK_LEDGER.private.csv',expert_rows)

    def score(arm,mode,n,owner,qid,field):
        return decisions[records[(arm,mode,n,owner,qid)]['payload_key']][field]
    def roleids(task,role):
        if role=='FIT':return [f"edit_{task['order']}_{i}" for i in range(1,5)]
        ids=list(dict.fromkeys(q for e in task['events'] if e['task']==role for q in e['all_probe_query_ids']))
        return ids if role=='T0' else [q for q in ids if mask[q]==role.endswith('L')]
    fields=('core_correct','full_supported')
    role_rows=[];matrices={};gains=collections.defaultdict(collections.Counter);damage=[];T2L=[]
    transitions=[]
    for mode,n in [('single',1)]+[('bank',n) for n in (1,50,100,146)]:
        phase=f'{mode}_{n}';matrices[phase]={}
        active={(t['native']['image_sha256'],t['native']['question']) for t in (tasks if mode=='bank' else [])[:n]}
        for task in tasks if mode=='single' else tasks[:n]:
            owner=task['order'];actual_owner=owner if mode=='single' else 0
            event_active={(task['native']['image_sha256'],task['native']['question'])} if mode=='single' else active
            for role in ('T0','FIT','T1G','T2G','T1L','T2L'):
                qids=roleids(task,role)
                qids=[q for q in qids if not(mode=='bank' and role.endswith('L') and records[('MED',mode,n,0,q)]['active_target'])]
                for field in fields:
                    matrix=matrices[phase].setdefault(role+'_'+field,collections.Counter())
                    med=[];legacy=[];effective=[];fallback=0
                    for qid in qids:
                        m=score('MED',mode,n,actual_owner,qid,field);l=score('LEGACY',mode,n,actual_owner,qid,field)
                        tag='both_correct' if m and l else 'MED_only' if m else 'M13P_only' if l else 'both_incorrect'
                        matrix[tag]+=1;med.append(m);legacy.append(l)
                        route=routes[(mode,n,actual_owner,qid)];effective.append(byid.get(route['MED_expert']))
                        fallback+=route['MED_expert'] is None
                        if role=='T2G':gains[(phase,field,groupindex[task['native']['source_group']])][tag]+=1
                        if role=='T2L' and field=='core_correct' and l and not m:
                            T2L.append(dict(phase=phase,owner=owner,query_id=qid,effective_owner=byid.get(route['MED_expert'])))
                    role_rows.append(dict(phase=phase,owner=owner,edit_id=task['edit_id'],
                        source_group=task['native']['source_group'],role=role,criterion=field,n=len(qids),
                        MED_correct=sum(med),M13P_correct=sum(legacy),effective_owners=effective,Base_fallbacks=fallback))
            qids={q for e in task['events'] for q in e['all_probe_query_ids']}
            for qid in qids:
                v=records[('MED',mode,n,actual_owner,qid)]
                if (v['image_sha256'],v['question']) in event_active:continue
                b,l,m=(score(arm,mode,n,actual_owner,qid,'core_correct') for arm in ('BASE','LEGACY','MED'))
                if b and l and not m:
                    r=routes[(mode,n,actual_owner,qid)]
                    damage.append(dict(phase=phase,owner=owner,query_id=qid,effective_owner=byid.get(r['MED_expert']),
                        MED_route=r['MED_route'],M13P_route=r['M13P_route'],MED_state_hash=r['MED_state_hash']))
        if mode=='bank':
            for task in tasks[:n]:
                for role in ('T1G','T2G'):
                    for qid in roleids(task,role):
                        left=routes[('single',1,task['order'],qid)];right=routes[('bank',n,0,qid)]
                        row=dict(prefix=n,owner=task['order'],role=role,query_id=qid,
                            MED_expert_changed=left['MED_expert']!=right['MED_expert'],
                            M13P_expert_changed=left['M13P_expert']!=right['M13P_expert'])
                        for arm in ('MED','LEGACY'):
                            a=score(arm,'single',1,task['order'],qid,'core_correct');b=score(arm,'bank',n,0,qid,'core_correct')
                            row[arm+'_lost']=a and not b;row[arm+'_repaired']=not a and b
                        transitions.append(row)
    table(private/'EDIT_PHASE_ROLE_LEDGER.private.csv',role_rows)
    table(private/'ROUTE_STATE_LEDGER.private.csv',metadata['routes'])
    save(private/'ITEM_DAMAGE_AUDIT.private.json',dict(fresh_CORE_damage=damage,T2L_CORE_regressions=T2L))
    save(private/'SINGLE_TO_BANK_TRANSITIONS.private.json',transitions)
    route_mismatch=[];max_distance_error=0.
    for r in routes.values():
        a,b=r['MED_route'],r['M13P_route']
        if r['MED_expert']!=r['M13P_expert'] or any(a[k]!=b[k] for k in ('logical_edit_id','nearest_logical_edit_id','activated','distance','radius')):
            route_mismatch.append(r)
        if a.get('nearest_distance') is not None and b.get('nearest_distance') is not None:
            max_distance_error=max(max_distance_error,abs(a['nearest_distance']-b['nearest_distance']))
    decompositions=[]
    for n in (1,50,100,146):
        for role in ('T1G','T2G'):
            rows=[r for r in transitions if r['prefix']==n and r['role']==role]
            medlost={(r['owner'],r['query_id']) for r in rows if r['MED_lost']}
            leglost={(r['owner'],r['query_id']) for r in rows if r['LEGACY_lost']}
            decompositions.append(dict(prefix=n,role=role,n=len(rows),MED_expert_changes=sum(r['MED_expert_changed'] for r in rows),
                M13P_expert_changes=sum(r['M13P_expert_changed'] for r in rows),MED_lost=len(medlost),M13P_lost=len(leglost),
                shared_lost=len(medlost&leglost),MED_repaired=sum(r['MED_repaired'] for r in rows),
                M13P_repaired=sum(r['LEGACY_repaired'] for r in rows)))
    source_rows=[dict(phase=p,criterion=f,source_group_ordinal=g,**c) for (p,f,g),c in gains.items()]
    save(public/'SOURCE_GROUP_DIFFERENCES.json',source_rows)
    save(public/'DIFFERENCE_MATRICES.json',matrices)
    sanitized=[{k:v for k,v in row.items() if k not in ('edit_id','source_group','state_hash')} for row in expert_rows]
    table(public/'EXPERT_RISK_LEDGER.csv',sanitized)
    summary=dict(status='INCOMPLETE',stage0_block='MISSING_REQUIRED_CURVATURE',method_implemented=False,GPU_experiments_started=0,
        baseline_experts_retained=146,MED_routes_checked=7494,M13P_routes_checked=7494,
        semantic_route_mismatches=len(route_mismatch),max_nearest_distance_difference=max_distance_error,
        raw_MED_lambda_range=[min(r['raw_MED_lambda'] for r in expert_rows),max(r['raw_MED_lambda'] for r in expert_rows)],
        raw_MED_operator_norm_range=[min(r['raw_MED_operator_norm'] for r in expert_rows),max(r['raw_MED_operator_norm'] for r in expert_rows)],
        curvature_files_present=metadata['cache_exists'],mask_check=metadata['mask_check'],
        mapped_alias_paths=metadata['mapped_alias_paths'],mapped_alias_mask_positions=metadata['mapped_alias_mask_positions'],
        saved_teacher_masks_checked=metadata['saved_teacher_masks_checked'],
        q_i_unavailable=146,source_final_gain_not_persisted=True,
        fresh_CORE_damage=[{k:v for k,v in row.items() if k in ('phase','owner','effective_owner')} for row in damage],
        T2L_CORE_regressions=[{k:v for k,v in row.items() if k in ('phase','owner','effective_owner')} for row in T2L],
        single_to_bank_decomposition=decompositions,new_model_calls=0,new_training=0,new_generation=0,new_Judge=0,
        V133_models_outputs_scores_ledgers_reports_unchanged=True,monitoring='OFF')
    save(public/'STAGE0_AUDIT.json',summary)
    return summary
