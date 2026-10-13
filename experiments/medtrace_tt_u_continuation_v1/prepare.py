"""Metadata-only finite local pool and deterministic queue freeze before students."""
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def prepare():
    r=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PARENT_RUN']);recent=Path(os.environ['RECENT_RUN']);src=Path(os.environ['TOOLS_SOURCE']);reports=Path(os.environ['REPORT_SOURCE'])
    assert not r.exists(),'Never reset an existing run'
    assert read(old/'private/CONTROLLER_COMPLETE.json')['status']=='GENERATION_COMPLETE'
    assert read(recent/'private/CONTROLLER_COMPLETE.json')['status']=='RESULTS_COMPLETE_SCIENTIFIC_REVIEW_PENDING'
    assert shutil.disk_usage(r.parent.parent).free>10*1024**3
    r.mkdir(parents=True);probe=r/'.write_probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
    for d in ['tools','teacher','tmp','edits','outputs','bank_bindings','work','judge_common']:(r/'private'/d).mkdir(parents=True,exist_ok=True)
    (r/'logs').mkdir();(r/'public').mkdir()
    for d in ['source','official_llava','cpu_gate','derived_inputs']:(r/'private'/d).symlink_to(old/'private'/d,target_is_directory=True)
    for n in ['SOURCE_COMMIT.json']:shutil.copy2(old/'private'/n,r/'private'/n)
    for n in ['structures.py','legacy_queue.py','audit.py','legacy_metrics.py']:shutil.copy2(old/'private/tools'/n,r/'private/tools'/n)
    for p in src.glob('*.py'):shutil.copy2(p,r/'private/tools'/p.name)
    for p in reports.glob('*'):
        if p.suffix in ('.json','.md'):shutil.copy2(p,r/'public'/p.name)
    for n in ['QWEN_JUDGE_AMENDMENT.json','RUNTIME_REPRODUCIBILITY_AMENDMENT.json']:shutil.copy2(old/n,r/n)
    shutil.copy2(old/'private/judge_common/RUNTIME.private.json',r/'private/judge_common/RUNTIME.private.json')
    l=read(old/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');b=read(old/'private/legacy_stage17/BINDINGS.json');join=read(old/'private/legacy_stage17/ROLE_BOUNDARY_JOIN.json');overlay=read(old/'private/legacy_stage17/SOURCE_OVERLAY.json')
    main=[next(t for t in l['tasks'] if t['edit_id']==e) for e in l['main_T0']];assert len(main)==146
    p1=[t for t in main if t['order'] in [31,35,49,71,87,125,126,146]];assert len(p1)==8
    strata=defaultdict(list)
    for t in main:
        if t in p1:continue
        q=t['native']['question'].lower();kind=q.split()[0].strip('?,') if q else 'unknown'
        strata[(t['native']['dataset'],kind)].append(t)
    chosen=[];groups=set()
    while len(chosen)<24:
        changed=False
        for k in sorted(strata):
            while strata[k] and strata[k][0]['native']['source_group'] in groups:strata[k].pop(0)
            if strata[k] and len(chosen)<24:
                t=strata[k].pop(0);chosen.append(t);groups.add(t['native']['source_group']);changed=True
        if not changed:break
    assert len(chosen)==24;p2=sorted(chosen,key=lambda t:t['order'])
    banned=set(map(tuple,read(os.environ['EXCLUSION_FILE'])['banned_images']))
    banned.update(map(tuple,join['evaluation_source_groups']+join['reserved_source_groups']))
    banned.update((q['dataset'],q['source_group'].split(':')[-1]) for q in l['queries'].values())
    banned.update((t['native']['dataset'],t['native']['source_group'].split(':')[-1]) for t in l['tasks'])
    # Prior source role inventory also excludes old H sources without reading their QA.
    train_file=Path(os.environ['DATA_ROOT'])/'knowledge_editing/data/m3bench/SLAKE/train.json';raw=read(train_file)
    image_groups=defaultdict(list)
    for x in raw:
        group=x['img_name'].split('/')[0]
        if x['q_lang']=='en' and ('SLAKE',group) not in banned and x['content_type']=='Organ':image_groups[group].append(x)
    u=[]
    for group,rows in sorted(image_groups.items()):
        x=min(rows,key=lambda x:x['qid']);path=train_file.parent/'imgs'/x['img_name']
        if not path.is_file():continue
        prior=overlay['image_roles'].get('SLAKE:'+l['support_source_provenance']['release']+':'+group)
        if prior is not None:continue
        u.append(dict(dataset='SLAKE',image_id=group,image_path=str(path),source_group='SLAKE:'+group,patient_id='UNKNOWN',question=x['question'],reference=x['answer'],source_qid=x['qid'],source_file=str(train_file),source_role='train',source_release=l['support_source_provenance']['release'],source_annotation=dict(content_type=x['content_type'],location=x['location'],modality=x['modality']),protection_scope='Unedited image-specific anatomical organ membership. Native edit targets apply to their bound inputs; they do not alter organ membership on this separate source. No patient independence claim.',prior_role='NO_ROLE_FOUND_IN_AUDITED_INVENTORIES'))
    # Reserve CAL/CHECK first; use the actual remaining bounded pool for FIT.
    if len(u)>=9:k=min(8,len(u)-8);fit=u[:k];cal=u[k:k+4];check=u[k+4:k+8]
    else:fit=u[:min(8,len(u))];cal=[];check=[]
    for role,rows in [('FIT',fit),('CAL',cal),('CHECK',check)]:
        for x in rows:x['role']=role
    roles=dict(FIT=fit,CAL=cal,CHECK=check);write(r/'private/U_ROLES.json',roles)
    historical=[]
    for t in p1:
        for x in t['U_fit']:historical.append(dict(order=t['order'],source_group=x['source_group'],excluded_by_inventory=(x['dataset'],x['image_id']) in banned,prior_role=overlay['image_roles'].get(x['source_group'])))
    assert all(x['excluded_by_inventory'] for x in historical)
    write(r/'private/U_ROLE_AUDIT.json',dict(historical=historical,roles=roles,exclusions_count=len(banned),source_inventory=os.environ['EXCLUSION_FILE'],patient='UNKNOWN',clinical_signoff=False,role_freeze_epoch=time.time(),student_outputs_read=False,finite_local_search_only=True))
    write(r/'public/U_ROLE_AUDIT.json',dict(status='CLEAN_POOL_FROZEN' if fit else 'BLOCKED_U_SOURCE',FIT_groups=len(fit),CAL_groups=len(cal),CHECK_groups=len(check),distinct_images=len(fit+cal+check),patient='UNKNOWN',source_groups='Image/source IDs only; unavailable patient/exam/paper identity cannot be asserted independent',old_shared_U='EXCLUDED_CONSERVATIVELY: historical role exclusion; old NO_H is diagnostic only',selection='Finite local author-train organ-membership QA; source disjoint from registered queries/future natives and historical exclusion inventory; deterministic IDs, no output selection',pool_target=8,actual_pool=len(fit),shortfall=max(0,8-len(fit))))
    tasks=[]
    for cohort,ts in [('P1',p1),('P2',p2)]:
        for alias,t in enumerate(ts,1):
            events=[e for e in t['events'] if e['task'] in ('T0','T1G','T2G','T1L','T2L')]
            clean={k:t[k] for k in ('edit_id','order','native','fit_questions','seed')};clean.update(cohort=cohort,anonymous_edit=cohort+'_E'+str(alias).zfill(3),events=events)
            tasks.append(clean)
    qs={qid:l['queries'][qid] for t in tasks for e in t['events'] for qid in e['all_probe_query_ids']}
    write(r/'private/QUEUES.json',dict(tasks=tasks,P2_blocks=[[t['edit_id'] for t in p2[i:i+8]] for i in (0,8,16)],rule='Round robin sorted dataset/question-first-token strata; original order within strata, then restore original order; one known native source group; no output access'))
    write(r/'private/EVAL_LEDGER.json',dict(queries=qs,Base_correctness={k:l['Base_correctness'][k] for k in qs},freeze_id=l['freeze_id'],locality_qualification='Original probes retained as historical diagnostic; no independently qualified scope denominator, T1L/T2L primary NA'))
    write(r/'private/EVAL_BINDINGS.json',{k:v for k,v in b.items() if k in {q['opaque_Base_id'] for q in qs.values()}})
    write(r/'private/RUNTIME_BINDING.json',dict(runtime_lock=read(old/'private/cpu_gate/locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json'),generation=next(iter(b.values()))['generation'],source_commit=read(old/'private/SOURCE_COMMIT.json'),reference_commits=dict(PR27='ab601a3ac76ce57a221e20b9586d7ab558ad7c99',PR30='0313b4fdbe030058cdb019ad9f17d2928f862c46')))
    inventory=[]
    import torch
    for t in p1:
        for slot in range(3):
            d=old/'private/edits'/('e'+str(t['order']).zfill(3))/('TT88_s'+str(slot)+'_WARMUP');p=d/'W0.pt';s=torch.load(p,map_location='cpu',weights_only=True)
            assert s['step']==320 and s['binding']['phase']=='W0' and set(s['expert'])=={'G1','G2','G3','G4'} and sum(v.numel() for v in s['expert'].values())==7168
            assert s['binding']['input']==t['native'] and s['binding']['fit']==t['fit_questions'] and s['binding']['seed']==t['seed']+slot*1000003
            assert not any(k in s['binding'] for k in ('H','H_fit','U','U_fit'))
            for phase in ['native','A2','W0']:
                tr=read(d/(phase+'_TRAINING.json'));assert tr['status']=='COMPLETE' and 'H' not in tr['binding'] and 'U' not in tr['binding']
            inventory.append(dict(order=t['order'],slot=slot,path=str(p),bytes=p.stat().st_size,state_hash=s['state_hash'],binding=s['binding']))
        assert (old/'private/edits'/('e'+str(t['order']).zfill(3))/'ROUTER.pt').is_file()
    write(r/'private/ASSET_REUSE_AUDIT.json',dict(W0=inventory,old_final_not_initializers=True,P1_curves='New CE-only and clean-U trajectories; old midpoints unavailable and old U conservatively excluded',P2_W0='No matching assets in fixed TT88 ancestor; uniform native140/A2_80/W0_320 planned',old_results_readonly=True))
    parentplan=read(old/'PLAN_CONFIG.json')
    plan=dict(stage='medtrace_tt_u_continuation_v1',hardware=parentplan['hardware'],P1=dict(edits=8,seeds=3,conditions=['FROZEN_W0','CE_ONLY','CE_U_CLEAN'],steps=320,nodes=[0,40,80,160,320],forced_nodes=[0,80,320]),P2=dict(edits=24,blocks=[8,8,8],seed_slot=0,conditions=['FROZEN_W0','CE_ONLY','CE_U_SINGLE','CE_U_MULTI'],warmup=[140,80,320],bank_prefixes=[8,16,24]),optimizer=dict(input_lr=1e-4,output_lr=1e-3,betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1),loss=dict(native=.5,fit=.5,U=.01,KL='Base||student full vocab token mean, source-balanced round robin'),structure='TT88',parameters=7168,selection=dict(lengths=[40,80,160,320],minimum_CAL_groups=4,T0_drop_pp=0,T1G_T2G_drop_pp=2,CAL_KL_relative_reduction=.2,require_CE_ONLY_improvement=True,require_above_P0_repeat_error=True,near_zero='max(1e-8,10*P0_repeat_error)',fallback=80,prefer_shortest=True),caps=dict(GPU_hours=24,wall_hours=24,Judge=6000,weights_GiB=2,reserve_GiB=8,max_training_GPUs=2),U=dict(FIT=len(fit),CAL=len(cal),CHECK=len(check)),judge_inherit_runs=[str(old),str(recent)]+[parentplan[k] for k in ('historical_structured_run','historical_cp_init_run','historical_tt_svd_run')],parent_run=str(old),recent_run=str(recent))
    write(r/'PLAN_CONFIG.json',plan);write(r/'public/PLAN_CONFIG.json',{k:v for k,v in plan.items() if k not in ('judge_inherit_runs','parent_run','recent_run')})
    write(r/'public/P2_QUEUE.json',dict(frozen_N=24,blocks=[8,8,8],rule=read(r/'private/QUEUES.json')['rule'],status='FROZEN_BEFORE_P1_OUTPUT',edits=[dict(edit=t['anonymous_edit'],source='S'+str(i+1).zfill(3),dataset=t['native']['dataset'],question_type=t['native']['question'].split()[0]) for i,t in enumerate(tasks) if t['cohort']=='P2']))
    env=read(old/'private/LAUNCH_ENV.json');env.update(RUN_ROOT=str(r),DATA_ROOT=os.environ['DATA_ROOT'],M3BENCH_MODEL_PATH=os.environ['M3BENCH_MODEL_PATH'],M3BENCH_VISION_PATH=os.environ['M3BENCH_VISION_PATH'],VLLM_RPC_BASE_PATH=str(Path(os.environ['TMPDIR'])/('ipc.'+str(os.getpid()))))
    Path(env['VLLM_RPC_BASE_PATH']).mkdir(exist_ok=True);write(r/'private/LAUNCH_ENV.json',env)
    # Clock starts at first neutral controller launch, never during source preparation.
    write(r/'RESOURCE_LEDGER.json',dict(gpu_seconds_used=0.,gpu_sessions=[],Judge_attempts=0,Judge_batches=[]))
    write(r/'private/GPU_SOURCE_VERSION.json',dict(commit=os.environ['EXECUTION_COMMIT'],tool_sha256={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in (r/'private/tools').glob('*.py')},reason='Frozen protocol source identity'))
    write(r/'private/CPU_ADMISSION.json',dict(status='PASS',W0_checked=24,roles_frozen=True,P2_frozen=24,training_schema_no_extra_roles=True,source_ready=True))
    print(json.dumps(dict(run=str(r),FIT=len(fit),CAL=len(cal),CHECK=len(check),W0=24,P2=24)))
if __name__=='__main__':prepare()
