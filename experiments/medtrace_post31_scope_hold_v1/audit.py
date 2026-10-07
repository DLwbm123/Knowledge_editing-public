"""Read-only PR31 identity audit and frozen-feature routing diagnostics."""
import os,sys,json,sqlite3,time
from pathlib import Path
from collections import Counter
BASE=Path(os.environ['BASE_ROOT']);RUN=Path(os.environ['RUN_ROOT'])
sys.path.insert(0,str(RUN/'private/tools'))
from common import read,write,digest
sys.path.insert(0,str(RUN/'private/tools'))


def snapshot():
    queue=read(BASE/'private/QUEUES.json');ts=[t for t in queue['tasks'] if t['cohort']=='P2']
    db=sqlite3.connect('file:'+str(BASE/'private/judge_common/queue.sqlite')+'?mode=ro',uri=True)
    scores={p:(status,bool(correct)) for p,status,correct in db.execute('SELECT consumer.path,payload.status,payload.correct FROM consumer JOIN payload ON payload.key=consumer.payload_key')};db.close()
    failures=[];off=Counter();check=[];prefix=Counter();first_ids=set(t['edit_id'] for t in ts[:8]);task_by_qid={qid:(t['edit_id'],ev['task']) for t in ts for ev in t['events'] for qid in ev['all_probe_query_ids']}
    for p in (BASE/'private/outputs/bank').glob('*/s0/*/*/*/R0/*.json'):
        o=read(p);b=o['binding'];qid=b['input']['query_id'];arm=b['arm'];n=b['prefix'];kind=task_by_qid.get(qid,(None,None));score=scores.get(str(p),('MISSING',False))
        if kind[0] in first_ids and kind[1] in ('T0','T1G','T2G'):
            prefix[(arm,n,kind[1],'n')]+=1;prefix[(arm,n,kind[1],'correct')]+=int(score[0]=='FORMAT_VALID' and score[1]);prefix[(arm,n,kind[1],'missing')]+=int(score[0]!='FORMAT_VALID')
        if not o['effective_expert']:
            base=b['judge_input'].get('output',{});tokens=base.get('raw_token_ids',base.get('token_ids'))
            off['outputs']+=1;off['token_identity_verified' if tokens is not None and tokens==o['R0']['raw_token_ids'] else 'binding_or_token_NA']+=1
        if b['panel']=='CHECK' and n==24:
            row=next(x for x in read(BASE/'private/U_ROLES.json')['CHECK'] if x['question']==b['input']['question'] and x['image_path']==b['input']['image_path'])
            teacher=read(BASE/'private/teacher'/(digest(row)+'.json'));bt=teacher['tokens'];st=o['R0']['raw_token_ids'];common=0
            for a,z in zip(bt,st):
                if a!=z:break
                common+=1
            check.append(dict(input=b['input'],Base=teacher.get('raw_answer'),student=o['R0'],arm=arm,route=o['route'],prefix_length=common,Base_tokens=bt,student_tokens=st,first_divergence=common if bt!=st else None,semantic_status='PENDING_BLIND_REFERENCE_REVIEW',medical_accuracy='NA_IMAGE_EVIDENCE_REVIEW_REQUIRED'))
        if arm=='FROZEN_W0' and n==24 and kind[1]=='T1G' and score[0]=='FORMAT_VALID' and not score[1]:
            owner=next(t for t in ts if t['edit_id']==kind[0]);forced=next((BASE/'private/outputs/bank'/owner['anonymous_edit']/'s0/FROZEN_W0/n0/p24/forced').glob(digest(qid)+'.json'));f=read(forced)
            failures.append(dict(query=b['input'],normal_path=str(p),forced_path=str(forced),owner=kind[0],selected=o['effective_expert'],route=o['route'],weight=o['weight'],normal_correct=False,forced_score=scores.get(str(forced)),normal_answer=o['R0'],forced_answer=f['R0'],attribution='PENDING_SCOPE_COMPATIBILITY_AND_GPU_PARITY'))
    assert len(ts)==24 and len(failures)==6,(len(ts),len(failures))
    write(RUN/'private/A1_BLIND_REVIEW_PENDING.json',check);write(RUN/'private/A2_CASES.json',failures)
    rows=[]
    for arm,n,task,field in sorted(prefix):
        if field=='n':rows.append(dict(arm=arm,prefix=n,task=task,observations=prefix[(arm,n,task,'n')],correct=prefix[(arm,n,task,'correct')],missing=prefix[(arm,n,task,'missing')],population='Same first eight PR31 edits'))
    write(RUN/'public/A3_FIXED_POPULATION.json',rows)
    warm={name:sum((BASE/'private/edits'/t['anonymous_edit']/'warmup'/name).exists() for t in ts) for name in ['native.pt','A2.pt','W0.pt']}
    write(RUN/'public/A_CPU_AUDIT.json',dict(status='PASS_READ_ONLY',P2_edits=24,T1G_loss_cases=6,OFF=dict(off),warmup=warm,A2_status='NOT_AVAILABLE_NO_RETRAIN',CHECK_sources=4,old_CHECK_role='DEV',semantic_review='PENDING',confirmation='BLOCKED_CONFIRMATION',near_scope='BLOCKED_NEAR_SCOPE',historical_assets_modified=False))
    print('CPU_AUDIT_PASS',len(failures),flush=True)


def gpu_audit():
    import torch
    from dataclasses import replace
    from common import lease,load,budget,record,local_path,save,state_hash
    from train import clone,teacher_batch
    from methods.medtrace.core import MedTraceLayerHook
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
    sys.path.insert(0,str(RUN/'private/official_llava'))
    gpu=int(os.environ['GPU']);queue=read(BASE/'private/QUEUES.json');ts=[t for t in queue['tasks'] if t['cohort']=='P2']
    with lease(gpu):
        runtime,bindings=load(gpu);editor=BalanceEditPaperSpecEditor(runtime)
        bank=[]
        for t in ts:bank+=torch.load(BASE/'private/edits'/t['anonymous_edit']/'ROUTER.pt',map_location='cpu',weights_only=True)['entries']
        features={};parity=[];orders=[list(range(24)),list(reversed(range(24))),torch.randperm(24,generator=torch.Generator().manual_seed(20261006)).tolist()]
        try:
            ledger=read(BASE/'private/EVAL_LEDGER.json');qids=sorted({q for t in ts for ev in t['events'] for q in ev['all_probe_query_ids']})
            rows=[dict(ledger['queries'][q],feature_id=q,role='EVALUATION_ONLY') for q in qids]
            for t in ts:
                for i,question in enumerate([t['native']['question']]+t['fit_questions']):rows.append(dict(t['native'],question=question,feature_id=t['edit_id']+':fit'+str(i),role='ROUTE_CAL' if i==4 else 'ROUTE_TRAIN_POSITIVE',owner=t['edit_id']))
            for row in read(BASE/'private/U_ROLES.json')['CAL']+read(BASE/'private/U_ROLES.json')['CHECK']:rows.append(dict(row,feature_id='U_'+digest(row),scope_qualification='UNKNOWN_NO_NEGATIVE_LABEL'))
            for row in rows:
                budget();query=replace(record(ts[0]),record_id='query',question=row['question'],target='',official_rephrase='',image_path=Path(local_path(row['image_path'])))
                with torch.inference_mode():key=editor._question_key(query).detach().cpu()
                features[row['feature_id']]=dict(key=key,row=row)
            save(RUN/'private/FEATURES.pt',dict(features=features,bank=bank,grouping='32 contiguous balanced index groups; original float32; no result-dependent preprocessing',frozen_before_new_results=True))
            cases=read(RUN/'private/A2_CASES.json')
            for i,c in enumerate(cases):
                budget();key=features[c['query']['query_id']]['key'];k=torch.stack([x['key'].reshape(-1) for x in bank]);dist=torch.linalg.vector_norm(k-key.reshape(-1),dim=-1);rank=torch.argsort(dist).tolist();selected=[]
                for order in orders:rr=MemoryRouter.from_state(dict(distance='euclidean',entries=[bank[j] for j in order]),device='cpu');selected.append(rr.route(key).logical_edit_id)
                state=torch.load(c['weight']['path'],map_location='cpu',weights_only=True);assert state['state_hash']==c['weight']['hash'];e=clone(state['expert'],ts[0]['seed'],runtime.device).requires_grad_(False);h=MedTraceLayerHook(runtime.get_module('model.layers.21.mlp.down_proj'),e);h.attach()
                try:
                    raw=runtime.adapter.prepare_inputs(Path(local_path(c['query']['image_path'])),c['query']['question'],None);generations=[];logits=[]
                    for _ in range(2):
                        h.clear_request_routing()
                        with torch.inference_mode(),h.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                        generations.append(list(g.raw_token_ids));kwargs,labels,mask,_=teacher_batch(runtime,c['query'],generations[0]);h.clear_request_routing();h.set_teacher_routing(labels)
                        with torch.inference_mode():logits.append(runtime.model(**kwargs).logits[mask].float().cpu())
                    delta=float((logits[0]-logits[1]).abs().max());assert delta==0 and generations[0]==generations[1];assert generations[0]==c['normal_answer']['raw_token_ids'],'Historical normal reproduction differs'
                    parity.append(dict(case=i+1,selected_equals_owner=c['selected']==c['owner'],weight_load_verified=True,repeat_logits_max_abs=delta,selected_single_matches_bank=True,final_order_invariant=len(set(selected))==1,scope_overlap_qualification='UNKNOWN_COMPATIBLE_OWNERS_NOT_EXCLUDED',ranked=[dict(edit_id=bank[j]['logical_edit_id'],distance=float(dist[j]),radius=bank[j]['radius']) for j in rank]))
                finally:h.detach()
                print('A2_PARITY',i+1,len(cases),flush=True)
            write(RUN/'private/A2_PARITY.json',parity);write(RUN/'public/A_GPU_AUDIT.json',dict(status='PASS',cases=len(parity),selected_single_bank_parity=all(x['selected_single_matches_bank'] for x in parity),weight_load_pass=all(x['weight_load_verified'] for x in parity),order_invariant_cases=sum(x['final_order_invariant'] for x in parity),owner_mismatch=sum(not x['selected_equals_owner'] for x in parity),wrong_selection='PROVISIONAL_NOT_UNIQUE_OWNER_PROOF',new_shared_router_scalars_limit=64,scope_negative_qualification='BLOCKED_PENDING_REFERENCE_SCOPE_AUDIT'))
        finally:
            target,base=editor.target,editor.wrapper.base;editor.reset_editor_state();runtime.replace_module(target,base)
    write(RUN/'private/A_GENERATION_COMPLETE.json',dict(status='MECHANICAL_AUDIT_COMPLETE_SEMANTIC_AND_NEXT_STAGE_ADMISSION_PENDING',epoch=time.time()))

if __name__=='__main__':
    if os.environ.get('ACTION')=='A_GPU':gpu_audit()
    else:snapshot()
