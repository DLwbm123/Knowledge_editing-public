"""Train-only H pool and paired source/length-matched selection; no student scores."""
import json,sys,random
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_support import conflict,train_row,validate_task
from scripts.medtrace.stage18_score import query_id
from scripts.medtrace.prepare_stage2_sources import normalized
from scripts.medtrace.astra_judge_bundle import write_new

SEED=22002


def select(candidates, edit_id):
    """Candidates contain only train reference metadata, frozen Base distance and token count."""
    assert candidates and len({r['id'] for r in candidates})==len(candidates)
    assert all(r['distance']>=0 and r['radius']>=0 and r['tokens']>0 for r in candidates)
    bucket=lambda r:0 if r['tokens']<=4 else 1 if r['tokens']<=8 else 2 if r['tokens']<=16 else 3
    k=min(4,len({r['source'] for r in candidates}))
    ranked=sorted(candidates,key=lambda r:(r['distance']>r['radius'],r['distance'],r['id']))
    s2=[];sources=set()
    for r in ranked:
        if r['source'] in sources:continue
        s2.append(r);sources.add(r['source'])
        if len(s2)==k:break
    quota=Counter(bucket(r) for r in s2)
    # One exact-question QA per source is required by pool construction.
    assert len({r['source'] for r in candidates})==len(candidates)
    s1=[]
    for b,n in sorted(quota.items()):
        choices=sorted([r for r in candidates if bucket(r)==b],key=lambda r:r['id'])
        random.Random(digest([SEED,edit_id,b])).shuffle(choices);s1.extend(choices[:n])
    assert len(s1)==len(s2)==k and Counter(bucket(r) for r in s1)==quota
    return dict(k=k,S1=[r['id'] for r in s1],S2=[r['id'] for r in s2],
                S2_inside_own_radius=sum(r['distance']<=r['radius'] for r in s2),
                pool_inside_own_radius=sum(r['distance']<=r['radius'] for r in candidates),
                length_bins=dict(quota),S1_tokens=[r['tokens'] for r in s1],S2_tokens=[r['tokens'] for r in s2],
                same_selection={r['id'] for r in s1}=={r['id'] for r in s2})


def task_with_support(original, rows):
    # Unused G is absent in this frozen H-only FASTTRACK branch; no G loss is introduced.
    task=dict(original,H_fit=[train_row(r,'H_fit') for r in rows],G_fit=[])
    return validate_task(task,fasttrack_branch='C_FACT')


def build_selections(runtime,root,cfg,pool,stream,prior,guard):
    """GPU consumer for a future registered 22C worker, never invoked by pool preparation."""
    import torch
    from dataclasses import replace,asdict
    from scripts.medtrace.stage19_fasttrack import record_for
    from scripts.medtrace.stage18_cfact import source_batch,assert_base_off
    from scripts.medtrace.run_selective_write import save
    from m3bench_repro.editors.routing import MemoryRouter,euclidean_distances
    root=Path(root);p=root/'private';pub=root/'public'
    assert digest({k:v for k,v in pool.items() if k!='binding'})==pool['binding']
    assert digest(stream)==pool['stream_binding']
    binding=dict(pool=pool['binding'],runtime=cfg['runtime_lock'],generation=cfg['generation_lock'],gpu_uuid=cfg['gpu_uuid'],code=cfg['code_commit'])
    path=p/'H_COVERAGE_FEATURES.pt'
    cache=torch.load(path,map_location='cpu',weights_only=True) if path.exists() else dict(binding=binding,features={})
    assert cache['binding']==binding
    record=record_for(stream['tasks'][0]);target='model.layers.31.mlp.up_proj'
    assert runtime.target_lock['balancedit']['targets'][0]==target
    for i,(q,row) in enumerate(sorted(pool['candidates'].items()),1):
        guard();assert_base_off(runtime)
        if q in cache['features']:continue
        with torch.inference_mode():
            batch=source_batch(runtime,record,row)
            tokens=list(batch.target_token_ids)
            assert runtime.adapter.tokenizer.eos_token_id in tokens
            query=runtime.build_question_batch(replace(record,question=row['question'],target='',image_path=Path(row['image_path'])))
            key=runtime.extract_layer_input_key(query,module_path=target,pooling='mean').cpu()
        cache['features'][q]=dict(key=key,target_token_ids=tokens)
        if i%10==0:save(path,cache)
        del batch,query
    save(path,cache)
    routers={n:MemoryRouter.from_state(dict(distance='euclidean',entries=[prior['routes'][t['canonical_edit_id']] for t in stream['tasks'][:n]]),device=runtime.device) for n in (19,45)}
    result=[]
    for t in pool['tasks']:
        guard();route=prior['routes'][t['edit_id']];items=[];active={}
        for q in t['candidate_ids']:
            row=pool['candidates'][q];f=cache['features'][q];key=f['key'].to(runtime.device)
            distance=float(euclidean_distances(route['key'].to(runtime.device),key)[0])
            items.append(dict(id=q,source=row['source_group'],distance=distance,radius=float(route['radius']),tokens=len(f['target_token_ids'])))
            active[q]={str(n):asdict(router.route(key)) for n,router in routers.items()}
        chosen=select(items,t['edit_id']);result.append(dict(position=t['position'],edit_id=t['edit_id'],**chosen,candidate_routes=active,candidates=items))
    freeze=dict(binding=binding,rows=result,pool_binding=pool['binding'],student_outputs_used=False)
    freeze['freeze_id']=digest(freeze);write_new(p/'H_SELECTION_FREEZE.json',freeze)
    summary=dict(pool_binding=pool['binding'],selection_binding=freeze['freeze_id'],development_edits=19,
                 different_selections_first19=sum(not r['same_selection'] for r in result[:19]),
                 status='SUPPORTED' if any(not r['same_selection'] for r in result[:19]) else 'UNSUPPORTED_NO_SELECTION_DIFFERENCE',
                 rows=[{k:r[k] for k in ('position','k','S2_inside_own_radius','pool_inside_own_radius','length_bins','S1_tokens','S2_tokens','same_selection')} for r in result],
                 keys='Frozen Base OFF L31 up_proj input mean; no H/eval-based layer or routing changes',
                 actual_activation='Private candidate_routes records both current own-radius coverage and competing naturally selected experts at19/45',
                 student_outputs_used=False)
    write_new(pub/'H_COVERAGE_AUDIT.json',summary)
    return freeze


def freeze(directory):
    d=Path(directory);p=d/'private';read=lambda path:json.loads(path.read_text())
    inv=read(p/'SOURCE_INVENTORY.json');stream=read(p/'run/private/STREAM.json');confirm=read(p/'CONFIRM_CANDIDATE_FREEZE.json')
    availability=read(p/'COVERAGE_IMAGE_AVAILABILITY.json');assert not availability['missing']
    evalrows=stream['core_rows']+stream['new_rows']+stream['positive_rows']
    # Native rewrites intentionally share native images; do not turn them into new evaluation sources.
    evalrows=[r for r in evalrows if r['role']!='native_text_extension']
    protected_sources={r['source_group'] for r in evalrows}|set(confirm['reserved_sources'])
    protected_images={r['image_sha256'] for r in evalrows}|{r['image_sha256'] for r in inv['CONFIRM_candidates']}
    pool=inv['train_H_candidate_pool'];tasks=[];unique={};counts=[]
    for t in stream['tasks']:
        bysource={}
        for r in pool:
            if r['reference_review']!='SUPPORTED' or not conflict(t['native'],r):continue
            if normalized(r['question'])!=normalized(t['native']['question']) or r['image_sha256']==t['native']['image_sha256']:continue
            assert r['source_group'] not in protected_sources and r['image_sha256'] not in protected_images
            assert all(query_id(r)!=query_id(u) for u in t['U_fit'])
            s=r['source_group']
            if s not in bysource or digest(r)<digest(bysource[s]):bysource[s]=r
        rows=sorted(bysource.values(),key=lambda r:query_id(r));assert rows
        task_with_support(t,rows)
        for r in rows:unique[query_id(r)]=r
        tasks.append(dict(position=t['order'],edit_id=t['canonical_edit_id'],original_task_binding=digest(t),candidate_ids=[query_id(r) for r in rows],k=min(4,len(rows))))
        counts.append(dict(position=t['order'],candidate_sources=len(rows),k=min(4,len(rows)),original_H_sources=len({r['source_group'] for r in t['H_fit']})))
    frozen=dict(status='REFERENCE_REUSED_EXACT_QUESTION_STRUCTURAL_RELATION_VALIDATED',tasks=tasks,candidates=unique,
                inventory_binding=digest(inv),stream_binding=digest(stream),selection_seed=SEED,
                reference='Exact QA/image/reference SUPPORTED verdict reuse from frozen source inventory; no new clinical or semantic judgment',
                relation='Same normalized question; different image/source; conflicting unchanged answers under existing reviewed attribute/yes-no contract',
                protected_confirm_binding=confirm['binding'],student_outputs_read=False,source_slot_deduplication='One QA per source by deterministic digest, no score selection')
    frozen['binding']=digest(frozen);write_new(p/'COVERAGE_POOL_FREEZE.json',frozen)
    summary=dict(status='POOL_FROZEN_SELECTION_PENDING_BASE_KEYS',rows=counts,unique_candidate_QA=len(unique),candidate_sources=len({r['source_group'] for r in unique.values()}),
                 development_edits=19,future_regression_edits=45,training_vs_eval_image_overlap=0,training_vs_confirm_source_overlap=0,
                 pool_binding=frozen['binding'],selection='S2 own-radius-first then nearest Base distance; S1 fixed-seed random within same coarse answer-length source quotas; no forced selection differences',
                 H_slots_per_update=1,updates=320,source_quota='At most one QA per selected source; S1/S2 k=min(4,qualified sources)',
                 S0='Original frozen support pool retained; singleton S0 is not duplicated to pretend four independent sources',
                 weights='Freeze only after all six valid Stage22B arms scored; fallback H1/U.01 if no feasible H candidate',
                 token_lengths='Exact runtime target completion including EOS; compute before selection, not using evaluation',
                 GPU_keys='Pending, charge all generation/tokenization/key load time to shared GPU budget',
                 missing_images=0,new_GPU_seconds=0,new_judgments=0,clinical_signoff=False,patient_study='UNKNOWN')
    write_new(d/'H_COVERAGE_PREPARATION.json',summary)
    print('Coverage frozen:',len(unique),'QA;',len(counts),'edits; first19 all k4:',all(x['k']==4 for x in counts[:19]))


def selfcheck():
    rows=[dict(id=str(i),source=str(i),distance=i/10,radius=.25,tokens=3 if i<5 else 9) for i in range(10)]
    a=select(rows,'e');b=select(list(reversed(rows)),'e');assert a==b and a['S2']==['0','1','2','3'] and a['S2_inside_own_radius']==3
    assert len(a['S1'])==4 and all(int(i)<5 for i in a['S1'])
    one=select(rows[:1],'e');assert one['k']==1 and one['same_selection']
    # Coincident selections remain coincident; never resample to force an apparent method difference.
    assert select(rows[:4],'e')['same_selection']
    try:select(rows+[dict(rows[0],id='copy')],'e')
    except AssertionError:pass
    else:raise AssertionError('Duplicate-source pool accepted')
    print('Coverage selection selfcheck passed')


if __name__=='__main__':
    if sys.argv[1:] == ['--selfcheck']:selfcheck()
    else:freeze(sys.argv[1])
