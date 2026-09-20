"""R1 CPU event compiler. No outputs from prospective arms are read."""
import collections
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.medtrace.stage17_prepare import digest, PROTOCOL, PROMPT
from scripts.medtrace.stage18_score import query_id, score_key
from preflight import identity, self_check

HERE = Path(__file__).parent
COMMON = ROOT/'reports/medtrace_stage22_20260919/private/run'
D = ROOT/'reports/medtrace_stage24d_20260920/private/run/private'
C = ROOT/'reports/medtrace_stage24c_20260919/private/run/private'

def read(p): return json.loads(Path(p).read_text())
def write(name, data): (HERE/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
def lines(name, rows): (HERE/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
def length_bin(n): return 0 if n<=4 else 1 if n<=8 else 2 if n<=16 else 3

def match(aligned, pool, rowmap, lengths, seed):
    """Uniform proposals; fixed finite rejection bound, valid aligned fallback."""
    target = collections.Counter(length_bin(lengths[q]) for q in aligned)
    sources = len({rowmap[q]['source_group'] for q in aligned})
    rng = random.Random(seed)
    for _ in range(4096):
        choice = sorted(rng.sample(sorted(pool), len(aligned)))
        if collections.Counter(length_bin(lengths[q]) for q in choice)==target and len({rowmap[q]['source_group'] for q in choice})==sources:
            return choice, 'seeded_uniform_rejection'
    # No deliberate exclusion of aligned examples; fallback cannot create contrast.
    return sorted(aligned), 'legal_aligned_fallback_after_4096_proposals'

def compile_plan():
    self_check()
    stream=read(COMMON/'private/STREAM.json'); tasks=stream['tasks'][:19]
    remote=read(HERE/'private/CPU_REMOTE_AUDIT.json')
    measured=read(ROOT/'reports/medtrace_stage24c_20260919/RESOURCE_PROFILE.json')['writers']
    H_lengths={}
    for task, writer in zip(tasks, measured):
        assert len(task['H_fit'])==1 and writer['position']==task['order']
        assert writer['tokens']['extra']%320==0
        q=query_id(task['H_fit'][0]); n=writer['tokens']['extra']//320
        assert q not in H_lengths or H_lengths[q]==n
        H_lengths[q]=n
    hist=read(D/'OWNERSHIP.json')['history']
    route={(r['prefix'],r['query_id']):r['winner'] for r in hist if r['N']==19}
    native_route={(r['prefix'],r['native']):r['winner'] for r in remote['routes']}
    assert len(native_route)==190
    refs=read(ROOT/'reports/medtrace_stage22_20260919/private/SOURCE_INVENTORY.json')['train_H_candidate_pool']
    verified={(r['image_sha256'],r['question'],r['reference']):r['reference_review'] for r in refs}
    scores=read(COMMON/'private/QUALIFIED_SCORE_CACHE.json');assert scores['protocol']==PROTOCOL
    base={r['query_id']:r for r in read(D/'BASE.json')['records']}
    teachers={query_id(r['binding']['source']):r for r in remote['teachers']}
    arrivals={};rowmap={};events=[];slots=[];pending={};last={};reference_events=[];mask_by_t={}
    for task in tasks:
        t=task['order']
        for role in ('native','H_fit','U_fit'):
            for row in ([task['native']] if role=='native' else task[role]):
                assert verified.get((row['image_sha256'],row['question'],row['reference']))=='SUPPORTED'
                q=query_id(row)
                if q in rowmap: assert rowmap[q]['reference']==row['reference'], 'Unresolved reference version conflict'
                rowmap[q]=row
                reference_events.append(dict(t=t,q=q,role=role,reference_binding=digest(row['reference']),provenance='Frozen Stage22 SOURCE_INVENTORY SUPPORTED; no future access',input_fingerprint=identity(row)))
                if role!='native': arrivals.setdefault((role,q),t)
        masked={q for role,q in arrivals if role=='U_fit' and scores['scores'][score_key(base[q]['source'],base[q]['output'])] is False}
        mask_by_t[str(t)]=sorted(masked)
        available={(role,q) for role,q in arrivals if role!='U_fit' or q not in masked}
        for k in list(pending):
            if k not in available: del pending[k]
        for role,q in sorted(available):
            w=route[t,q];old=last.get((role,q))
            if old is None or old!=w:
                events.append(dict(t=t,role=role,q=q,old_winner=old,winner=w,kind='arrival' if old is None else 'winner_change'))
                if w: pending[role,q]=dict(t=min(t,pending.get((role,q),{}).get('t',t)),winner=w)
                else: pending.pop((role,q),None)
            last[role,q]=w
        selected=[]
        if any(v['winner']==t for v in pending.values()): selected.append(t)
        old=[(v['t'],identity(rowmap[q]),v['winner']) for (role,q),v in pending.items() if 0<v['winner']<t]
        if old: selected.append(min(old)[2])
        for j in selected:
            slot=dict(t=t,expert=j,steps=20,kind='current' if j==t else 'old',roles={})
            for role in ('H_fit','U_fit'):
                own=[q for r,q in available if r==role and route[t,q]==j]
                own.sort(key=lambda q:(0 if (role,q) in pending else 1,pending.get((role,q),{}).get('t',t),identity(rowmap[q])))
                own=own[:20]
                if not own:
                    slot['roles'][role]=dict(active=False,reason='NO_ASSIGNED_ROLE');continue
                pool=[q for r,q in available if r==role]
                lengths=H_lengths if role=='H_fit' else {q:len(v['binding']['teacher_tokens']) for q,v in teachers.items()}
                seed=int(digest([tasks[j-1]['seed'],t,j,role,'24E-R1-replay'])[:16],16)
                random_q,method=match(own,pool,rowmap,lengths,seed)
                assert set(random_q)<=set(pool)
                slot['roles'][role]=dict(active=True,B2=[own[k%len(own)] for k in range(20)],B1=[random_q[k%len(random_q)] for k in range(20)],distinct_QA=len(own),sources=len({rowmap[q]['source_group'] for q in own}),bins=dict(collections.Counter(length_bin(lengths[q]) for q in own)),method=method,seed=seed,contrast=set(own)!=set(random_q),tokens_B2=sum(lengths[own[k%len(own)]] for k in range(20)),tokens_B1=sum(lengths[random_q[k%len(random_q)]] for k in range(20)))
                for q in own: pending.pop((role,q),None)
            slots.append(slot)
    # Causal schedule is common and independent of actual sampled learning success.
    for s in slots:
        for role,d in s['roles'].items():
            if d['active']:
                assert all(arrivals[role,q]<=s['t'] for a in ('B1','B2') for q in d[a])
                assert all(route[s['t'],q]==s['expert'] for q in d['B2'])
    contrast=[(s,d) for s in slots for d in s['roles'].values() if d.get('contrast')]
    contrast_times={s['t'] for s,d in contrast}
    contrast_sources={rowmap[q]['source_group'] for s,d in contrast for q in set(d['B1'])^set(d['B2'])}
    # Full native consumers and symbolic selected-parameter versions.
    deps=[]
    for arm in ('B0','B1','B2'):
        versions={}
        for t in range(1,20):
            checks=[dict(expert=t,kind='insert')]+([s for s in slots if s['t']==t] if arm!='B0' else [])
            for serial,event in enumerate(checks):
                j=event['expert'];versions[j]=f'{arm}:t{t}:write{serial}:e{j}'
                for i in range(1,t+1):
                    w=native_route[t,i]
                    dep=dict(arm=arm,t=t,event=serial,native=i,winner=w,symbolic_version=versions[w] if w else 'BASE',input=identity(tasks[i-1]['native']),reference=digest(tasks[i-1]['native']['reference']),fixed_context='EXEC_DEPENDENCIES_R1')
                    dep['state']=digest([dep['input'],dep['reference'],arm,dep['symbolic_version'],w,dep['fixed_context']]);deps.append(dep)
    count=len({r['state'] for r in deps})
    byarm={a:len({r['state'] for r in deps if r['arm']==a}) for a in ('B0','B1','B2')}
    lines('EVAL_DEPENDENCY_MANIFEST.jsonl',deps)
    lines('private/ONLINE_EVENT_LOG.jsonl',events);lines('private/MATCHED_REPLAY_MANIFEST.jsonl',slots)
    write('private/REFERENCE_ARRIVAL_LEDGER.json',reference_events)
    write('private/U_MASK_BY_PREFIX.json',mask_by_t)
    write('ONLINE_SCHEDULE_SUMMARY.json',dict(slots=len(slots),current=sum(s['kind']=='current' for s in slots),old=sum(s['kind']=='old' for s in slots),maintenance_steps_per_replay_arm=20*len(slots),remaining_pending=len(pending),contrast_slots=len(contrast),contrast_times=len(contrast_times),contrast_sources=len(contrast_sources),contrast_pass=len(contrast_times)>=2 and len(contrast_sources)>=2,token_bins=[4,8,16,'above16'],sampling='seeded uniform subsets with fixed rejection limit; matching source count, uniqueQA and length-bin counts; fallback to aligned subset does not create contrast',mask_counts_by_prefix={t:len(v) for t,v in mask_by_t.items()},U_policy='All original U references already SUPPORTED; their own arrival makes reference available. Base-wrong U is masked at that arrival, not only when an H role later appears.',masked_U_final=len(mask_by_t['19'])))
    # Known paraphrases include four fit templates and explicit anchor evaluation lineage.
    alltrain=[r for t in tasks for r in [t['native']]+[dict(t['native'],question=q) for q in t['fit_questions']]+t['H_fit']+t['U_fit']]
    trained={identity(r) for r in alltrain};images={r['image_sha256'] for r in alltrain}
    memberships=read(C/'PANEL_MEMBERSHIPS.json');panel=read(C/'DEV_UNION.json')['rows']
    lineage=set(memberships['DEV_rewrite'])|set(memberships['old_rewrite'])
    audit=[]
    for row in panel:
        q=query_id(row);same=identity(row) in trained;known=q in lineage
        audit.append(dict(q=q,input=identity(row),source=row['source_group'],trained_exact=same,known_native_rephrase=known,shared_training_image=row['image_sha256'] in images,untrained_input=not same and not known))
    write('private/EVAL_SUBPANEL_FREEZE.json',dict(rows=audit,definition='Exclude exact trained inputs and known native paraphrase lineage. Does not assert no unknown paraphrases, pretraining exposure or patient independence.',binding=digest(audit)))
    lookup={r['q']:r for r in audit}
    summary={name:dict(total=len(set(qs)),untrained=len([q for q in set(qs) if lookup[q]['untrained_input']]),sources=len({lookup[q]['source'] for q in qs if lookup[q]['untrained_input']})) for name,qs in memberships.items()}
    write('EVAL_SUBPANEL_SUMMARY.json',dict(panels=summary,original79_preserved=True,known_rewrite_excluded_from_untrained=True,fit_and_explicit_anchor_lineage=True,no_patient_independence=True))
    # Base/cache completeness, not prospective output cache hits.
    coutputs=[json.loads(l) for l in (C/'OUTPUTS.jsonl').read_text().splitlines()]
    bases={query_id(r['source']):dict(source=r['source'],output=r['Base']) for r in coutputs};bases.update(base)
    needed={query_id(r) for r in panel+alltrain if r.get('role')!='native' or r['question'] in [t['native']['question'] for t in tasks]}
    # Fit paraphrases have no independent evaluation/teacher task; do not add invented judges.
    needed={query_id(r) for r in panel}|{q for role,q in arrivals}
    ca=[dict(q=q,key=score_key(bases[q]['source'],bases[q]['output']),completed=type(scores['scores'].get(score_key(bases[q]['source'],bases[q]['output']))) is bool) for q in sorted(needed)]
    missing=sum(not r['completed'] for r in ca)
    write('private/SCORING_CACHE_KEYS.json',ca)
    write('SCORING_CACHE_AUDIT.json',dict(protocol=PROTOCOL,prompt_binding=digest(PROMPT),model='gpt-6-astra',reasoning='high',Base_required=len(ca),Base_completed=len(ca)-missing,Base_missing=missing,teacher_count=len({q for role,q in arrivals if role=='U_fit'}),teacher_tokens_immutable=True,teacher_reference_consistency='Existing Stage24D exact teacher-to-Base replay plus frozen reference judgments',judge_payload='question, gold_answer, actual decoded candidate; source-agreement protocol does not visually inspect image. Image identity remains conservative cache namespace.',legacy_key='image hash/question/reference/actual raw_answer',R1_key='legacy key plus frozen prompt/model/protocol/normalizer/source-reference task namespace; teacher-consistency task separately reserved',prospective_hit_rate_assumed=False))
    # Reserve separate teacher consistency rather than reinterpret gold correctness.
    extras=dict(nonnative_endpoints=360,support_correctness=96,teacher_consistency=51,Base_cache_missing=missing,acceptance=20,failed_attempt_and_closeout=40)
    total=count+sum(extras.values())
    write('JUDGE_BUDGET_CERTIFICATE.json',dict(status='CPU_CONDITIONAL_BOUND_NOT_RUNTIME_ACCEPTANCE',native_state_count=count,native_by_arm=byarm,native_generation_consumers=len(deps),additional_reservations=extras,worst_case_new_items=total,cap=800,headroom=800-total,shared_remaining=2000-188,arithmetic_pass=total<=800,native_self_routing=all(r['native']==r['winner'] for r in remote['routes']),CPU_route_cases=len(remote['routes']),single_selected_hook_code='methods/medtrace/hsic.py:bound_hook + stage15.generate, no bank argument in forward',runtime_invariance='PENDING finite all-path acceptance; certificate cannot authorize GPU until all CPU resource gates pass',future_arm_parameter_equality_assumed=False,historical_hit_rate_assumed=False,all_native_checks_regenerated=True))
    # Compact immutable versions, separate active optimizer/RNG state; no dense banks.
    nversions=57+2*len(slots);param_bytes=4*4*(14336+4096)
    storage=dict(version_count=nversions,parameter_bytes_each=param_bytes,immutable_versions=nversions*(param_bytes+16384),active_resume_and_atomic_copies=3*4*(3*param_bytes+65536),outputs_and_score_indices=128*1024**2,code_and_temporary_files=32*1024**2,shared_model_W0_teacher_copies=0)
    peak=sum(storage[k] for k in ('immutable_versions','active_resume_and_atomic_copies','outputs_and_score_indices','code_and_temporary_files'))
    write('STORAGE_BUDGET.json',dict(components=storage,additional_peak_bytes=peak,free_bytes=remote['disk_free'],reserve_bytes=8*1024**3,headroom_after_peak=remote['disk_free']-8*1024**3-peak,pass_budget=remote['disk_free']-peak>=8*1024**3,immutable_shared_reference_hash_verification='PENDING before finite GPU acceptance',history_deletion=False))
    # Same-backend measured C19 is conservative: it includes HSIC, absent here.
    profile=read(ROOT/'reports/medtrace_stage24c_20260919/RESOURCE_PROFILE.json')
    original=sum(r['session_seconds'] for r in profile['writers'])*3
    maintenance=2*20*len(slots)*max(r['session_seconds']/320 for r in profile['writers'])
    # Maximum observed native generation; nonnative/support measured historical mean with 50% margin.
    native_times=[r['output']['seconds'] for r in coutputs if r['query_id'] in memberships['native']]
    other_times=[r['output']['seconds'] for r in coutputs if r['query_id'] not in memberships['native']]
    gen_native=len(deps)*max(native_times)*1.5
    gen_other=(360+96)*sum(other_times)/len(other_times)*1.5
    forecast=original+maintenance+gen_native+gen_other+600+300
    write('GPU_BUDGET.json',dict(status='MEASURED_CONSERVATIVE_FORECAST_NOT_HARD_RUNTIME_GUARANTEE',original_57_writers=original,maintenance=maintenance,native_true_generation=gen_native,endpoint_support_generation=gen_other,finite_acceptance_and_loading=600,failure_exit_reserve=300,forecast_seconds=forecast,stage_cap=9000,shared_remaining=28800-sum(r['seconds'] for r in remote['ledger']['sessions']),pass_budget=forecast<=9000,limitations='Does not yet cover controller semantic-wait residence; only increases required GPU budget. No speedup from Judge reuse assumed. C19 same-backend HSIC-on is a conservative timing reference, not a claim HSIC-off costs exactly as much.'))
    write('private/COMPILED_PLAN.json',dict(tasks=tasks,slots=slots,U_masks=mask_by_t,reference_events=reference_events,rows=rowmap,eval_rows=panel,native_routes=remote['routes'],binding=digest([slots,mask_by_t,audit]),seed='Original task seed plus independent per-slot RNG',status='CPU_COMPILED_NOT_EXECUTED'))
    print(json.dumps(dict(slots=len(slots),native_states=count,judge=total,GPU_forecast=forecast,storage_peak=peak,contrast_times=len(contrast_times),contrast_sources=len(contrast_sources))))

if __name__=='__main__': compile_plan()
