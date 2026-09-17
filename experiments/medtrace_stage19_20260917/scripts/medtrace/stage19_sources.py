"""Score-independent existing-source V3 proposal; no claim of Base qualification."""
from pathlib import Path
import json
from collections import Counter,defaultdict
import collections,random
from scripts.medtrace.stage18_support import conflict,train_row,validate_task
from scripts.medtrace.prepare_stage2_sources import normalized,reviewed_attribute
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.astra_judge_bundle import read,write_new


def build(source,private,output):
    p,v=Path(source),Path(private)
    d=read(p/'CURRENT_DATA_ALL_TRAIN_V2.json');ov=read(p/'SOURCE_OVERLAY.json')
    release=ov['rows'][0]['source_release'];train=[]
    for r in d['rows']:
     g=r['img_name'].split('/')[0]
     train.append(dict(dataset='SLAKE',image_path=str(Path(ov['rows'][0]['image_path']).parents[1]/r['img_name']),image_sha256=d['image_hashes'][g],source_group='SLAKE:'+release+':'+g,source_qid=r['qid'],question=r['question'],reference=r['answer'],role='native'))
    review=json.load(open(v/'V3_EVAL_REFERENCE_REVIEW.json'))['records']+json.load(open(v/'BLIND_REFERENCE_REVIEW.json'))['records']
    rs={str(r['source_qid']):r for r in review};ev=[r for r in ov['rows'] if ov['image_roles'][r['source_group']]=='evaluation'];good=[r for r in ev if rs[str(r['source_qid'])]['verdict']=='SUPPORTED']
    h=[r for r in good if any(conflict(n,r) for n in train)]
    near=read(v/'NEAR_DUPLICATE_SCREEN.json')
    quarantine={r[k] for r in near['candidate_pairs'] for k in ('a','b')}
    train=[r for r in train if r['source_group'].split(':')[-1] not in quarantine]
    groups=sorted({r['source_group'] for r in train});random.Random(19001).shuffle(groups);aux=set(groups[:37]);native=[r for r in train if r['source_group'] not in aux];support=[r for r in train if r['source_group'] in aux]
    uses=collections.Counter();tasks=[];ng=collections.Counter();hu=collections.Counter();gu=collections.Counter();uu=collections.Counter();ee=collections.Counter()
    bygroup=collections.defaultdict(list)
    for r in support:bygroup[r['source_group']].append(r)
    # Round-robin native source first; support ties prefer less reused image then qid.
    for n in sorted(native,key=lambda r:(r['source_qid'])):
     hs=[r for r in support if conflict(n,r) and hu[r['source_qid']]<3];he=[r for r in h if conflict(n,r)]
     if not hs or not he:continue
     pairs=[]
     for x in hs:
      gs=[g for g in bygroup[x['source_group']] if normalized(g['question'])!=normalized(x['question']) and reviewed_attribute(g['question'])!=reviewed_attribute(x['question']) and gu[g['source_qid']]<3 and len(g['reference'].split())<=4 and len(x['reference'].split())<=4 and (normalized(x['reference']) in {'yes','no'})==(normalized(g['reference']) in {'yes','no'})]
      for g in gs:pairs.append((x,g))
     if not pairs:continue
     tasks.append((n,pairs,he))
    selected=[]
    while len(selected)<48 and tasks:
     tasks.sort(key=lambda t:(ng[t[0]['source_group']],digest([19001,t[0]['source_qid']])))
     n,pairs,he=tasks.pop(0);pairs=[(x,g) for x,g in pairs if hu[x['source_qid']]<3 and gu[g['source_qid']]<3]
     if not pairs:continue
     x,g=min(pairs,key=lambda pair:(uses[pair[0]['source_group']],hu[pair[0]['source_qid']]+gu[pair[1]['source_qid']],digest([19001,pair[0]['source_qid'],pair[1]['source_qid']])))
     u=min((r for r in support if r['source_group']!=x['source_group'] and reviewed_attribute(r['question'])!=reviewed_attribute(n['question']) and normalized(r['question'])!=normalized(n['question'])),key=lambda r:(uu[r['source_group']],digest([19001,r['source_qid']])))
     hs=sorted(he,key=lambda r:(ee[r['source_qid']],digest([19001,r['source_qid']])));e=hs[0]
     hu[x['source_qid']]+=1;gu[g['source_qid']]+=1;uu[u['source_group']]+=1;uses[x['source_group']]+=1;ng[n['source_group']]+=1;ee[e['source_qid']]+=1
     selected.append(dict(native=n,H_fit=[x],G_fit=[g],U_fit=[u],H_eval=[e],H_eval_candidates=he))

    result=dict(candidates=selected,evaluation_pool=good,training_rows=train,aux_groups=sorted(aux),native_groups=sorted(set(groups)-aux),H_eval_structural=h,
        selection_seed=19001,source_split_aux=37,per_QA_H_G_reuse_cap=3,maximum_candidates=48,student_results_used=False)
    write_new(Path(output),result)
    return result

if __name__=='__main__':
    import sys
    result=build(*sys.argv[1:]);print(len(result['candidates']))


def finalize(private,public):
    """Freeze source-only review repairs before joining Base eligibility; no students."""
    private,public=Path(private),Path(public)
    draft=read(private/'CANDIDATE_PACKAGES_V3.json')['packages'];ledger=read(private/'DEV_SOURCE_ROLE_LEDGER.json')
    review=read(private/'V3_TRAIN_REFERENCE_REVIEW.json');reviewed={r['source_qid']:r for r in review['records']}
    quarantine={g for r in review['visible_near_duplicate_candidates'] for g in r['images']}
    auxiliary=set(ledger['auxiliary_groups']);good=[r for r in review['records'] if r['verdict']=='SUPPORTED' and r['source_group'] in auxiliary and r['source_group'].split(':')[-1] not in quarantine]
    hused=Counter();gused=Counter();uused=Counter();repaired=[];flow=[]
    # The full candidate order is processed before any Base decisions are read.
    for p in draft:
        t=p['training'];n=t['native'];reason=[]
        if reviewed[n['source_qid']]['verdict']!='SUPPORTED':reason.append('NATIVE_REFERENCE_UNVERIFIED')
        if n['source_group'].split(':')[-1] in quarantine:reason.append('NEAR_DUPLICATE_CANDIDATE')
        hs=[r for r in good if conflict(n,r) and hused[r['source_qid']]<3];pairs=[]
        for h in hs:
            for g in good:
                if (h['source_group']==g['source_group'] and gused[g['source_qid']]<3 and normalized(g['question'])!=normalized(h['question'])
                    and reviewed_attribute(g['question'])!=reviewed_attribute(h['question']) and len(g['reference'].split())<=4 and len(h['reference'].split())<=4
                    and (normalized(h['reference']) in {'yes','no'})==(normalized(g['reference']) in {'yes','no'})):pairs.append((h,g))
        us=[r for r in good if normalized(r['question'])!=normalized(n['question']) and reviewed_attribute(r['question'])!=reviewed_attribute(n['question'])]
        if not pairs:reason.append('NO_REVIEWED_MATCHED_H_G')
        if not us:reason.append('NO_REVIEWED_U')
        if not reason:
            h,g=min(pairs,key=lambda x:(hused[x[0]['source_qid']]+gused[x[1]['source_qid']],digest([19001,x[0]['source_qid'],x[1]['source_qid']])))
            u=min(us,key=lambda r:(uused[r['source_group']],digest([19001,r['source_qid']])))
            hused[h['source_qid']]+=1;gused[g['source_qid']]+=1;uused[u['source_group']]+=1
            updated=dict(t,H_fit=[train_row(h,'H_fit')],G_fit=[train_row(g,'G_fit')],U_fit=[train_row(u,'U_fit')]);validate_task(updated)
            repaired.append(dict(p,training=updated,status='AI_REFERENCE_SOURCE_SUPPORTED'))
        flow.append(dict(candidate_id=p['candidate_id'],source_review_eligible=not reason,excluded_reasons=reason))
    write_new(private/'REVIEWED_SOURCE_PACKAGES_V3.json',dict(packages=repaired,freeze_id=digest(repaired),reassignment='All48 source-only order, blind AI review only, no Base/student score selection; old draft preserved'))
    base=read(private/'BASE_SCORE_CLOSEOUT.json')['scores_by_source_qid'];selected=[]
    for p in repaired:
        if not base[str(p['training']['native']['source_qid'])]:selected.append(p)
    for i,p in enumerate(selected,1):p['training']['order']=i
    hrows={digest([r['image_sha256'],r['question']]):r for p in selected for r in p['evaluation']}
    # U is unrelated to every selected native predicate/question; report this realized definition.
    natives=[p['training']['native'] for p in selected]
    upool=[r for r in ledger['evaluation_pool'] if all(normalized(r['question'])!=normalized(n['question']) and reviewed_attribute(r['question'])!=reviewed_attribute(n['question']) for n in natives)]
    for p in selected:p['evaluation'] += [train_row(r,'U_eval') for r in upool]
    from scripts.medtrace.stage18_pilot import isolation
    training=[r for p in selected for r in [p['training']['native']]+p['training']['H_fit']+p['training']['G_fit']+p['training']['U_fit']]
    evaluation=[r for p in selected for r in p['evaluation']];isolation(training,evaluation)
    ids=[p['candidate_id'] for p in selected];orders={'canonical':ids}
    for seed in (18001,18002):
        perm=list(ids);random.Random(seed).shuffle(perm);orders[str(seed)]=perm
    packet=dict(version='STAGE19_REVIEWED_V3',candidate_packages=selected,orders=orders,early_anchors=ids[:10],prefixes=sorted({p for p in (1,10,25,50,100,len(ids)) if 0<p<=len(ids)}),
        status='SOURCE_AND_BASE_QUALIFIED_BUDGET_APPROVAL_PENDING',clinical_signoff=False,patient_study='UNKNOWN',new_training=False,
        initial_candidate_count=48,source_review_before_Base_selection=True,not_global_maximum_over_1154_QA=True)
    packet['freeze_id']=digest(packet);write_new(private/'DEV_EVALUATION_V3.json',packet)
    tasks=[p['training'] for p in selected];write_new(private/'TRAINING_TASKS_V3.json',dict(scope='STAGE19_REVIEWED_V3',tasks=tasks,freeze_id=digest(tasks)))
    def count(rows):
        unique={digest([r['image_sha256'],r['question']]):r for r in rows}
        return dict(slots=len(rows),unique_QA=len(unique),source_images=len({r['image_sha256'] for r in rows}),Base_correct=sum(base.get(str(r['source_qid']),False) for r in unique.values()))
    counts={role:count([r for r in training if r['role']==role]) for role in ('native','H_fit','G_fit','U_fit')}
    for role in ('H_fit','G_fit','U_fit'):counts[role].pop('Base_correct') # These sources were not all queried.
    counts['H_eval']=count([r for p in selected for r in p['evaluation'] if r['role']=='H_eval']);counts['U_eval']=count(upool)
    result=dict(status=packet['status'],candidate_edits=48,candidate_source_images=48,Base_wrong_candidates=sum(not base[str(p['training']['native']['source_qid'])] for p in draft),
        source_review_eligible_candidates=len(repaired),qualified_edits=len(selected),roles=counts,targets=dict(edits_min=32,H_QA=32,H_sources=16,U_QA=32,U_sources=16,Base_correct_H=20,Base_correct_U=20),
        deficits=dict(edits=max(0,32-len(selected)),H_QA=max(0,32-counts['H_eval']['unique_QA']),H_sources=max(0,16-counts['H_eval']['source_images']),U_QA=max(0,32-counts['U_eval']['unique_QA']),U_sources=max(0,16-counts['U_eval']['source_images']),Base_correct_H=max(0,20-counts['H_eval']['Base_correct']),Base_correct_U=max(0,20-counts['U_eval']['Base_correct'])),
        candidate_pool_not_exhaustive=True,training_freeze=digest(tasks),evaluation_freeze=packet['freeze_id'],review_type='AI_NOT_CLINICAL',new_training=False,
        near_duplicate_screen=dict(screened_images=119,automated_candidates=1,AI_extra_candidates=1,all_four_images_quarantined=True),
        H_G_max_reuse=max(Counter(r['source_qid'] for p in selected for r in p['training']['H_fit']+p['training']['G_fit']).values(),default=0))
    write_new(public/'COHORT_FLOW_V3.json',result);write_new(private/'COHORT_FLOW_V3_PRIVATE.json',dict(rows=flow,selected=ids))
    return packet,result


def near_duplicate_screen(paths):
    """Conservative candidate screen, not proof of patient/source identity."""
    from PIL import Image
    entries=[];pairs=[]
    for group,path in paths.items():
        with Image.open(path) as image:
            a=list(image.convert('L').resize((17,16),Image.Resampling.LANCZOS).getdata())
            bits=sum(int(a[y*17+x]>a[y*17+x+1])<<(y*16+x) for y in range(16) for x in range(16))
            entries.append(dict(group=group,dhash=hex(bits),size=image.size))
    # ponytail: pairwise scan is bounded to119 images; index hashes if a larger pool is approved.
    for i,a in enumerate(entries):
        for b in entries[:i]:
            distance=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
            if distance<=8:pairs.append(dict(a=a['group'],b=b['group'],distance=distance))
    return dict(images=len(entries),method='horizontal_dHash_256_LANCZOS_threshold8',threshold_frozen_before_compute=True,entries=entries,candidate_pairs=pairs)


def extension_rephrase(question):
    import re
    if question=='What is the largest organ in the picture?':return 'Which visible organ is largest in this image?'
    match=re.fullmatch(r'Does the picture contain (.+)\?',question)
    if match:return 'Is '+match[1]+' visible in this image?'
    raise ValueError('No approved meaning-preserving template for question')
