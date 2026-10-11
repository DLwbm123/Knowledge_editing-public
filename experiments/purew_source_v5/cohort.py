"""Source-only cohort and sealed evaluation; no old failure packet is read."""
import json,re
from pathlib import Path

def reformulations(q):
    q=q.strip();stem=q.rstrip(' ?.')
    match=re.fullmatch(r'Where is (.+)',stem,re.I)
    if match:
        obj=re.sub(r'\s+located(?=\s+in|\s+on|$)','',match[1],flags=re.I)
        return [f'What is the location of {obj}?',f'Identify the location of {obj}.',f'Specify where {obj} is located.',f'Please state the position of {obj}.'], 'LOCATION_REWRITE'
    match=re.fullmatch(r'What (?:type|kind) of (.+) is (.+)',stem,re.I)
    if match:
        obj,context=match.groups()
        return [f'Which type of {obj} is {context}?',f'Identify the type of {obj} that is {context}.',f'Please specify the type of {obj} that is {context}.',f'What category of {obj} is {context}?'],'TYPE_REWRITE'
    # shortcut: unparsed questions retain their exact proposition in a new frame;
    # report these separately rather than claiming four independent paraphrases.
    return [f'Please answer this image question: {q}',f'For the displayed image, answer: {q}',f'Considering the image shown, {q}',f'Question about this study: {q} Give the requested information.'],'VERBATIM_REFRAME'

def construct(pool):
    images=sorted({(x['dataset'],x['image_id']) for x in pool if x['raw_source']['answer_type']=='OPEN'})
    held=set(images[::5]);source=[];protected=[];annotations={}
    for x in sorted(pool,key=lambda x:(x['dataset'],x['image_id'],int(x['qid']))):
        assert x['question']==x['raw_source']['question'] and x['answer']==x['raw_source']['answer']
        source_file=x['source_file']
        if source_file not in annotations:
            annotations[source_file]={str(r['qid']):r for r in json.loads(Path(source_file).read_text())}
        assert annotations[source_file][str(x['qid'])]==x['raw_source'] and Path(x['image_path']).is_file()
        row=dict(dataset=x['dataset'],image_path=x['image_path'],original_image_path=x['image_path'],source_group=x['dataset']+':'+x['image_id'],question=x['question'],reference=x['answer'],source_file=x['source_file'],source_qid=x['qid'],owner=0,original_primary=False)
        if (x['dataset'],x['image_id']) in held:
            protected.append(dict(row,query_id='newheld_'+str(len(protected)),audit_role='NEW_HELDOUT'))
        elif x['raw_source']['answer_type']=='OPEN' and 'damage will be caused' not in x['question'].lower():
            variants,kind=reformulations(row['question'])
            assert len(set(variants))==4
            source.append(dict(row,query_id='candidate_'+str(len(source)),audit_role='CANDIDATE',evaluation_questions=variants,variant_kind=kind,source_admission='SOURCE_BOUND_ROLE_ELIGIBLE'))
    assert (len(source),len(protected),len(images),len(held))==(84,45,50,10)
    assert not {r['source_group'] for r in source}&{r['source_group'] for r in protected}
    return dict(candidates=source,new_protection=protected,patient_independence='UNKNOWN')

def select(packet,scores):
    assert set(scores)=={x['query_id'] for x in packet['candidates']}
    if any(x is None for x in scores.values()):raise ValueError('Missing source qualification score; no cohort admitted')
    selected=[];seen=set()
    for row in packet['candidates']:
        if scores[row['query_id']]==0 and row['source_group'] not in seen:
            selected.append(row);seen.add(row['source_group'])
            if len(selected)==24:break
    if len(selected)!=24:raise ValueError(f'Only {len(selected)} distinct Base-wrong sources; frozen target24 not met')
    rows=[]
    for owner,source in enumerate(selected,1):
        row={k:v for k,v in source.items() if k not in ('evaluation_questions','variant_kind')}
        row.update(owner=owner,audit_role='NATIVE',source_admission='PASS')
        rows.append(row)
        rows.extend(dict(row,query_id=f't2g_{owner}_{i}',audit_role='T2G',source_admission='EVALUATION_ONLY',question=q,variant_kind=source['variant_kind']) for i,q in enumerate(source['evaluation_questions']))
    return selected,rows+packet['new_protection']

def training_rows(rows):
    source=[r for r in rows if r['audit_role']=='NATIVE']
    assert [r['owner'] for r in source]==list(range(1,25)) and all(r['source_admission']=='PASS' for r in source)
    return source

def past_source(rows,owner,step):
    source=training_rows(rows)
    return None if owner==1 else source[(step-1)%(owner-1)]

def selfcheck():
    rows=[dict(owner=i,query_id=str(i),audit_role='NATIVE',source_admission='PASS') for i in range(1,25)]
    rows+=[dict(owner=1,query_id='forbidden',audit_role='T2G',source_admission='EVALUATION_ONLY')]
    assert past_source(rows,1,1) is None
    assert {past_source(rows,24,s)['owner'] for s in range(1,161)}==set(range(1,24))
    assert all(past_source(rows,4,s)['audit_role']=='NATIVE' for s in range(1,161))
    assert len(set(reformulations('Where is the lesion?')[0]))==4
    bad=[dict(r,source_admission='PENDING') if r['query_id']=='1' else r for r in rows]
    try:training_rows(bad)
    except AssertionError:pass
    else:raise AssertionError('Unadmitted source entered training')
    return dict(status='PASS',history_source_only=True,evaluation_excluded=True,all_prior_owners_covered=True)
