"""Native-only support selection; requires an already isolated auxiliary pool."""
import json
import re
from pathlib import Path


def family(question):
    q=question.casefold()
    # ponytail: conservative fixed intent vocabulary; unresolved intents get no hard-U label.
    if any(w in q for w in ['treat','prevent','cause','symptom','症状']):return 'unavailable'
    if 'function' in q or 'effect of the organ' in q:return 'function'
    if any(w in q for w in ['modality','contrast ct']):return 'modality'
    if any(w in q for w in ['disease','patholog','abnormal','healthy','consolidation']):return 'pathology'
    if any(w in q for w in ['where','located','which is the kidney']):return 'location'
    if any(w in q for w in ['shape','color','intensit']):return 'appearance'
    if any(w in q for w in ['size','wide','biggest','largest','bigger','smaller']):return 'size'
    if any(w in q for w in ['how many','what two']):return 'count'
    if any(w in q for w in ['organ','body','contain','exist']):return 'identity'
    return 'unavailable'


def select(native,pool):
    used=set();hard=[];diverse=[]
    def add(row,destination):
        key=' '.join(row['question'].casefold().split())
        if key not in used:
            used.add(key);destination.append(row)
    clean=[r for r in pool if r['image_sha256']!=native['image_sha256']]
    native_family=family(native['question'])
    if native_family!='unavailable':
        for r in clean:
            if family(r['question'])==native_family and len(hard)<4:add(r,hard)
    remaining=[r for r in clean if ' '.join(r['question'].casefold().split()) not in used]
    intents=set();sources={r['source_group'] for r in hard}
    for _ in range(4):
        choices=[r for r in remaining if ' '.join(r['question'].casefold().split()) not in used]
        if not choices:break
        r=min(choices,key=lambda r:(r['source_group'] in sources,family(r['question']) in intents,int(r['source_qid'])))
        add(r,diverse);intents.add(family(r['question']));sources.add(r['source_group'])
    return hard,diverse


def prepare(tasks,pool,all_evaluation_images):
    if any(r['image_sha256'] in all_evaluation_images for r in pool):
        raise ValueError('Auxiliary pool intersects evaluation images')
    eval_text={' '.join(r['question'].casefold().split()) for t in tasks for r in t['evaluation'] if r['task']!='T0'}
    out=[]
    for t in tasks:
        p=t['semantic_fit_questions']
        if len(p)!=4 or len(set(p))!=4 or any(' '.join(q.casefold().split()) in eval_text for q in p):
            raise ValueError('P fit duplication or official evaluation text collision')
        hard,diverse=select(t['native'],pool)
        u=hard+diverse
        if not u:raise ValueError('No legal expanded U')
        out.append(dict(t,U_expanded=u,U_counts=dict(hard=len(hard),diverse=len(diverse),actual=len(u),target=8,
            unique_questions=len({r['question'].casefold() for r in u}),sources=len({r['source_group'] for r in u}),
            hard_shortfall=4-len(hard),diverse_shortfall=4-len(diverse),modality='image+text',
            hard_selection='fixed native question intent; no evaluation probe/output access')))
    return out
