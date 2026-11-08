"""Freeze the source-reviewed DEV panel before any new model output."""
import collections
import json
from pathlib import Path

# Review of bilingual finding text, before all new model outputs. Source-sorted
# ordinal IDs are reproducible from the declared metadata snapshot, not IDs of patients.
EXPLICIT={1,5,9,14,15,16,17,23,24,35,40,41,42,43,45,46,47,49,50,51,52,54,56,57}
DATED_MISMATCH={19,29};DATED_UNVERIFIED={0,7,8,10,30,59}
LABELS=['pleural effusion','alveolar pattern'];CLASSES=['improving','stable','worsening']


def questions(finding):
    return [
        f'Compare the earlier and later chest radiographs. How has {finding} changed? Answer with exactly one word: improving, stable, or worsening.',
        f'From the first radiograph to the second, classify the change in {finding}. Answer only improving, stable, or worsening.',
        f'For {finding}, is the appearance improving, stable, or worsening between the earlier and later images? Give one word.',
        f'Assess the longitudinal change in {finding} using both radiographs in chronological order. Reply with one word: improving, stable, or worsening.',
        f'What is the change in {finding} from the earlier image to the later image? Respond only improving, stable, or worsening.',
        f'Looking at both time points, describe the direction of change of {finding}: improving, stable, or worsening. One word only.',
        f'Which longitudinal category applies to {finding} in the second chest radiograph relative to the first? Choose improving, stable, or worsening and output only that word.'
    ]


def build(audit,selection,remote_root,dataset):
    root=Path(remote_root);by={a['id']:a for a in audit};pairs={};queries={};anchors=[];probes=[];private=[]
    chosen=[dict(by[x['candidate_id']],role=x['role']) for x in selection['selection']]
    assert len(chosen)==12 and len({x['patient'] for x in chosen})==12
    for order,a in enumerate(chosen,1):
        r=a['report'];f=r['findings'][a['finding_index']];label=a['targets'][0];state=f['progression'];assert a['id'] in EXPLICIT
        name=f'pair_{a["id"]:03d}.json';paths=[str(Path(dataset)/'images'/r[k]) for k in ['PreviousImageID','ImageID']]
        pairs[name]=dict(schema='ORDERED_CXR_PAIR_V1',paths=paths)
        finding=label+' (annotated region: '+', '.join(f.get('locations') or [])+')'
        qs=questions(finding);rows=[]
        for j,q in enumerate(qs if a['role']=='anchor' else qs[:1]):
            qid=f'cxr_{a["id"]:03d}_{j}';rows.append(dict(query_id=qid,dataset='padchest_longitudinal_DEV',question=q,reference=state.capitalize(),
                image_path=str(root/'pairs'/name),original_image_path='PRIVATE_ORDERED_PAIR',progression=state,label=label,
                audit_role='NATIVE' if j==0 else 'FIT' if j<5 else 'PARAPHRASE'))
            queries[qid]=rows[-1]
        if a['role']=='anchor':
            n=len(anchors)+1;anchors.append(dict(order=n,edit_id=f'long_{n:03d}',seed=2026110800+n,native=rows[0],fit_questions=qs[1:5],
                all_rows=rows,progression=state,label=label,sequential_role_safe=True,patient_group=f'anchor_{n}'))
        else:probes.append(rows[0])
        private.append(dict(candidate_id=a['id'],role=a['role'],patient=a['patient'],original_split=a['split'],
            source_review='EXPLICIT_COMPARATIVE_FINDING_TEXT; OFFICIAL_LINK_POINTER_ONLY'))
    base_plan=[]
    for t in anchors:
        for q in t['all_rows']:base_plan.append(dict(edit_id=t['edit_id'],owner=t['order'],role=q['audit_role'],query=q))
    for q in probes:base_plan.append(dict(edit_id=anchors[0]['edit_id'],owner=0,role='PROBE_BASE',query=q))
    assert len(base_plan)==48
    return dict(tasks=anchors,probes=probes,base_plan=base_plan),pairs,private
