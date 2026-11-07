"""Freeze label-independent 16-bit display conversion and source-bound T5 inputs."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    import numpy as np
    from PIL import Image
    root=Path(os.environ['T5_REMOTE_ROOT']);cfg=json.loads((root/'CONFIG.private.json').read_text())
    source=Path(cfg['dataset']);clock=json.loads((root/'CLOCK.json').read_text())
    assert time.time()<clock['hard_end']-120
    def write(name,value):
        with (root/name).open('x') as f:json.dump(value,f,ensure_ascii=False)
    cases=[json.loads(s) for s in (source/'T5_CANONICAL.private.jsonl').read_text().splitlines()]
    sequential={json.loads(s)['case_id'] for s in (source/'T5_SEQUENTIAL_ROLE_SAFE.private.jsonl').read_text().splitlines()}
    assert len(cases)==322 and len(sequential)==315
    # Detect inconsistent references for an identical image/question before any model answer.
    references=defaultdict(set)
    for case in cases:
        for role in ('edit_case','eval_case'):
            q=case[role];references[(q['image_id'],q['question'])].add(q['gold_answer'])
    conflicts={k for k,v in references.items() if len(v)>1}
    excluded=[c['case_id'] for c in cases if any((c[k]['image_id'],c[k]['question']) in conflicts for k in ('edit_case','eval_case'))]
    cases=[c for c in cases if c['case_id'] not in excluded]
    assert cases
    images={c[k]['image_id'] for c in cases for k in ('edit_case','eval_case')}
    (root/'images').mkdir()
    image_bindings={};stats=Counter();new_bytes=0
    for index,name in enumerate(sorted(images),1):
        assert Path(name).name==name and time.time()<clock['hard_end']-120
        raw=source/'images'/name
        with Image.open(raw) as im:
            a=np.asarray(im)
        assert a.ndim==2 and np.isfinite(a).all()
        low,high=int(a.min()),int(a.max());assert 0<=low<high<=65535
        # No fitted window, percentile, clipping search, spatial transform, or label-dependent rule.
        expected=np.rint((a.astype(np.float64)-low)*(255.0/(high-low))).astype(np.uint8)
        dest=root/'images'/name
        Image.fromarray(expected).convert('RGB').save(dest,format='PNG')
        new_bytes+=dest.stat().st_size
        assert new_bytes+1024**3<4_000_000_000,'Rendered-input storage guard'
        if index%25==0:
            fs=os.statvfs(root);assert fs.f_bavail*fs.f_frsize>8*1024**3
        with Image.open(dest) as im:
            restored=np.asarray(im)
            assert im.mode=='RGB' and restored.shape==expected.shape+(3,)
            assert all(np.array_equal(restored[:,:,c],expected) for c in range(3))
        # Required new model-input identity for the existing scientific prediction/Judge cache.
        # Raw images and archives are not rehashed.
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        image_bindings[name]=dict(image_path=str(dest),original_image_path=str(raw),image_sha256=digest,
            source_image_id=name,raw_min=low,raw_max=high,width=int(a.shape[1]),height=int(a.shape[0]))
        stats['images']+=1;stats['source_range_above_255']+=high>255
    import sys
    sys.path.insert(0,cfg['parent_tools'])
    import common
    tasks=[];events=[];plan=[]
    for case in cases:
        owner=int(case['case_id'].split('_')[-1]);label=case['label']
        fits=[f'Does this chest radiograph show {label}?',f'Is {label} present in this chest X-ray?',
            f'Can {label} be seen in this chest radiograph?',f'Is this chest X-ray positive for {label}?']
        def query(part,role):
            q=case[part]
            return dict(query_id=case['case_id']+('_prior' if part=='edit_case' else '_followup'),
                dataset='PadChest-GR',question=q['question'],reference=q['gold_answer'],audit_role=role,
                source_group=case['patient_id'],**image_bindings[q['image_id']])
        native=query('edit_case','NATIVE');follow=query('eval_case','T5')
        task=dict(order=owner,edit_id=case['case_id'],seed=int(common.digest([20261107,case['case_id']])[:8],16),
            native=native,fit_questions=fits,fit_status='SUPPORTED_SOURCE_ONLY',
            patient_id=case['patient_id'],sequential_role_safe=case['case_id'] in sequential,
            linkage_audit=case['linkage_audit'],source_split=case['source_split'])
        tasks.append(task)
        events.append(dict(event_id=case['case_id'],owner=owner,edit_query_id=native['query_id'],
            followup_query_id=follow['query_id'],sequential_role_safe=task['sequential_role_safe'],
            presence_code=case['presence_code'],patient_id=case['patient_id']))
        plan.extend([dict(owner=owner,edit_id=task['edit_id'],role='T0',query=native),
                     dict(owner=owner,edit_id=task['edit_id'],role='T5',query=follow)])
    write('SOURCE_INPUTS.private.json',dict(tasks=tasks,events=events,base_plan=plan))
    write('IMAGE_BINDINGS.private.json',image_bindings)
    write('SOURCE_ADMISSION.json',dict(status='PASS',source_cases=322,admitted_cases=len(cases),
        excluded_reference_conflicts=len(excluded),excluded_case_ids=excluded,Base_queries=len(plan),
        image_stats=dict(stats),render='per-image full-range affine min/max to rounded uint8 RGB; source geometry unchanged',
        rendered_bytes=new_bytes,pixel_roundtrip_checked=True,new_input_hash_scope='rendered inputs only; required cache binding',
        model_calls=0,Judge_calls=0,time=time.time()))


if __name__=='__main__':main()
