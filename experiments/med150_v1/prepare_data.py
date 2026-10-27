"""Materialize the four frozen development additions using existing source data."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import time


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def identity(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    import numpy as np
    from PIL import Image
    cfg=read(os.environ['PREP_CONFIG']);root=Path(cfg['root']);clock=cfg['clock']
    assert time.time()<clock['hard_end']-620
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Original clock')))
    signal.alarm(max(1,int(clock['hard_end']-time.time()-30)))
    assert cfg['base_qualification']['status']=='PASS_ALL_FOUR_BASE_CORE_INCORRECT'
    assert len(cfg['candidates'])==len(cfg['probes'])==4
    assert not root.exists(), 'Preserve any existing run'
    root.parent.mkdir(parents=True,exist_ok=True)
    free=os.statvfs(root.parent).f_bavail*os.statvfs(root.parent).f_frsize
    assert free>8*1024**3
    root.mkdir()
    test=root/'.write_test';test.write_text('ok');assert test.read_text()=='ok';test.unlink()
    write(root/'CLOCK.json',clock)
    code=Path(cfg['official_t1g_builder'])
    spec=importlib.util.spec_from_file_location('m3bench_t1g_builder',code)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    random.seed(42);np.random.seed(42)
    ledger=read(cfg['historical_ledger']);bindings=read(cfg['historical_bindings'])
    query_rows=ledger['queries'];prepared=[]
    for owner,(candidate,probe) in enumerate(zip(cfg['candidates'],cfg['probes']),147):
        task=candidate['task'];native=task['native'];source=Path(candidate['image_path'])
        assert task['edit_id']==probe['edit_id'] and identity(source)==native['image_sha256']
        with Image.open(source) as img:img.verify()
        task=dict(task,order=owner,cohort='MED150_DEVELOPMENT_EXTENSION',native=dict(native,image_path=str(source)))
        source_rows=[dict(task['native'],query_id=f'edit_{owner}_{i}',question=q,
            audit_role='NATIVE' if i==0 else 'FIT') for i,q in enumerate([native['question']]+task['fit_questions'])]
        assert len(source_rows)==5 and len(probe['T2G'])==4
        t2g=[dict(task['native'],**dict(query_id=q['query_id'],question=q['question'],reference=q['reference'],
            audit_role='T2G',evaluation_provenance='constructed_same_semantic_development')) for q in probe['T2G']]
        t1g=[]
        with Image.open(source) as img:original=img.convert('RGB')
        for kind in ('blur','crop','contrast','noise'):
            assert time.time()<clock['hard_end']-620
            assert os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>8*1024**3
            dest=root/'variants'/str(owner)/(kind+'.jpg');dest.parent.mkdir(parents=True,exist_ok=True)
            assert not dest.exists()
            value=official.PERTURBATIONS[kind](original);value.save(dest,format='JPEG',quality=92)
            with Image.open(dest) as img:img.verify()
            t1g.append(dict(task['native'],query_id=f'dev150_{owner}_t1g_{kind}',image_path=str(dest),
                image_sha256=identity(dest),audit_role='T1G',perturbation=kind,
                semantic_status='SOURCE_ASSUMED_VISUAL_REVIEW_PENDING'))
        t2l=[]
        for question in cfg['locality_questions'].get(str(owner),[]):
            matches=[v for q,v in query_rows.items() if q!=task['edit_id'] and v['image_sha256']==native['image_sha256'] and v['question']==question]
            assert len(matches)==1
            row=matches[0];binding=bindings[row['opaque_Base_id']]
            assert binding['question']==row['question'] and binding['reference']==row['reference'] and binding['image_sha256']==row['image_sha256']
            assert binding['output']['prompt_token_ids']==binding['prompt_ids']
            t2l.append(dict(row,image_path=str(source),audit_role='T2L',historical_base_binding=binding))
        prepared.append(dict(task=task,source_rows=source_rows,T1G=t1g,T2G=t2g,T2L=t2l))
    data=dict(status='DATA_FROZEN_REVIEW_PENDING',tasks=prepared,original146_input_binding=cfg['original146_input_binding'],
        candidate_binding=cfg['candidate_binding'],seed=42,source_version=cfg['source_version'])
    write(root/'ADDITIONAL_INPUTS.json',data)
    write(root/'DATA_PREPARATION.json',dict(status='PASS_REVIEW_PENDING',tasks=4,source_rows=20,T1G=16,T2G=16,T2L=3,
        source_image_identity_checked=4,predefined_roles=True,model_calls=0,independent_clinical_verification=False,time=time.time()))
    print(json.dumps(dict(status='PASS_REVIEW_PENDING',tasks=4,T1G=16,T2G=16,T2L=3,free_bytes=free)))


if __name__=='__main__':main()
