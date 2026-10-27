"""Bind exactly the admitted source cohort to fixed 8-bit images and questions."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import time
import zipfile


def main():
    import numpy as np
    from PIL import Image
    root = Path(os.environ['TASK_ROOT'])
    out = root/'.private/capability_trial_v104_20261024'
    cfg = json.loads((out/'LOCAL_CONFIG.json').read_text())
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original acquisition window')))
    signal.alarm(max(1, int(cfg['hard_end']-time.time()-30)))
    assert not (out/'SOURCE_ROWS.json').exists()
    cases = json.loads((root/'.private/capability_source_v103_20261024/MODEL_COHORT_FROZEN.json').read_text())
    assert len(cases) == len({r['patient'] for r in cases}) == 32
    spec = importlib.util.spec_from_file_location('volumes', root/'experiments/relation_source_v97/source_audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = Path(cfg['source_root'])
    archive = zipfile.ZipFile(module.Volumes(sorted((source/'Padchest_GR_files').glob('PadChest_GR.zip.*'))))
    members = {Path(x.filename).name:x for x in archive.infolist() if not x.is_dir()}
    (out/'images').mkdir(exist_ok=True)
    rows, bindings, counters = [], [], {}
    for case in cases:
        assert time.time() < cfg['hard_end']
        assert case['source_record']['ImageID'] == case['image']
        member = members[case['image']]
        with Image.open(io.BytesIO(archive.read(member))) as image:
            image.load()
            pixels = np.asarray(image)
            assert pixels.ndim == 2 and np.issubdtype(pixels.dtype, np.integer)
            assert 0 <= int(pixels.min()) < int(pixels.max()) <= 65535
            rgb = Image.fromarray(np.rint(pixels.astype('float64')/257).astype('uint8')).convert('RGB')
            rgb.save(out/'images'/(case['case']+'.png'))
            bindings.append(dict(case=case['case'], image=case['image'], archive_member=member.filename,
                source_image_matches=True, size=list(image.size), source_mode=image.mode,
                source_range=[int(pixels.min()),int(pixels.max())], mapped_range=list(rgb.getextrema()[0])))
        key=(case['panel'],case['label'],case['polarity'])
        rank=counters.get(key,0)
        counters[key]=rank+1
        noun='pulmonary nodules' if case['label']=='nodule' else 'pulmonary infiltrates'
        for form, question in [('direct','Are there '+noun+' in this chest X-ray?'),
                               ('reverse','Is this chest X-ray free of '+noun+'?')]:
            present=case['polarity']=='positive'
            rows.append(dict(qid=case['case']+'_'+form,case=case['case'],panel=case['panel'],label=case['label'],
                polarity=case['polarity'],form=form,pair=case['panel']+'_'+case['label']+'_'+str(rank),
                question=question,reference='Yes' if (present if form=='direct' else not present) else 'No',
                image_path=cfg['remote_root']+'/images/'+case['case']+'.png'))
    assert len(rows)==64 and all(n==4 for n in counters.values())
    (out/'SOURCE_ROWS.json').write_text(json.dumps(rows,indent=2))
    (out/'IMAGE_BINDINGS_PRIVATE.json').write_text(json.dumps(bindings,indent=2))
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    assert size < 2_000_000_000
    (out/'MATERIALIZE_COMPLETE.json').write_text(json.dumps(dict(images=32,questions=64,local_bytes=size,model_calls=0,time=time.time()),indent=2))
    print(json.dumps(dict(images=32,questions=64,local_bytes=size)))


if __name__=='__main__':
    main()
