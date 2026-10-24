"""Eight-image CPU display audit and reuse of saved first-token logits."""
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import time
import zipfile

ROOT=Path(os.environ['AUDIT_ROOT'])
CFG=json.loads((ROOT/'CONFIG.json').read_text())


def write(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def check():
    assert time.time()<CFG['hard_end'],'Original deadline'
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<256_000_000,'Storage cap'


def stats(a):
    import numpy as np
    return dict(dtype=str(a.dtype),shape=list(a.shape),minimum=int(a.min()),maximum=int(a.max()),
        percentiles=np.percentile(a,[0,1,5,50,95,99,100]).tolist(),unique_levels=int(np.unique(a).size),
        zero_fraction=float((a==0).mean()),at_observed_max_fraction=float((a==a.max()).mean()))


def images():
    import numpy as np
    from PIL import Image
    spec=importlib.util.spec_from_file_location('support',CFG['source_helper'])
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    files=sorted((Path(CFG['source_root'])/'Padchest_GR_files').glob('PadChest_GR.zip.*'))
    archive=zipfile.ZipFile(support.Volumes(files))
    members={}
    for f in archive.infolist():
        if not f.is_dir():members.setdefault(Path(f.filename).name,[]).append(f.filename)
    candidates={r['group']:r for r in json.loads(Path(CFG['candidates']).read_text())}
    cases=json.loads(Path(CFG['cases']).read_text());assert len(cases)==8
    results=[]
    (ROOT/'display').mkdir(exist_ok=True)
    for index,row in enumerate(cases):
        check();positive=row['role']=='positive_direct';c=candidates[row['source_group']]
        record=c['positive_control'] if positive else c
        assert record['image']==record['source_record']['ImageID']
        names=members[record['image']];assert len(names)==1
        with Image.open(io.BytesIO(archive.read(names[0]))) as im:
            im.load();mode=im.mode;raw=np.asarray(im);size=im.size
        with Image.open(row['image_path']) as im:
            im.load();rgb=np.asarray(im);prepared_size=im.size;prepared_mode=im.mode
        assert size==prepared_size and prepared_mode=='RGB'
        assert np.array_equal(rgb[:,:,0],rgb[:,:,1]) and np.array_equal(rgb[:,:,0],rgb[:,:,2])
        fixed=np.rint(raw.astype(np.float64)/257).astype('uint8')
        raw_s,prepared_s,expected_s=stats(raw),stats(rgb[:,:,0]),stats(fixed)
        alerts=[]
        if raw.max()<=255:alerts.append('RAW_RANGE_AT_MOST_8BIT_DIVIDED_BY_257')
        if prepared_s['percentiles'][-2]-prepared_s['percentiles'][1]<20:alerts.append('LOW_GLOBAL_1_TO_99_RANGE')
        if raw.min()<0 or raw.max()>65535:alerts.append('OUTSIDE_FIXED_MAPPING_RANGE')
        if prepared_s!=expected_s:alerts.append('PREPARED_STATISTICS_DIFFER_FROM_DECLARED_MAP')
        boxes=record['finding']['boxes'] if positive else []
        # Existing square padding can differ by one original pixel. Box display is approximate, not clinical localization validation.
        W,H=size;scale=336/max(W,H);padx=(max(W,H)-W)/2;pady=(max(W,H)-H)/2
        box336=[[(b[0]*W+padx)*scale,(b[1]*H+pady)*scale,(b[2]*W+padx)*scale,(b[3]*H+pady)*scale] for b in boxes]
        preview=Image.fromarray(fixed).convert('RGB');preview.thumbnail((768,768));preview.save(ROOT/'display'/f'{index}_raw_fixed.png')
        existing=Image.fromarray(rgb);existing.thumbnail((768,768));existing.save(ROOT/'display'/f'{index}_prepared.png')
        results.append(dict(index=index,qid=row['qid'],group=row['source_group'],positive=positive,source_member=names[0],
            raw_mode=mode,raw=raw_s,prepared=prepared_s,declared_map=expected_s,alerts=alerts,source_boxes=boxes,boxes_at_336=box336))
        write('IMAGE_AUDIT_PRIVATE.json',results)
    write('IMAGE_COMPLETE.json',dict(time=time.time(),count=len(results),alerts=sum(len(r['alerts']) for r in results),model_calls=0))


def logits():
    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoTokenizer
    assert not torch.cuda.is_available(),'CPU-only audit must hide all CUDA devices'
    tokenizer=AutoTokenizer.from_pretrained(CFG['tokenizer'],local_files_only=True)
    tokens={s:tokenizer.encode(s,add_special_tokens=False) for s in ['Yes','No',' yes',' no',' Yes',' No']}
    yes,no=tokens['Yes'],tokens['No']
    assert len(yes)==len(no)==1 and tokenizer.decode(yes).strip()=='Yes' and tokenizer.decode(no).strip()=='No'
    cases=json.loads(Path(CFG['cases']).read_text());assert len(cases)==8
    prior=Path(CFG['prior']);result={}
    for route in ['C','N','R']:
        saved=torch.load(prior/(route+'_LOGITS.pt'),map_location='cpu',weights_only=True)
        values={}
        for key,logit in saved.items():
            assert bool(torch.isfinite(logit).all())
            raw=json.loads((prior/'outputs'/route/(key+'.json')).read_text())
            assert raw['raw_token_ids'][0] in [yes[0],no[0]]
            assert int(logit.argmax())==raw['raw_token_ids'][0]
            values[key]=dict(s=float(logit[yes[0]].double()-logit[no[0]].double()),first_token=int(logit.argmax()))
        pairs=[]
        for neg,pos in zip(cases[::2],cases[1::2]):
            assert neg['source_group']==pos['source_group']
            a,b=values[str(neg['qid'])]['s'],values[str(pos['qid'])]['s']
            pairs.append(dict(group=neg['source_group'],finding=neg['protection_label'],negative_s=a,positive_s=b,delta=b-a,noimage_s=values[neg['source_group']+'_noimage']['s']))
        result[route]=pairs
    write('LOGIT_AUDIT_PRIVATE.json',dict(tokenization=tokens,canonical_ids=dict(Yes=yes,No=no),routes=result,model_calls=0))
    inputs=torch.load(prior/'C_INPUTS.pt',map_location='cpu',weights_only=True)
    means=torch.tensor([.48145466,.4578275,.40821073])[:,None,None]
    stds=torch.tensor([.26862954,.26130258,.27577711])[:,None,None]
    (ROOT/'model_display').mkdir(exist_ok=True)
    for i,row in enumerate(cases):
        pixels=inputs[str(row['qid'])]['pixels'].float().squeeze(0)
        assert pixels.shape==(3,336,336)
        display=((pixels*stds+means).clamp(0,1)*255).round().byte().permute(1,2,0).numpy()
        Image.fromarray(display).save(ROOT/'model_display'/f'{i}_actual_336.png')
    write('LOGIT_COMPLETE.json',dict(time=time.time(),routes=3,pairs_per_route=4,canonical_ids=dict(Yes=yes,No=no),model_calls=0))


if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU audit deadline')))
    signal.alarm(max(1,int(CFG['hard_end']-time.time())))
    check()
    if os.environ['AUDIT_MODE']=='images':images()
    else:logits()
