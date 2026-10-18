"""Freeze source candidates once, before reading any new Base outcomes."""
import argparse,json,re,time,shutil,collections
from pathlib import Path
from PIL import Image,ImageDraw

def freeze(data,repo,private,remote):
    private.mkdir(parents=True,exist_ok=True)
    assert not (private/'CANDIDATES_FROZEN.json').exists(),'Never replace frozen cohort'
    history=[json.loads((private/n).read_text()) for n in ('HISTORY_MY_GPU.json','HISTORY_PRO5000.json')]
    assert all(not h['errors'] for h in history),'History metadata coverage unresolved'
    exposed=set().union(*(set(h['historical_image_keys']) for h in history))
    pattern=re.compile(r'part\d+_row_\d+_image',re.I)
    def key(s):
        m=pattern.search(s);return m.group(0) if m else Path(s).name
    train=json.loads((repo/'data/train_data.json').read_text());evaluation=json.loads((repo/'data/eval_data_threehop_final.json').read_text());allrows=train+evaluation
    parent={}
    def root(s):
        parent.setdefault(s,s)
        if parent[s]!=s:parent[s]=root(parent[s])
        return parent[s]
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for v in allrows:union(v['image'],v['image_rephrase'])
    components=collections.defaultdict(set)
    for s in parent:components[root(s)].add(s)
    banned={k for k,paths in components.items() if any(key(s) in exposed for s in paths)}
    chosen=[];seen=set();excluded=[];images=private/'images';images.mkdir(exist_ok=True)
    for v in evaluation:
        if v['modality']!='CT':continue
        group=root(v['image'])
        if group in banned:excluded.append(dict(id=v['id'],reason='HISTORICAL_CONNECTED_IMAGE_GROUP'));continue
        if group in seen:continue
        # A single CT slice cannot certify pathological subtype or clinical stage.
        if re.search(r'\bstage\b|\bt[0-4]\s*n[0-3]\b|squamous cell|large cell carcinoma|adenocarcinoma',v['pred'],re.I):
            excluded.append(dict(id=v['id'],reason='OVERPRECISE_SINGLE_IMAGE_REFERENCE'));continue
        if v['image']==v['image_rephrase']:
            excluded.append(dict(id=v['id'],reason='SAME_IMAGE_PAIR'));continue
        try:
            for role in ('image','image_rephrase'):
                source=data/v[role]
                with Image.open(source) as im:im.verify()
        except Exception as e:excluded.append(dict(id=v['id'],reason='IMAGE_UNREADABLE',error=repr(e)));continue
        seen.add(group);index=len(chosen)+1;rows=[]
        for role,name,question in [('image','native',v['src']),('image_rephrase','cross',v['src'])]:
            source=data/v[role];dest=images/(f'C{index:02d}_{name}'+source.suffix.lower());shutil.copyfile(source,dest)
            rows.append(dict(query_id=f'C{index:02d}_{name}',image_path=remote+'/images/'+dest.name,question=question,reference=v['pred'],dataset='SOURCE_ANCHOR',source_group=f'CT_COMPONENT_{index:02d}',audit_role='BASE_QUALIFICATION'))
        chosen.append(dict(index=index,original=v,component=group,component_paths=sorted(components[group]),rows=rows,reference_status='Original benchmark source answer; not independent clinical adjudication',patient_independence='UNKNOWN'))
        if len(chosen)==24:break
    rows=[r for c in chosen for r in c['rows']]
    manifest=dict(time=time.time(),source_revision='d9f38639ec2285a0e9f541e22156ec14f87271d8',candidates=chosen,rows=rows,excluded_before_base=excluded,history_components_excluded=len(banned),all_components=len(components),historical_image_keys=len(exposed),frozen_before_base=True,source_only=True,original_record_order=True)
    (private/'CANDIDATES_FROZEN.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Keep each pair on one GPU; no overlap between partitions.
    for part in (0,1):
        partition=[r for c in chosen[part::2] for r in c['rows']];assert len(partition)<=24
        (private/f'CANDIDATES_{part}.json').write_text(json.dumps(partition,indent=2)+'\n')
    for start in range(0,len(chosen),8):
        chunk=chosen[start:start+8];canvas=Image.new('RGB',(1600,((len(chunk)+1)//2)*360),'white');draw=ImageDraw.Draw(canvas)
        for i,c in enumerate(chunk):
            for j,r in enumerate(c['rows']):
                path=images/Path(r['image_path']).name;im=Image.open(path).convert('RGB');im.thumbnail((390,320));x=(i%2)*800+j*400;y=(i//2)*360;canvas.paste(im,(x,y+30));draw.text((x,y),r['query_id']+' '+c['original']['pred'][:45],fill='black')
        canvas.save(private/f'OVERVIEW_{start//8}.png')
    print(json.dumps(dict(candidates=len(chosen),base_inputs=len(rows),history_components_excluded=len(banned),hist_keys=len(exposed),source_gate='VISUAL_REVIEW_PENDING')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--remote',required=True);a=p.parse_args();freeze(a.data,a.repo,a.private,a.remote)
