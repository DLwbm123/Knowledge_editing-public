"""Freeze train/eval CT locality roles and upstream answer-choice bindings before Base."""
import json,re,csv,sys,time,shutil
from pathlib import Path
from PIL import Image,ImageDraw
csv.field_size_limit(sys.maxsize)
def prepare(data,repo,private,old,remote):
    def read(p):return json.loads(p.read_text())
    def article(path):
        m=re.search(r'PMC\d+',path,re.I);return m.group(0).upper() if m else None
    def norm(x):return ' '.join(str(x).casefold().split())
    private.mkdir(exist_ok=True,parents=True);assert not (private/'CANDIDATES_FROZEN.json').exists()
    prior_articles=set()
    for version in ('v2','v3'):
        prior=read(private.parent/('medmkeb_locality6_'+version+'_20261018')/'CANDIDATES_BEFORE_VISUAL_REVIEW.json')
        prior_articles.update(c['article'] for c in prior['candidates'])
    train=read(repo/'data/train_data.json');ev=read(repo/'data/eval_data_threehop_final.json')
    groups={n:{article(v['m_loc']) for v in a} for n,a in [('train',train),('eval',ev)]};shared=groups['train']&groups['eval']
    exposed=set()
    for n in ('HISTORY_MY_GPU.json','HISTORY_PRO5000.json'):
        h=read(old/n)
        for f in h['files_with_image_references']:
            if 'EXCLUSION_IDS' in f['path']:continue
            exposed.update(article(k) for k in f['image_keys'] if article(k))
    exposed.update(prior_articles)
    candidates=[];skips=[]
    for split,a,limit in [('train',train,64),('eval',ev,256)]:
        seen=set();pool=[]
        for v in a:
            group=article(v['m_loc'])
            reason='CROSS_SPLIT_ARTICLE' if group in shared else 'HISTORICAL_ARTICLE' if group in exposed else 'DUPLICATE_ARTICLE' if group in seen else None
            if reason:skips.append(dict(split=split,id=v['id'],reason=reason));continue
            if not group or re.search(r'^(?:image|figure|panel)\s+[a-z0-9]|^\([a-z]\)',v['m_loc_a'],re.I) or re.search(r'which image|consecutive.*slides',v['m_loc_q'],re.I):skips.append(dict(split=split,id=v['id'],reason='AMBIGUOUS_PANEL_TARGET'));continue
            try:
                with Image.open(data/v['m_loc']) as im:im.verify()
            except Exception as e:skips.append(dict(split=split,id=v['id'],reason='IMAGE_UNREADABLE',error=repr(e)));continue
            seen.add(group);pool.append(dict(split=split,original=v,article=group))
        candidates.extend(pool)
    wanted={(Path(v['original']['m_loc']).name,norm(v['original']['m_loc_q'])) for v in candidates};annotations={}
    for name in ('train.csv','train_2.csv','test.csv','test_2.csv','test_clean.csv'):
        source=data/'sources/PMC-VQA'/name
        if not source.exists():continue
        with source.open() as f:
            for row in csv.DictReader(f):
                key=(Path(row.get('Figure_path','')).name,norm(row.get('Question','')))
                if key in wanted:
                    ann=dict(row,source_csv=name);answer=str(row.get('Answer','')).strip()
                    if answer in ('A','B','C','D'):
                        ann['Answer_label']=answer;ann['Answer']=re.sub(r'^\s*[A-D]\s*[:.)]\s*','',row.get('Choice '+answer,'')).strip()
                    annotations.setdefault(key,[]).append(ann)
    admitted=[]
    for split,limit in [('train',64),('eval',256)]:
        for c in [x for x in candidates if x['split']==split]:
            v=c['original'];key=(Path(v['m_loc']).name,norm(v['m_loc_q']));matches=[a for a in annotations.get(key,[]) if norm(a['Answer'])==norm(v['m_loc_a'])]
            if not matches:skips.append(dict(split=split,id=v['id'],reason='UPSTREAM_QA_NOT_EXACT'));continue
            ann=matches[0];choices=[re.sub(r'^\s*[A-D]\s*[:.)]\s*','',ann.get('Choice '+letter,'')).strip() for letter in 'ABCD'];wrong=[s for s in choices if s and norm(s)!=norm(v['m_loc_a'])]
            if not wrong:skips.append(dict(split=split,id=v['id'],reason='NO_DISTINCT_WRONG_CHOICE'));continue
            c.update(annotation=ann,wrong_choice=wrong[0]);admitted.append(c)
            if sum(x['split']==split for x in admitted)==limit:break
    images=private/'images';images.mkdir();rows=[]
    for i,c in enumerate(admitted):
        v=c['original'];qid=('T' if c['split']=='train' else 'V')+str(sum(x['split']==c['split'] for x in admitted[:i+1])).zfill(2);source=data/v['m_loc'];dest=images/(qid+source.suffix.lower());shutil.copyfile(source,dest)
        c['row']=dict(query_id=qid,image_path=remote+'/images/'+dest.name,question=v['m_loc_q'],reference=v['m_loc_a'],dataset='SOURCE_ANCHOR',source_group=c['article'],audit_role='RETENTION_TRAIN_CANDIDATE' if c['split']=='train' else 'HELDOUT_LOCALITY_CANDIDATE');rows.append(c['row'])
    out=dict(time=time.time(),candidates=admitted,rows=rows,excluded=skips,train_count=sum(c['split']=='train' for c in admitted),eval_count=sum(c['split']=='eval' for c in admitted),historical_articles_excluded=len(exposed),shared_articles_excluded=len(shared),before_new_base=True,role='MedMKEB train-locality vs eval-locality; no private external-only support',visual_review_pending=True)
    (private/'CANDIDATES_FROZEN.json').write_text(json.dumps(out,indent=2)+'\n')
    for i in (0,1):(private/f'CANDIDATES_{i}.json').write_text(json.dumps(rows[i::2],indent=2)+'\n')
    for start in range(0,len(admitted),24):
        chunk=admitted[start:start+24];canvas=Image.new('RGB',(1600,((len(chunk)+3)//4)*370),'white');d=ImageDraw.Draw(canvas)
        for i,c in enumerate(chunk):
            row=c['row'];im=Image.open(images/Path(row['image_path']).name).convert('RGB');im.thumbnail((390,325));x=i%4*400;y=i//4*370;canvas.paste(im,(x,y+40));d.text((x,y),row['query_id'],fill='black')
        canvas.save(private/f'OVERVIEW_{start//24}.png')
    print({k:out[k] for k in ('train_count','eval_count','historical_articles_excluded','shared_articles_excluded')})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for n in ('data','repo','private','old'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--remote',required=True);a=p.parse_args();prepare(a.data,a.repo,a.private,a.old,a.remote)
