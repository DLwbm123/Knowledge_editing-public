"""Bind the frozen CT candidates to unchanged upstream PMC questions and choices."""
import csv,json,re,sys
from pathlib import Path
csv.field_size_limit(sys.maxsize)
read=lambda p:json.loads(Path(p).read_text())
norm=lambda s:' '.join(str(s).casefold().split())
def bind(data,private):
    assert not (private/'SOURCE_BINDINGS.json').exists();co=read(private/'CANDIDATES_FROZEN.json');review={x['id']:x for x in read(private/'MODALITY_REVIEW.json')['reviewed_items']};counts={};wanted={};edits=[]
    for c in co['candidates']:
        role=c['role'];counts[role]=counts.get(role,0)+1;i=counts[role];v=c['representative']['original'];key=('E' if role=='EDIT_EVAL' else 'T' if role=='RETENTION_TRAIN' else 'V')+f'{i:03}'
        if role=='EDIT_EVAL':
            if all(review[key+k]['ct_candidate'] for k in ['N','C']):edits.append(dict(id=key,candidate=c))
        elif review[key]['ct_candidate']:wanted[(Path(v['m_loc']).name,norm(v['m_loc_q']))]=dict(id=key,candidate=c,annotations=[])
    for name in ('train.csv','train_2.csv','test.csv','test_2.csv','test_clean.csv'):
        with (data/'sources/PMC-VQA'/name).open() as f:
            for row in csv.DictReader(f):
                key=(Path(row.get('Figure_path','')).name,norm(row.get('Question','')))
                if key not in wanted:continue
                ann=dict(row,source_csv=name);a=ann.get('Answer','').strip()
                if a in 'ABCD' and len(a)==1:ann['Answer_label']=a;ann['Answer']=re.sub(r'^\s*[A-D]\s*[:.)]\s*','',ann.get('Choice '+a,'')).strip()
                wanted[key]['annotations'].append(ann)
    out=[]
    for x in wanted.values():
        v=x['candidate']['representative']['original'];matches=[a for a in x['annotations'] if norm(a['Answer'])==norm(v['m_loc_a'])];x['upstream_exact']=bool(matches)
        if matches:
            a=matches[0];wrong=[re.sub(r'^\s*[A-D]\s*[:.)]\s*','',a.get('Choice '+k,'')).strip() for k in 'ABCD'];wrong=[w for w in wrong if w and norm(w)!=norm(v['m_loc_a'])];x['wrong_choice']=wrong[0] if wrong else None
        out.append(x)
    result=dict(retention=out,edits=edits);(private/'SOURCE_BINDINGS.json').write_text(json.dumps(result,indent=2)+'\n')
    return dict(retention=len(out),exact=sum(x['upstream_exact'] for x in out),edits=len(edits))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--private',type=Path,required=True);a=p.parse_args();print(bind(a.data,a.private))
