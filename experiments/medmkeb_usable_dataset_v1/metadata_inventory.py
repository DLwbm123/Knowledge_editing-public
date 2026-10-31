"""Count image-group overlap while skipping all question/answer fields."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root/'experiments/med150_source_review_v1'),
                str(root/'experiments/med_fixed_depth_v1'),
                str(root/'experiments/medmkeb_independent_admission_v15')]
import stream_extract as s
import score_index
from inventory import image_key, article, Groups


def records(path):
    result=[]
    with path.open('rb') as f:
        if s.nonspace(f) != b'[': raise ValueError('expected array')
        while True:
            token=s.nonspace(f)
            if token==b']': break
            if token!=b'{': raise ValueError('expected record')
            row=score_index.fields(f,{'image','image_rephrase','m_loc'})
            if set(row)!={'image','image_rephrase','m_loc'} or not all(isinstance(v,str) and v for v in row.values()):
                raise ValueError('required identity field missing')
            result.append((image_key(row['image']),image_key(row['image_rephrase']),article(row['m_loc'])))
            token=s.nonspace(f)
            if token==b']': break
            if token!=b',': raise ValueError('expected array delimiter')
        if f.read().strip(): raise ValueError('trailing content')
    return result


def main():
    state=json.loads((root/'.private/medmkeb_independent_admission_v15_20261019/STATE.json').read_text())
    data=Path(state['local_data_root'])
    paths=[data/'raw/train_data.json',data/'raw/eval_data_threehop_final.json']
    if any(not p.is_file() or p.stat().st_size>64*1024**2 for p in paths):raise ValueError('source missing or oversized')
    train,ev=[records(p) for p in paths]
    if (len(train),len(ev))!=(4490,2497): raise ValueError('source version or count changed')
    g=Groups()
    for a,b,_ in train+ev:g.union(a,b)
    tg={g.root(a) for a,_,_ in train};eg={g.root(a) for a,_,_ in ev}
    ta={a for _,_,a in train if a};ea={a for _,_,a in ev if a}
    result={'source_records':{'train':len(train),'eval':len(ev)},
            'native_cross_groups':{'train':len(tg),'eval':len(eg),'shared':len(tg&eg)},
            'locality_article_groups':{'train':len(ta),'eval':len(ea),'shared':len(ta&ea)},
            'unresolved_article_records':{'train':sum(a is None for _,_,a in train),'eval':sum(a is None for _,_,a in ev)},
            'questions_answers_decoded':0,'pixels':0,'model_calls':0,
            'limits':['metadata only; overlap is not model exposure or data admission',
                      'image alias and source article identifiers may be incomplete',
                      'no historical role exclusion or candidate selection']}
    out=root/'reports/medmkeb_usable_dataset_v1_20261031/METADATA_RESULTS.json'
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
