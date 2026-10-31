"""Build a private quarantine lock from identity fields only, not QA."""
import collections, json, os, sys
from pathlib import Path
root = Path(os.environ['TASK_REPO'])
sys.path.insert(0, str(root / 'experiments/medmkeb_usable_dataset_v1'))
from metadata_inventory import records, Groups
import stream_extract as s
import score_index as si

private = root / '.private/medmkeb_independent_admission_v15_20261019'
state = json.loads((private / 'STATE.json').read_text())
data = Path(state['local_data_root'])
paths = [data / 'raw/train_data.json', data / 'raw/eval_data_threehop_final.json']
if any(not p.is_file() or p.stat().st_size > 64 * 1024**2 for p in paths):
    raise ValueError('source missing or oversized')
rows = dict(zip(('train','eval'), map(records, paths)))
if (len(rows['train']),len(rows['eval'])) != (4490,2497):
    raise ValueError('unexpected data version')
selected = []
with (private / 'CANDIDATES_FROZEN.json').open('rb') as f:
    if s.nonspace(f) != b'{': raise ValueError('expected root')
    seen = set()
    while True:
        t = s.nonspace(f)
        if t == b'}': break
        if t != b'"': raise ValueError('invalid key')
        k = s.string(f, True)
        if k in seen: raise ValueError('duplicate root key')
        seen.add(k)
        if s.nonspace(f) != b':': raise ValueError('invalid separator')
        t = s.nonspace(f)
        if k == 'candidates':
            if t != b'[': raise ValueError('invalid candidates')
            while True:
                t = s.nonspace(f)
                if t == b']': break
                if t != b'{': raise ValueError('invalid candidate')
                v = si.fields(f, {'role','split','index'})
                if set(v) != {'role','split','index'}: raise ValueError('missing identity')
                if v['role'] not in ('RETENTION','EDIT') or v['split'] not in rows or type(v['index']) is not int or not 0 <= v['index'] < len(rows[v['split']]): raise ValueError('bad identity')
                selected.append(v)
                t = s.nonspace(f)
                if t == b']': break
                if t != b',': raise ValueError('invalid array separator')
        else: si.skip(f,t)
        t = s.nonspace(f)
        if t == b'}': break
        if t != b',': raise ValueError('invalid root separator')
    if 'candidates' not in seen or f.read().strip(): raise ValueError('missing candidates or trailing content')
if len({(v['split'],v['index']) for v in selected}) != len(selected): raise ValueError('duplicate identity')
g = Groups()
for split in rows.values():
    for a,b,_ in split: g.union(a,b)
protected_groups=set(); protected_articles=set()
for v in selected:
    a,b,c = rows[v['split']][v['index']]
    protected_groups.add(g.root(a))
    if not c: raise ValueError('candidate lacks article identity')
    protected_articles.add(c)
lock=[]; aggregate=collections.Counter()
for split, rs in rows.items():
    for i,(a,b,c) in enumerate(rs):
        group=g.root(a)
        flags=[]
        if group in protected_groups: flags.append('V15_FROZEN_IMAGE_GROUP')
        if c in protected_articles: flags.append('V15_FROZEN_LOCALITY_ARTICLE')
        category='QUARANTINE_KNOWN' if flags else 'UNKNOWN'
        aggregate[category]+=1
        lock.append({'official_split':split,'index':i,'edit_image_group':group,'locality_article':c,'status':category,'reasons':flags})
out=Path(os.environ['TASK_PRIVATE_OUT'])
out.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f: json.dump({'status':'IDENTITY_ONLY_NOT_ADMITTED','records':lock},f,separators=(',',':'))
print(json.dumps({'records':len(lock),'v15_frozen_candidates':len(selected),'known_quarantine':aggregate['QUARANTINE_KNOWN'],'unknown':aggregate['UNKNOWN'],'qa_decoded':0,'images_read':0,'models':0}))
