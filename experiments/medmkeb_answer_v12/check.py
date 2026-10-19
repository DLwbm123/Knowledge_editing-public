"""Exact registered answer consumer coverage; no missing output is completion."""
import json
from pathlib import Path

def check(folder,ids):
    files={p.stem:p for p in Path(folder).glob('*.json')};assert set(files)==set(ids)
    for key,p in files.items():
        v=json.loads(p.read_text());assert isinstance(v['raw_answer'],str) and v['raw_token_ids'] and len(v['raw_token_ids'])<=1024
        assert isinstance(v['EOS'],bool) and isinstance(v['at_cap'],bool) and v['generation_trace']
    return len(files)

def selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'q.json';p.write_text(json.dumps(dict(raw_answer='x',raw_token_ids=[2],EOS=True,at_cap=False,generation_trace=[{}])));assert check(t,['q'])==1
        try:check(t,['q','missing'])
        except AssertionError:pass
        else:raise AssertionError('Missing answer accepted')
    return dict(status='PASS',complete_coverage=True,missing_rejected=True,new_model_calls=0)
