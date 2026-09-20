"""Read-only CPU teacher replay check; no teacher targets or weights are changed."""
from pathlib import Path
import json,torch,sys
sys.path.insert(0,'/root/rivermind-data/job-524d/code')
from scripts.medtrace.stage18_score import query_id
p=Path('/root/rivermind-data/job-524d/run/private');base={r['query_id']:r for r in json.loads((p/'BASE.json').read_text())['records']};rows=[]
for f in Path('/root/rivermind-data/job-522/run/private/teacher').glob('*.pt'):
 x=torch.load(f,map_location='cpu',weights_only=True);b=x['binding'];q=query_id(b['source'])
 if q in base:rows.append(dict(query_id=q,teacher_tokens_equal_current_Base=b['teacher_tokens']==base[q]['output']['raw_token_ids'],teacher_kind=b['teacher']))
(p/'TEACHER_REPLAY.json').write_text(json.dumps(dict(rows=rows),indent=2));print(dict(checked=len(rows),exact=sum(x['teacher_tokens_equal_current_Base'] for x in rows)))
