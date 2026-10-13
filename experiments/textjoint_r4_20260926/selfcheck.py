"""Check strict DEV gate, frozen selection, and independent sampler semantics."""
import os,sys,tempfile,random
from pathlib import Path
root=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(root))
import report_r4 as report
from budget import write,read
with tempfile.TemporaryDirectory(dir=root/'tmp') as temp:
 report.ROOT=Path(temp);rows=[]
 for edit in range(24):
  for task in ['T0','T1G','T2G','T2L_PRESSURE']:
   base=task=='T2L_PRESSURE';bid=f'base-{edit}-{task}';write(report.ROOT/'private/judge/scores'/f'{bid}.json',dict(is_correct=base))
   for arm in report.ARMS:
    key=f'{arm}-{edit}-{task}';correct=(edit<12 if task=='T2L_PRESSURE' and arm=='P' else True)
    if task=='T2G' and arm=='B125' and edit==0:correct=False
    write(report.ROOT/'private/judge/scores'/f'{key}.json',dict(is_correct=correct))
    rows.append(dict(arm=arm+'@80_R0',edit=str(edit),task=task,query_id=str(edit)+task,source_group=str(edit),mode='single',prefix=1,judge_key=key,base_judge_key=bid,exact_Base_token_consistency=False))
 write(report.ROOT/'jobs/dev-check/STATUS.json',dict(status='GPU_COMPLETE'));write(report.ROOT/'jobs/dev-check/x/p001/CONSUMERS.json',rows)
 locked=report.select();assert locked['selected_writer']=='B25','One T2G error exceeds 1pp; tie prefers beta .25 over .5'
 assert report.select()['selected_writer']=='B25'
 rng=random.getstate();indices=[random.Random(123*1000003+s).randrange(4) for s in range(1,81)];assert random.getstate()==rng
 assert indices==[random.Random(123*1000003+s).randrange(4) for s in range(1,81)]
print('PASS: strict paired gate, beta tie break, immutable selection and independent RNG')
