"""Run directly: python test_final_audit.py."""
from final_audit import summarize, paired
r=dict(edit='e',task='t',query_id='q',source_group='s',judge_key='a')
s=dict(r,judge_key='b')
assert summarize([r],{})['bounds']==[0,1]
assert summarize([r],{})['micro'] is None
p=paired([r],[s],{'a':False,'b':True})
assert p['delta_bounds']==[1,1] and p['ci95_complete_pairs']==[1,1]
assert paired([r],[s],{})['missing_pairs']==1
try:
 paired([r],[],{})
except AssertionError:
 pass
else:
 raise AssertionError('unpaired rows accepted')
print('final audit self-check PASS')
