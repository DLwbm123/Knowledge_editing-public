from auxiliary_audit import routing
r=dict(edit='e',task='NEW_STRESS_HOLDOUT',source_group='s',judge_key='a',base_judge_key='b',route={'activated':True,'logical_edit_id':'x'})
x=routing([r],{'a':False,'b':True})
assert x['negative_activation']==1 and x['activated_negative_damage']['Base_correct_to_wrong']==1
assert routing([r],{})['unconditional_damage']['error_rate_bounds']==[0,1]
print('auxiliary audit self-check PASS')

from holdout_audit import summary
s=summary([r],{})
assert s['source_macro_bounds']==[0,1] and 'edit_macro_bounds' not in s
