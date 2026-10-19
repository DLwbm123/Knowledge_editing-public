"""Input-identity reuse, without using outcome labels to select an expert."""
ARMS=('SOURCE','CE','KL','CE_PROJ','RESERVE')
FIELDS=('image_sha256','question','reference','prompt_ids','attention_mask','runtime','generation')
def same_input(a,b):return all(a[k]==b[k] for k in FIELDS)
def resolve(selected,qid,base,forced):
    if selected is None:return dict(kind='BASE',path=base[qid])
    if (selected,qid) in forced:return dict(kind='FORCED',path=forced[selected,qid])
    return dict(kind='MISSING',selected=selected,query_id=qid)
def validate(rows,expected):
    keys=[(r['arm'],r['query_id']) for r in rows];assert len(keys)==len(set(keys)) and set(keys)==set(expected),'Missing/duplicate/extra natural consumer'
    assert all(r['route_source']=='FROZEN_R0_TWO_EXPERT' and r['output_path'] for r in rows)
    return len(rows)
def selfcheck():
    assert resolve(None,'q',{'q':'base'},{} )['kind']=='BASE'
    assert resolve('a','q',{'q':'base'},{('a','q'):'forced'})['path']=='forced'
    assert resolve('b','q',{'q':'base'},{('a','q'):'forced'})['kind']=='MISSING'
    r=dict(arm='SOURCE',query_id='q',route_source='FROZEN_R0_TWO_EXPERT',output_path='base')
    assert validate([r],[('SOURCE','q')])==1
    for rows in ([],[r,r]):
        try:validate(rows,[('SOURCE','q')])
        except AssertionError:pass
        else:raise AssertionError('Missing/duplicate accepted')
    b={k:k for k in FIELDS};assert same_input(b,dict(b));assert not same_input(b,dict(b,reference='changed'))
    return dict(status='PASS',reuse_OFF_own_wrong_expert=True,no_result_based_expert_selection=True,missing_duplicate_and_input_mismatch_rejected=True)
