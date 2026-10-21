"""Offline complete-denominator performance aggregation; never invokes a model."""
TASKS=('T0','T1G','T2G','T1L','T2L')


def summarize(panel,scores,base):
    result={}
    for task in TASKS:
        rows=[(t['alias'],e['query']) for t in panel for e in t['panel'] if e['task']==task]
        known=[(t,q) for t,q in rows if task=='T0' or q in base]
        qualified=[(t,q) for t,q in known if task=='T0' or base[q]==task.endswith('L')]
        def counts(items):
            yes=sum(scores.get(t+'_'+q) is True for t,q in items)
            missing=sum(t+'_'+q not in scores for t,q in items)
            n=len(items)
            return dict(correct=yes,denominator=n,missing_post=missing,rate=None if not n or missing else yes/n,
                rate_bounds=None if not n else [yes/n,(yes+missing)/n])
        eligible=counts(qualified);unknown=len(rows)-len(known)
        macros=[]
        for t in panel:
            selected=[(a,q) for a,q in qualified if a==t['alias']]
            if selected:macros.append(counts(selected))
        result[task]=dict(raw=counts(rows),qualified=eligible,unknown_base_support=unknown,
            qualified_denominator_bounds=[len(qualified),len(qualified)+unknown],
            primary_rate=eligible['rate'] if not unknown else None,
            eligible_edit_count=len(macros),edit_macro=None if unknown or not macros or any(c['rate'] is None for c in macros) else sum(c['rate'] for c in macros)/len(macros))
        if task.endswith('L'):
            result[task]['correct_to_wrong']=sum(scores.get(t+'_'+q) is False for t,q in qualified)
        result[task]['base_correct_raw']=sum(base.get(q) is True for _,q in rows)
    return result


def check():
    p=[dict(alias='E01',panel=[dict(query='Q1',task='T0'),dict(query='Q2',task='T1G'),dict(query='Q3',task='T1L')])]
    r=summarize(p,{'E01_Q1':False,'E01_Q2':True},{'Q1':False,'Q2':False,'Q3':True})
    assert r['T0']['primary_rate']==0 and r['T1G']['primary_rate']==1
    assert r['T1L']['qualified']['denominator']==1 and r['T1L']['qualified']['missing_post']==1
    assert r['T1L']['qualified']['rate_bounds']==[0,1] and r['T2L']['primary_rate'] is None
    r=summarize(p,{},{});assert r['T1L']['unknown_base_support']==1 and r['T1L']['qualified_denominator_bounds']==[0,1]
    return True
