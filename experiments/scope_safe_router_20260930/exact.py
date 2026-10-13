"""Exact shared-Boolean missing bounds; no Judge, no probability assumptions."""
from collections import Counter,defaultdict
from itertools import product

def identity(r):return (r['edit'],r['task'],r['input_id'])
def matched(a,b):
    aa={identity(r):r for r in a};bb={identity(r):r for r in b}
    assert len(aa)==len(a) and len(bb)==len(b) and aa and aa.keys()==bb.keys(),'Unmatched comparison'
    return [(aa[k]['judge_key'],bb[k]['judge_key']) for k in aa]

def delta(a,b,scores):
    """Each missing key has ONE Boolean, even across arms and repeated rows."""
    pairs=matched(a,b);c=Counter();constant=0;used=set()
    for ka,kb in pairs:
        for k,w in [(ka,1),(kb,-1)]:
            if k in scores:constant+=w*int(scores[k])
            else:c[k]+=w;used.add(k)
    n=len(pairs);c={k:v for k,v in c.items() if v}
    lo=constant+sum(min(v,0) for v in c.values());hi=constant+sum(max(v,0) for v in c.values())
    return dict(n=n,delta_min=lo/n,delta_max=hi/n,count_min=lo,count_max=hi,missing_keys=len(used),active_missing_variables=len(c),cancelled_missing_variables=len(used)-len(c))

def new_errors(a,b,scores):
    """Exact sum baseline*(1-candidate); enumerate only connected components."""
    pairs=matched(a,b);terms=Counter();constant=0
    for ka,kb in pairs:
        if ka==kb:continue
        va=scores.get(ka);vb=scores.get(kb)
        if va is True or vb is False:continue
        if va is False and vb is True:constant+=1
        elif va is False:terms[(kb,)]+=1
        elif vb is True:constant+=1;terms[(ka,)]-=1
        else:terms[(kb,)]+=1;terms[tuple(sorted((ka,kb)))]-=1
    terms={k:v for k,v in terms.items() if v};adj=defaultdict(set)
    for term in terms:
        for k in term:adj[k].update(term)
    lo=hi=constant;seen=set();sizes=[]
    for k in adj:
        if k in seen:continue
        stack=[k];component=set()
        while stack:
            v=stack.pop()
            if v in component:continue
            component.add(v);stack.extend(adj[v]-component)
        seen.update(component);ks=sorted(component);sizes.append(len(ks))
        assert len(ks)<=20,'Exact component too large; stop instead of approximate bound'
        local={t:w for t,w in terms.items() if t[0] in component};values=[]
        for bits in product((0,1),repeat=len(ks)):
            z=dict(zip(ks,bits));values.append(sum(w*all(z[v] for v in t) for t,w in local.items()))
        lo+=min(values);hi+=max(values)
    assert 0<=lo<=hi<=len(pairs)
    return dict(n=len(pairs),new_error_min=lo,new_error_max=hi,unknown_variables=len(adj),component_sizes=sizes)

def gate(d,minimum):return 'PASS' if d['delta_min']>=minimum-1e-12 else 'FAIL' if d['delta_max']<minimum-1e-12 else 'INCONCLUSIVE'

if __name__=='__main__':
    r=lambda i,k:dict(edit='e',task='t',input_id=str(i),judge_key=k)
    a=[r(1,'z'),r(2,'yes')];b=[r(1,'z'),r(2,'no')];s=dict(yes=True,no=False)
    d=delta(a,b,s);assert d['delta_min']==d['delta_max']==.5 and d['cancelled_missing_variables']==1
    assert new_errors([r(1,'z')],[r(1,'z')],{})['new_error_max']==0
    a=[r(1,'x'),r(2,'y')];b=[r(1,'y'),r(2,'x')]
    e=new_errors(a,b,{});assert (e['new_error_min'],e['new_error_max'])==(0,1)
    print('PASS: shared missing cancellation and exact correlated error count')
