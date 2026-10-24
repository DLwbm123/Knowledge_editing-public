"""Same-supervision ridge baselines; no model answers enter feature construction."""
import torch
TAUS=(-1.,-.5,-.25,0.,.1,.25,.5,.75,1.,1.5,2.,3.)

def unit(x):
    x=x.detach().cpu().double().flatten();assert torch.isfinite(x).all() and x.norm()>0
    return x/x.norm()
def projection(dim):
    generator=torch.Generator(device='cpu').manual_seed(20261020)
    return torch.randn((dim,32),generator=generator,dtype=torch.float64)/(32**.5)
def centroid(vectors):
    return unit(torch.stack([unit(v) for v in vectors]).mean(0)) if vectors else None

def features(query,source,positive,negative,kind,matrix):
    q={k:unit(v) for k,v in query.items()};s={k:unit(v) for k,v in source.items()}
    pos=centroid([v['joint'] for v in positive]);neg=centroid([v['joint'] for v in negative]);assert pos is not None
    simple=torch.tensor([float(q['joint']@pos)-(float(q['joint']@neg) if neg is not None else 0.),float(q['joint']@s['joint']),float(q['image']@s['image']),float(q['text']@s['text']),1.],dtype=torch.float64)
    if kind=='SIMPLE':return simple
    assert kind=='CONDITIONAL'
    qi=q['image']@matrix;qt=q['text']@matrix;si=s['image']@matrix;st=s['text']@matrix
    return torch.cat((simple,qi*si,qt*st,(qi-si).abs(),(qt-st).abs(),(qi*qt-si*st).abs()))
def ridge(x,y):
    assert x.ndim==2 and y.shape==(len(x),) and len(x)>0
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    return torch.linalg.solve(x.T@x+torch.eye(x.shape[1],dtype=x.dtype),x.T@y)
def risk_target(base,expert):
    if base not in ('CORRECT','INCORRECT') or expert not in ('CORRECT','INCORRECT'):return None
    return 1. if base=='CORRECT' and expert=='INCORRECT' else -1.
def choose(scores,tau):
    index=max(range(len(scores)),key=lambda i:scores[i])
    return index if scores[index]>=tau else None

def fit_heads(x,applicability,harm):
    known=[i for i,v in enumerate(applicability) if v is not None];assert known
    comp=ridge(x[known],torch.tensor([applicability[i] for i in known],dtype=x.dtype))
    known_risk=[i for i,v in enumerate(harm) if v is not None]
    values={harm[i] for i in known_risk}
    if values=={-1.,1.}:
        risk=ridge(x[known_risk],torch.tensor([harm[i] for i in known_risk],dtype=x.dtype));status='BOTH_RISK_CLASSES_DEV'
    else:risk=torch.zeros(x.shape[1],dtype=x.dtype);status='NA_RISK_COVERAGE_USE_ZERO_FOR_BOTH_METHODS'
    return comp,risk,status

def loss_counts(ids,decisions,experts,verdict):
    result=dict(repairs=0,lost_correct=0,correct=0)
    for q in ids:
        selected=decisions[q];b=verdict['BASE',q];v=b if selected is None else verdict[experts[selected],q]
        result['repairs']+=b=='INCORRECT' and v=='CORRECT';result['lost_correct']+=b=='CORRECT' and v!='CORRECT';result['correct']+=v=='CORRECT'
    return result

def select_threshold(dev_ids,score_rows,experts,verdict,r0_decisions):
    reference=loss_counts(dev_ids,r0_decisions,experts,verdict);trials=[]
    for tau in TAUS:
        selected={q:choose(score_rows[q],tau) for q in dev_ids};metrics=loss_counts(dev_ids,selected,experts,verdict);trials.append(dict(tau=tau,**metrics))
    valid=[x for x in trials if x['repairs']>=reference['repairs']]
    if not valid:return dict(status='NA_NO_DEV_WORKPOINT',reference=reference,trials=trials)
    best=max(valid,key=lambda x:(-x['lost_correct'],x['correct'],x['tau']))
    return dict(status='PASS',tau=best['tau'],reference=reference,trials=trials)

def selfcheck():
    k={'joint':torch.tensor([1.,0.]),'image':torch.tensor([1.,0.]),'text':torch.tensor([0.,1.])};p=projection(2)
    assert torch.equal(p,projection(2));a=features(k,k,[k],[],'SIMPLE',p);b=features(k,k,[k],[],'CONDITIONAL',p);assert a.shape==(5,) and b.shape==(165,) and torch.equal(a,b[:5])
    assert risk_target('CORRECT','INCORRECT')==1 and risk_target('INCORRECT','CORRECT')==-1 and risk_target('CORRECT','UNCERTAIN') is None
    comp,risk,status=fit_heads(torch.stack([a,a]),[1.,-1.],[-1.,-1.]);assert torch.count_nonzero(risk)==0 and status.startswith('NA_')
    assert choose([.5,.5],.5)==0 and choose([.5,.5],.6) is None
    ids=['dev'];scores={'dev':[.7]};experts=['E'];v={('BASE','dev'):'INCORRECT',('E','dev'):'CORRECT',('BASE','held'):'CORRECT',('E','held'):'INCORRECT'};one=select_threshold(ids,scores,experts,v,{'dev':0});v['E','held']='CORRECT';two=select_threshold(ids,scores,experts,v,{'dev':0});assert one==two and one['tau']==.5
    return dict(status='PASS',conditional_dimensions=165,same_simple_features=True,heldout_verdicts_not_used_in_threshold=True,absent_risk_class_not_fabricated=True,uncertain_not_clear_harm=True,new_model_calls=0)
if __name__=='__main__':print(selfcheck())
