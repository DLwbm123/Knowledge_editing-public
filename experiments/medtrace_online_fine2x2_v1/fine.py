"""Fixed 2x2 online acceptance study using the unchanged group proposal solver."""
import os,time,traceback,subprocess
from collections import deque
import torch
import scope as s
import fine_math as fm
s.q.GPUS=(6,7)
s.PROTECTED_ACTIONS.update(('fine_reference','fine_train'))
s.REFERENCE_ARMS.update(N='C',T='C',NT='C')
PARENT=s.Path(os.environ['SCOPE_PARENT'])
ARMS=('N','T','NT')
_original_baseline=s.baseline
_original_point=s.point
_original_compare=s.q.compare
_original_measure=s.measure
_original_restore=s.restore
_original_load=s.c.load
_original_emit=s.emit
_current=None
_recent=deque(maxlen=2)
_checks={}
_generating=False


def baseline(t,row):
    path=_original_baseline(t,row)
    return PARENT/path.relative_to(s.RUN) if path.is_relative_to(s.RUN) else path
s.baseline=baseline


def jobs(protected=None):
    action=os.environ['ACTION']
    if action=='scope_raw':return [(t,'C') for t in s.tasks()]
    if action=='fine_reference':return [(t,'D') for t in s.tasks()]
    if action=='fine_train':
        return [(t,ARMS[(t['order']-1+j)%3]) for t in s.tasks() for j in range(3)]
    return [(t,a) for t in s.tasks() for a in ('C','D')+ARMS]
s.jobs=jobs


def point(t,arm):
    global _current
    _current=(arm,t['order'])
    return _original_point(t,arm)
s.point=point


def compare(reference,candidate,target):
    value=_original_compare(reference,candidate,target)
    target_logit=candidate.gather(1,target[:,None]).squeeze(1)
    other=candidate.clone();other.scatter_(1,target[:,None],float('-inf'))
    value['target_margin']=(target_logit-other.max(dim=1).values).tolist()
    return value
s.q.compare=compare


def measure(runtime,hook,expert,state,batches,refs,counts,gradient=False):
    result=_original_measure(runtime,hook,expert,state,batches,refs,counts,gradient)
    _recent.append((id(state),result[1]))
    return result
s.measure=measure


def restore(before,proposed,initial,scale,radius,linear_g,required_linear,current,required,measure_gain):
    arm,owner=_current
    if arm=='D':return _original_restore(before,proposed,initial,scale,radius,linear_g,required_linear,current,required,measure_gain)
    assert arm in ARMS and [x[0] for x in _recent]==[id(before),id(proposed)]
    fine_required=fm.constraints(_recent[0][1],_recent[1][1]);attempts=[]
    def guarded(state):
        gain=measure_gain(state);assert _recent[-1][0]==id(state)
        ok,flags=fm.check(_recent[-1][1],fine_required,arm);attempts.append(flags)
        return gain if ok else torch.full_like(gain,float('-inf'))
    chosen,mode,trace=_original_restore(before,proposed,initial,scale,radius,linear_g,required_linear,current,required,guarded)
    rows=_checks.setdefault((arm,owner),[])
    rows.append(dict(step=len(rows)+1,mode=mode,checks=len(attempts),nll_failures=sum(not x['nll'] for x in attempts),token_failures=sum(not x['token'] for x in attempts),required_gain=fine_required[0].tolist(),protected_tokens=sum(int(x.sum()) for x in fine_required[2])))
    if len(rows)%20==0:s.c.write(s.RUN/'private/fine_checks'/arm/f'{owner}.json',rows)
    return chosen,mode,trace
s.restore=restore


def pre_endpoint(runtime,hook,expert,state,batches,refs,counts,task,arm):
    gain,values,_,_=_original_measure(runtime,hook,expert,state,batches,refs,counts)
    s.c.write(s.RUN/'private/pre_endpoint'/arm/f"{task['order']}.json",dict(group_gain=gain.tolist(),values=values,teacher_prefix_only=True))
s.PRE_ENDPOINT=pre_endpoint


def load(gpu):
    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    assert free>=45*1024, 'Frozen 45 GiB launch reserve not available'
    runtime,bindings=_original_load(gpu);action=os.environ['ACTION']
    limit=390000 if action=='fine_train' else 130000 if action in ('scope_raw','fine_reference') else 15000 if action=='fine_route' else 5000
    counter=[0]
    def count(*_):
        if not _generating:
            counter[0]+=1
            if counter[0]>limit:raise RuntimeError('Frozen non-generation forward partition cap reached')
        if counter[0]%100==0:s.c.budget()
    runtime.model.register_forward_pre_hook(count)
    return runtime,bindings
s.c.load=load


def emit(*args,**kwargs):
    global _generating
    assert not _generating;_generating=True
    try:return _original_emit(*args,**kwargs)
    finally:_generating=False
s.emit=emit


def verify_reference():
    for t in s.tasks():
        for arm in ('C','D'):
            for row in s.source(t):
                current=s.c.read(s.output(arm,t['order'],row));prior=s.c.read(PARENT/'private/outputs'/arm/str(t['order'])/(row['query_id']+'.json'))
                assert current['binding']==prior['binding'] and current['R0']==prior['R0'],'Rebuilt reference mismatch; do not start new arms'
    s.c.write(s.RUN/'private/REFERENCE_EXACT.json',dict(status='PASS',source_outputs=112,epoch=time.time()))


def evaluate():
    assert s.c.read(s.RUN/'private/CANDIDATES_FROZEN.json')['candidates']==40
    part=int(os.environ['PARTITION']);gpu=s.q.GPUS[part];arm=os.environ['ACTION'].removeprefix('fine_eval_');assert arm in ARMS
    ts=s.tasks()[part::2];s.old.calibration.require_memory(gpu)
    counts=dict(forwards=0,backwards=0,updates=0,generation_attempts=0)
    with s.p.lease(gpu):
        runtime,bindings=s.c.load(gpu);frozen={t['order']:s.prepare(runtime,bindings,t,s.events(t)+s.previous.data.held()) for t in ts};s.old.fp32(runtime)
        h=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
        try:
            for t in ts:
                saved=torch.load(s.point(t,arm),map_location='cpu',weights_only=True);expert=s.p.expert(saved['states']['FINAL'],t['seed'],runtime.device).requires_grad_(False)
                hook=s.q.scoped.replay.MedTraceLayerHook(runtime.get_module(s.c.LAYER),expert);hook.attach()
                try:
                    for item in frozen[t['order']]:s.emit(runtime,hook,t,arm,item,counts)
                finally:hook.detach()
        finally:h.remove();s.c.write(s.RUN/'private'/f'COUNTS_{os.environ["ACTION"]}_{part}.json',counts)


def controller():
    import pipeline
    for action in ('scope_raw','fine_reference'):
        pipeline.wait([pipeline.launch('fine.py',action,g,i) for i,g in enumerate(s.q.GPUS)])
    verify_reference()
    pipeline.wait([pipeline.launch('fine.py','fine_train',g,i) for i,g in enumerate(s.q.GPUS)])
    assert all(s.point(t,a).exists() for t in s.tasks() for a in ('C','D')+ARMS)
    s.c.write(s.RUN/'private/CANDIDATES_FROZEN.json',dict(candidates=40,epoch=time.time(),selection=False))
    for arm in ARMS:pipeline.wait([pipeline.launch('fine.py','fine_eval_'+arm,g,i) for i,g in enumerate(s.q.GPUS)])
    pipeline.wait([pipeline.launch('fine_route.py','fine_route',6,0)])
    counts=[s.c.read(p) for p in (s.RUN/'private').glob('COUNTS_*.json')]
    outputs=[s.c.read(p) for p in (s.RUN/'private/outputs').glob('*/*/*.json')]
    assert len(outputs)==3586 and sum(c['updates'] for c in counts)==6400
    assert sum(c['backwards'] for c in counts)<=304896 and sum(c['generation_attempts'] for c in counts)<=3586
    nong=sum(c['forwards'] for c in counts)-sum(o['forwards'] for o in outputs if not o.get('alias_of'));assert nong<=1500000
    s.c.write(s.RUN/'private/FINE_GENERATION_COMPLETE.json',dict(status='GENERATED_SCORING_PENDING',outputs=3586,counts=counts,non_generation_forwards=nong,epoch=time.time()))

if __name__=='__main__':
    try:
        fm.selfcheck();action=os.environ['ACTION']
        if action in ('scope_raw','fine_reference','fine_train'):s.train()
        elif action.startswith('fine_eval_'):evaluate()
        elif action=='fine_controller':controller()
        else:raise ValueError(action)
    except BaseException as error:
        s.c.write(s.RUN/'private'/f"FAILURE_{os.environ.get('ACTION')}_{os.environ.get('PARTITION')}.json",dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()));raise
