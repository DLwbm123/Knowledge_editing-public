"""CPU tensor fixture executes the actual learner through both terminal branches."""
import contextlib,json,os,shutil,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
root=Path(tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp'))
os.environ.update(RUN_ROOT=str(root),ARM='HC',ACTION='learn_HC_1',OWNER='1',CUDA_VISIBLE_DEVICES='',DATA_ROOT=str(root),SCOPE_DIAGNOSTICS='1')
import torch,numpy as np
import worker,source_loss,medical,runtime
from core import Memory
c,b=worker.c,worker.b
torch.set_num_threads(2)
def forbidden(*a,**kw):raise AssertionError('Formal CPU fixture cannot initialize CUDA')
torch.cuda.init=torch.cuda._lazy_init=forbidden
c.rng=lambda:dict(torch_rng=torch.get_rng_state())
c.restore_rng=lambda state:torch.set_rng_state(state['torch_rng'])
original_read=Path.read_text
def guarded_read(p,*args,**kwargs):
    if p in (root/'private/EVAL_LEDGER.json',root/'private/operator/source.json'):raise PermissionError('fixture boundary')
    return original_read(p,*args,**kwargs)
c.write(root/'RUN_MANIFEST.json',dict(owned_weight_limit_bytes=1024**3))
c.write(root/'RESOURCE_LEDGER.json',dict(weights_observed_peak_bytes=0))
results=[]
for positive,trigger in ((True,'initial'),(False,'initial'),(True,'first'),(True,'never'),(True,'last')):
    shutil.rmtree(root/'private',ignore_errors=True);shutil.rmtree(root/'public',ignore_errors=True)
    module=torch.nn.Linear(3,2,bias=False,dtype=torch.float16)
    level=2.9 if trigger in ('first','last') else 5.
    with torch.no_grad():module.weight.copy_(torch.tensor([[level,0.,-1.],[0.,level,-1.]]))
    module.weight.requires_grad_(False);module.eval()
    model=SimpleNamespace(model=module,device=torch.device('cpu'),get_module=lambda _:module,adapter=SimpleNamespace(tokenizer=None))
    y=torch.tensor([0,1]);full=list(range(5));removed=list(range(5,10))
    prefix=torch.zeros((1,3),dtype=torch.float64)
    def position(_,k):return module.weight.double(),y,prefix
    def logits(_,k):return module.weight.double()*(.9 if k>=5 and positive else 1.),y
    def prepare(m,items):m._projection_cache=[source_loss.teacher(module.weight.detach().double(),y,prefix)]*5
    b.COUNTS=dict(forwards=0,backwards=0,committed_updates=0,generation_attempts=0)
    b.STOP_REQUESTED=False
    c.write(root/'private/window/HC/current.json',dict(quick_owner=1,task=dict(seed=310,order=1,native=dict(reference='fixture'))))
    c.write(root/'private/FROZEN_PROTOCOL.json',dict(active_arms=['HC']))
    for kind in ('REAL_VLM_ADMISSION','BASE_IDENTITY'):c.write(root/'public'/(kind+'_HC.json'),dict(status='PASS'))
    (root/'private/calibration').mkdir(parents=True)
    torch.save(Memory(module.weight,'J').state(),root/'private/calibration/Base_plain.pt')
    def count():
        assert b.COUNTS.get('attempt_slots',0)<=20
        assert b.COUNTS['backwards']+21<=300
        assert b.COUNTS['committed_updates']+1<=21
    real_measure=worker.measure;measure_calls=0
    def measured(*args):
        global measure_calls
        measure_calls+=1;result=real_measure(*args)
        if trigger=='never' or trigger=='first' and measure_calls==1 or trigger=='last' and measure_calls<=20:
            result['weighted_all_FIT_CE']=.1
        return result
    with contextlib.ExitStack() as stack:
        for obj,name,value in [(worker,'measure',measured),(worker,'load_learner',lambda:model),(worker,'track',lambda _:contextlib.ExitStack()),(worker,'install',lambda *a:None),(worker,'batches',lambda *a:(full,removed)),(worker,'peak_memory',lambda:{}),(source_loss,'prepare',prepare),(source_loss,'position_logits',position),(b,'logits',logits),(b,'counts',count),(b,'call',lambda *a:None),(c,'budget',lambda:None),(medical,'source_concepts',lambda _:['fixture']),(medical,'concept_indices',lambda *a:[[0],[1]])]:stack.enter_context(patch.object(obj,name,value))
        stack.enter_context(patch.object(Path,'read_text',guarded_read))
        stack.enter_context(patch.object(torch.cuda,'manual_seed_all',lambda *a:None))
        worker.learner()
    state=worker.validate_state(torch.load(root/'private/active/HC/state.pt',weights_only=True))
    assert len(list((root/'private/steps/HC/1').glob('*.json')))==20
    qualifies=positive and trigger!='never'
    assert state['boundary']==qualifies and state['slot']==20
    phases=[c.read(root/'private/steps/HC/1'/(str(i)+'.json')) for i in range(1,21)]
    expected_switch={'initial':0,'first':1,'never':None,'last':20}[trigger]
    assert state['phase']['first_fit_slot']==expected_switch,(trigger,state['phase'],[(x['slot'],x['accepted']) for x in phases])
    assert sum(x['phase_before']['name']=='EVIDENCE' for x in phases)==(20-expected_switch if expected_switch is not None else 0)
    assert all('evidence_before' in x and 'evidence_after_slot' in x and 'evidence_direction' in x for x in phases)
    assert (root/'private/control/HC/SCIENTIFIC_STOP.json').exists()==(not qualifies)
    assert (root/'private/final/HC/native.pt').exists()
    assert c.read(root/'public/MARGIN_DIAGNOSTIC_HC.json')['positive']==(10 if positive else 0)
    assert (root/'public/MARGIN_BIRTH_HC.json').exists()==qualifies
    assert not module.weight.requires_grad and not torch.cuda.is_initialized()
    results.append(dict(positive_margin=positive,trigger=trigger,slots=20,backwards=b.COUNTS['backwards'],native_final=True))
shutil.rmtree(root)
print(json.dumps(dict(status='PASS',actual_learner=True,branches=results,CUDA_calls=0,real_model_calls=0)))
