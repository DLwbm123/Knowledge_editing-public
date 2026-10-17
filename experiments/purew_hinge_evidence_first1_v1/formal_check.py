"""CPU tensor fixture executes the actual learner through both terminal branches."""
import contextlib,json,os,shutil,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
root=Path(tempfile.mkdtemp(prefix='e.',dir='/data/bmw/tmp'))
os.environ.update(RUN_ROOT=str(root),ARM='HE',ACTION='learn_HE_1',OWNER='1',CUDA_VISIBLE_DEVICES='',DATA_ROOT=str(root),SCOPE_DIAGNOSTICS='1')
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
for positive in (True,False):
    shutil.rmtree(root/'private',ignore_errors=True);shutil.rmtree(root/'public',ignore_errors=True)
    module=torch.nn.Linear(3,2,bias=False,dtype=torch.float16)
    with torch.no_grad():module.weight.copy_(torch.tensor([[5.,0.,-1.],[0.,5.,-1.]]))
    module.weight.requires_grad_(False);module.eval()
    model=SimpleNamespace(model=module,device=torch.device('cpu'),get_module=lambda _:module,adapter=SimpleNamespace(tokenizer=None))
    y=torch.tensor([0,1]);full=list(range(5));removed=list(range(5,10))
    prefix=torch.zeros((1,3),dtype=torch.float64)
    def position(_,k):return module.weight.double(),y,prefix
    def logits(_,k):return module.weight.double()*(.9 if k>=5 and positive else 1.),y
    def prepare(m,items):m._projection_cache=[source_loss.teacher(module.weight.detach().double(),y,prefix)]*5
    b.COUNTS=dict(forwards=0,backwards=0,committed_updates=0,generation_attempts=0)
    b.STOP_REQUESTED=False
    c.write(root/'private/window/HE/current.json',dict(quick_owner=1,task=dict(seed=310,order=1,native=dict(reference='fixture'))))
    c.write(root/'private/FROZEN_PROTOCOL.json',dict(active_arms=['HE']))
    for kind in ('REAL_VLM_ADMISSION','BASE_IDENTITY'):c.write(root/'public'/(kind+'_HE.json'),dict(status='PASS'))
    (root/'private/calibration').mkdir(parents=True)
    torch.save(Memory(module.weight,'J').state(),root/'private/calibration/Base_plain.pt')
    def count():
        assert b.COUNTS.get('attempt_slots',0)<=20
        assert b.COUNTS['backwards']+21<=300
        assert b.COUNTS['committed_updates']+1<=21
    with contextlib.ExitStack() as stack:
        for obj,name,value in [(worker,'load_learner',lambda:model),(worker,'track',lambda _:contextlib.ExitStack()),(worker,'install',lambda *a:None),(worker,'batches',lambda *a:(full,removed)),(worker,'peak_memory',lambda:{}),(source_loss,'prepare',prepare),(source_loss,'position_logits',position),(b,'logits',logits),(b,'counts',count),(b,'call',lambda *a:None),(c,'budget',lambda:None),(medical,'source_concepts',lambda _:['fixture']),(medical,'concept_indices',lambda *a:[[0],[1]])]:stack.enter_context(patch.object(obj,name,value))
        stack.enter_context(patch.object(Path,'read_text',guarded_read))
        stack.enter_context(patch.object(torch.cuda,'manual_seed_all',lambda *a:None))
        worker.learner()
    state=worker.validate_state(torch.load(root/'private/active/HE/state.pt',weights_only=True))
    assert len(list((root/'private/steps/HE/1').glob('*.json')))==20
    assert state['boundary']==positive and state['slot']==20
    assert (root/'private/control/HE/SCIENTIFIC_STOP.json').exists()==(not positive)
    assert (root/'private/final/HE/native.pt').exists()
    assert c.read(root/'public/MARGIN_DIAGNOSTIC_HE.json')['positive']==(10 if positive else 0)
    assert (root/'public/MARGIN_BIRTH_HE.json').exists()==positive
    assert not module.weight.requires_grad and not torch.cuda.is_initialized()
    results.append(dict(positive_margin=positive,slots=20,backwards=b.COUNTS['backwards'],native_final=True))
shutil.rmtree(root)
print(json.dumps(dict(status='PASS',actual_learner=True,branches=results,CUDA_calls=0,real_model_calls=0)))
