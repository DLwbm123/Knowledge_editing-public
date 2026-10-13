import ast, json, types, tempfile
from pathlib import Path
import torch
import numpy as np
source=Path(__file__).with_name('run146.py').read_text()
tree=ast.parse(source)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('trajectory','request_stop')]

def check(stop=False):
    with tempfile.TemporaryDirectory() as folder:
        r=Path(folder); active=r/'private/active/single_R0.pt'; active.parent.mkdir(parents=True)
        weight=torch.nn.Parameter(torch.tensor([[.1,.2],[.3,.4]]))
        state=dict(lock='lock',mode='single_R0',owner=1,step=1,committed=23201,master=weight.detach().clone(),native=weight.detach().clone(),rng={},numpy_rng=('MT19937',torch.from_numpy(np.random.get_state()[1].astype('int64')),0,0,0.),event={})
        torch.save(state,active)
        saved=[];writes={};panels=[];counter=dict(committed_updates=0,backwards=0)
        def save(p,s):
            if 'step' in s:saved.append(s['step'])
            p.parent.mkdir(parents=True,exist_ok=True);torch.save(s,p)
        c=types.SimpleNamespace(read=lambda p:{},digest=lambda p:'lock',restore_rng=lambda r:None,rng=lambda:{},write=lambda p,d:writes.update({str(p):d}),budget=lambda:None)
        runtime=types.SimpleNamespace(device='cpu')
        def logits(runtime,batch):return weight,torch.tensor([0,1])
        ns=dict(c=c,R=r,Path=Path,COUNTS=counter,STOP_REQUESTED=stop,batch_cpu=lambda *a:None,logits=logits,proposals=lambda g,m:(m-.01*g,{}),save=save,call=lambda k:None,counts=lambda:None,guard=lambda *a:None,write_progress=lambda **d:None,panel=lambda *a:panels.append(a),PREFIXES=[],print=lambda *a,**k:None)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'trajectory','exec'),ns)
        try:ns['trajectory'](runtime,{},[dict(fit_questions=[1,2,3,4])],weight,weight.detach().clone(),[],{},'single_R0')
        except SystemExit:
            assert stop
        if stop:
            assert saved==[2] and counter['committed_updates']==1 and not panels
            restored=torch.load(active,weights_only=True)
            assert restored['step']==2 and restored['committed']==23202
        else:
            assert saved==list(range(20,161,20)) and counter['committed_updates']==159
            assert len(panels)==1 and not active.exists()
        assert counter['backwards']==counter['committed_updates']*2
check();check(True)
print('PASS: sparse checkpoint boundaries, resume step, final evaluation boundary, cooperative stop, compute counters')
