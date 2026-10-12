"""Small resume, representation, missing-bound and role-boundary checks."""
import ast, json, tempfile, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent.parent/'source'))
import journal
from tt_cores import TT4

def run():
    torch.set_num_threads(1)
    e=TT4(20261012,8,8);assert sum(p.numel() for p in e.parameters())==7168
    x=torch.randn(1,2,14336);assert torch.equal(e.residual(x),torch.zeros(1,2,4096))
    opt=torch.optim.AdamW(e.parameters(),lr=.001,weight_decay=0)
    target=torch.ones(1,2,4096)*.01
    def step(e,opt):
        opt.zero_grad(set_to_none=True);loss=(e.residual(x)-target).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(e.parameters(),1.);opt.step()
    step(e,opt)
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'p.pt';torch.save(dict(expert=e.state_dict(),optimizer=opt.state_dict()),p)
        s=torch.load(p,weights_only=True);f=TT4(20261012,8,8);f.load_state_dict(s['expert']);fo=torch.optim.AdamW(f.parameters(),lr=.001,weight_decay=0);fo.load_state_dict(s['optimizer']);step(e,opt);step(f,fo)
        assert all(torch.equal(p,f.state_dict()[k]) for k,p in e.state_dict().items())
    here=Path(__file__).parent
    namespace={}
    def extract(path,name):return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
    nodes=[extract(here/'damage_queue.py','summary'),extract(here/'bench_queue.py','bounds')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'bounds','exec'),namespace)
    bounds=namespace['bounds']([(1,None,False),(None,None,True),(0,1,False)])['bounds'];assert bounds['new_damage']==[0,1] and bounds['repaired']==[1,1]
    from m3bench_repro.editors.routed_layers import RoutedFullLinear
    base=torch.nn.Linear(3,2,bias=False);w=RoutedFullLinear(base);expert=w.add_edit('a');parameters=w.train_only('a');w.set_active('a');before=base.weight.detach().clone()
    loss=w(torch.ones(1,3)).square().sum();loss.backward();assert expert.weight.grad is not None and base.weight.grad is None
    opt=torch.optim.Adam(parameters,lr=.01);opt.step();assert torch.equal(base.weight,before)
    w.train_only(None);w.set_active(None);assert torch.equal(w(torch.ones(1,3)),base(torch.ones(1,3)))
    state=w.export_state();z=RoutedFullLinear(base);z.load_exported_state(state);z.set_active('a');w.set_active('a');assert torch.equal(w(torch.ones(1,3)),z(torch.ones(1,3)))
    from m3bench_repro.editors.routing import MemoryRouter
    r=MemoryRouter('euclidean');r.add('nearest',torch.tensor([0.]),.01);r.add('far',torch.tensor([1.]),2.)
    assert not r.route(torch.tensor([.2])).activated,'No fallback to a farther passing entry'
    rows=json.loads((here.parent/'ROWS.json').read_text());native=[x for x in rows if x['audit_role']=='NATIVE']
    assert len(rows)==261 and len(native)==24 and all(x['source_admission']=='PASS' for x in native)
    assert len([x for x in rows if x['audit_role']=='T2G'])==96
    for path in here.glob('*.py'):compile(path.read_text(),str(path),'exec')
    return dict(status='PASS',scope='CPU_SYNTHETIC_AND_ROLE_CHECKS',TT_parameters=7168,TT_resume_bit_exact=True,Balanc_native_frozen_and_bank_reload=True,zero_TT_exact=True,nearest_router_no_fallback=True,missing_bounds_shared_payload_coupling=True,journal=journal.selfcheck(),source24_roles=True,real_VLM_admission='PENDING_GPU_DEPENDENCY')
