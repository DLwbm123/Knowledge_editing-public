"""Production CPU operators, finite dual, native rollback and restored next-step checks."""
import io,json,time
import torch
from core import Memory,Block,TILE
from medical import concept_indices,source_concepts,validate_margins,MedicalAdmissionRejected

def exact(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def main(protocol):
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(83)
    # Active constraints constructed from a known convex KKT optimum, not an implementation mirror.
    w=torch.zeros(4,5).half();mem=Memory(w,'C');d=torch.randn(20,dtype=torch.float64);d*=.025/d.norm()
    for name in ('Base','history','removed','margin'):
        h=torch.randn(3,20,dtype=torch.float64);h*=((.02/(h@d).square().sum())**.5)
        mem.consolidate(h,name)
    l=torch.tensor([2.,3.,4.,5.],dtype=torch.float64)
    g=-d/.01
    for k,x in zip(mem.blocks,l):g-=x*(mem.blocks[k].B.T@(mem.blocks[k].B@d))
    result,event=mem.solve(g)
    assert event['solver_reason']=='PASS',event
    assert float((result-d).norm())<1e-7,(result,d,event)
    assert all(x['native_risk_bound']<=.01000001 for x in mem.inspect(result).values())
    # Multi-birth dense covariance oracle through native quantization and checkpoint serialization.
    mem=Memory(w,'D');theta=w.double().flatten().clone();birth=[]
    for name in ('Base','history'):
        rows=torch.randn(4,20,dtype=torch.float64)*.03;mem.consolidate(rows,name);birth.append((name,rows.clone(),theta.clone()))
    master=w.float().clone()
    for slot in range(1,7):
        direction,_=mem.solve(torch.randn_like(w)*.2)
        native=(master+direction.reshape_as(w).float()).half();actual=native.double().flatten()-w.double().flatten()
        mem.advance(actual);master+=direction.reshape_as(w).float();w=native;theta+=actual
        if slot==3:
            rows=torch.randn(4,20,dtype=torch.float64)*.02;oldq=mem.blocks['history'].q;mem.consolidate(rows,'history');assert oldq==mem.blocks['history'].q
            birth.append(('history',rows.clone(),theta.clone()))
        for name,block in mem.blocks.items():
            oracle=sum(.5*float((row@(theta-anchor)).square().sum()) for key,row,anchor in birth if key==name)
            assert abs(oracle-block.q)<1e-12
    stream=io.BytesIO();torch.save(dict(native=w,master=master,memory=mem.state(),slot=6,clock=1,rng=torch.get_rng_state()),stream)
    stream.seek(0);state=torch.load(stream,weights_only=True);restored=Memory.restore(state['memory'])
    assert exact(mem.state(),restored.state()) and torch.equal(state['native'],state['master'].half())
    g=torch.randn_like(w);d1,e1=mem.solve(g);d2,e2=restored.solve(g);assert torch.equal(d1,d2) and e1==e2
    # No permissive row cap or GPU transfer: cross an actual column-tile boundary at all 68 rows.
    large=Memory(torch.zeros(TILE+3,dtype=torch.float16),'D')
    large.consolidate(torch.randn(16,TILE+3,dtype=torch.float64)*1e-5,'Base')
    large.consolidate(torch.randn(52,TILE+3,dtype=torch.float64)*1e-5,'history')
    assert sum(len(b.B) for b in large.blocks.values())==68
    large.prepare();assert large.prepared['gram'].shape==(68,68)
    v=torch.randn(TILE+3,dtype=torch.float64)*1e-4
    check=large.inspect(v);assert all(x['native_risk_bound']>=x['sampled_native_risk'] for x in check.values())
    assert all(b.B.device.type=='cpu' and b.B.dtype==torch.float64 for b in large.blocks.values())
    class Tokenizer:
        def decode(self,ids,skip_special_tokens=True):return ''.join(chr(x) for x in ids)
    source='Concept Alpha, Concept Beta';spans=concept_indices(Tokenizer(),torch.tensor([ord(x) for x in source]),source_concepts(source))
    assert [source[min(s):max(s)+1] for s in spans]==['Concept Alpha','Concept Beta']
    assert validate_margins([.2]*10)==[.2]*10
    for bad in ([0.]+[.1]*9,[-.1]+[.1]*9,[float('nan')]+[.1]*9):
        try:validate_margins(bad)
        except MedicalAdmissionRejected:pass
        else:raise AssertionError('Invalid C margin bypassed')
    assert protocol['cohort']['new_total_formal_slots']==180 and protocol['evaluation']['all_methods_total_consumers']==644
    assert protocol['budgets']['maximum_generations']==506
    return dict(status='PASS',production_multi_dual_KKT=True,multi_birth_dense_oracle=True,native_master_roundtrip_next_exact=True,
        complete_68_row_CPU_operator=True,column_tile_boundary_test=True,strict_ten_margin_gate=True,
        matrix_dimension_test=TILE+3,production_parameter_dimension=58720256,full_31GiB_live_factor_measured=False,
        numerical_envelope_not_global_model_certificate=True,CPU_seconds=time.monotonic()-started,CUDA_calls=0,model_calls=0,Judge_calls=0)
