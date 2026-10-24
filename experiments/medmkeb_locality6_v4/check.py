"""CPU checks for fresh reference, source isolation, and actual full-linear baseline loops."""
import json,tempfile,copy,types
from pathlib import Path
import torch
import module_run as r
import training_math
from fresh import source_step
from separated_step import selfcheck as separated_check
import equal_baseline as be
from m3bench_repro.editors.llava_runtime import EditorRecord
from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor

def run():
    separated_check()
    initial=r.p.tr.TT4(17,8,8).state_dict();before=r.mm.canonicalize(initial)[0]
    generator=torch.Generator().manual_seed(41);grad={k:torch.randn(v.shape,generator=generator,dtype=v.dtype) for k,v in before.items()}
    scalar=dict(step=0,second_moments=[0.,0.]);selected,updated,target=source_step(r,before,grad,scalar)
    other,other_scalar,_=training_math.step(r,before,grad,None,0.,scalar,target,'M1')
    for k in before:torch.testing.assert_close(selected[k],other[k],rtol=1e-5,atol=1e-7)
    assert scalar['step']==0 and updated==other_scalar
    class Layer(torch.nn.Linear):
        def forward(self,x,**kw):return (super().forward(x),)
    layers=torch.nn.ModuleList([Layer(3,3,bias=False) for _ in range(21)])
    with torch.no_grad():
        for layer in layers:layer.weight.copy_(torch.eye(3))
    layers.requires_grad_(False)
    class Runtime:
        def __init__(self):
            self.target_lock={'balancedit':{'targets':['head']}};self.device=torch.device('cpu');self.head=torch.nn.Linear(3,3,bias=False)
            self.head.requires_grad_(False);self.run_root=Path('.');self.model=types.SimpleNamespace(model=types.SimpleNamespace(layers=layers),training=False,config=types.SimpleNamespace(output_hidden_states=False,output_attentions=False))
        def get_module(self,name):return self.head
        def replace_module(self,name,module):self.head=module
        def build_question_batch(self,record,question=None,image_path=None):return (question,image_path)
        def extract_layer_input_key(self,batch,**kwargs):
            question,image=batch
            return torch.tensor([-1.,0.,0.]) if image else torch.tensor([1.,.1 if question else 0.,0.])
        def make_black_image(self,record,black_dir):black_dir.mkdir(parents=True,exist_ok=True);p=black_dir/'blank';p.write_text('synthetic');return p
        def build_edit_batch(self,record):return batch
        def compute_loss(self,b):
            x=b.forward_kwargs()['inputs_embeds']
            for layer in layers:x=layer(x,use_cache=False,output_attentions=False)[0]
            return (self.head(x)-b.y).square().mean()
    class Batch:
        target_token_ids=(1,);y=torch.tensor([[.2,.1,-.4]])
        x=torch.tensor([[.2,.3,.5]])
        def forward_kwargs(self):return dict(inputs_embeds=self.x,use_cache=False)
        def mask_report(self):return dict(synthetic=True)
    batch=Batch();runtime=Runtime();record=EditorRecord('E01','synthetic','Q','A','Please answer this question: Q',Path('image'), 'image',1,'test','native-only')
    editor=BalanceEditPaperSpecEditor(runtime);training=editor.apply_edit(record)
    expected=copy.deepcopy(editor.wrapper.export_state());runtime.replace_module(editor.target,editor.wrapper.base)
    task=dict(key='E01',source=dict(question='Q',reference='A',image_path='image'))
    stub=types.SimpleNamespace(c=types.SimpleNamespace(record=lambda _:record),p=types.SimpleNamespace(load_state=lambda p:torch.load(p,map_location='cpu',weights_only=False)))
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,default=str))
    def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);torch.save(v,p)
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp);counts=dict(updates=0,backwards=0)
        be.train(runtime,stub,{},task,batch,[batch]*6,0,folder/'plain',counts,lambda:None,save,write)
        state=stub.p.load_state(folder/'plain/BALANCEDIT.pt')['wrapper']
        def tensor_values(x):
            if isinstance(x,torch.Tensor):return [x]
            if isinstance(x,dict):return sum((tensor_values(v) for v in x.values()),[])
            if isinstance(x,(list,tuple)):return sum((tensor_values(v) for v in x),[])
            return []
        a,b=tensor_values(expected),tensor_values(state);assert len(a)==len(b)>0
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert counts==dict(updates=50,backwards=50)
        counts=dict(updates=0,backwards=0)
        be.train(runtime,stub,{},task,batch,[batch]*6,1,folder/'ce',counts,lambda:None,save,write)
        assert counts==dict(updates=50,backwards=200)
        ce=tensor_values(stub.p.load_state(folder/'ce/BALANCEDIT_CE.pt')['wrapper'])
        assert any(not torch.equal(x,y) for x,y in zip(b,ce))
    return dict(status='PASS',fresh_source_reference_math=True,no_historical_Fisher_or_schedule=True,original_BalancEdit_50step_tiny_runtime_exact_parity=True,shared_data_CE_50step_gradient_counts=True,source_reference_inputs_fixed=True,synthetic_only=True,new_VLM_calls=0)
if __name__=='__main__':print(json.dumps(run()))
