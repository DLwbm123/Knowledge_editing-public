"""Verify variable-depth caching does not hide an edited layer or its gradients."""
import torch
from types import SimpleNamespace
from v51_cache import PrefixCache

def main():
    torch.set_num_threads(2);torch.manual_seed(51)
    class Layer(torch.nn.Linear):
        def forward(self,hidden_states,**kwargs):return (torch.tanh(super().forward(hidden_states)),)
    results=[]
    for target in [18,21,29,31]:
        layers=torch.nn.ModuleList([Layer(4,4) for _ in range(32)]);layers.requires_grad_(False)
        param=torch.nn.Parameter(torch.randn(4)*.01)
        def hook(module,args,output):return (output[0]+args[0]*param,)
        handle=layers[target].register_forward_hook(hook)
        model=SimpleNamespace(model=SimpleNamespace(layers=layers),training=False,config=SimpleNamespace(output_hidden_states=False,output_attentions=False))
        x=torch.randn(2,4);batch=SimpleNamespace(forward_kwargs=lambda:dict(inputs_embeds=x,use_cache=False))
        def loss(batch):
            out=batch.forward_kwargs()['inputs_embeds']
            for layer in layers:out=layer(out,use_cache=False,output_attentions=False)[0]
            return out.square().sum()
        runtime=SimpleNamespace(model=model,compute_loss=loss)
        expected=loss(batch);g=torch.autograd.grad(expected,param)[0]
        with PrefixCache(runtime,[batch],target) as cache:
            for _ in range(2):
                actual=runtime.compute_loss(batch);ga=torch.autograd.grad(actual,param)[0]
                torch.testing.assert_close(actual,expected,atol=0,rtol=0);torch.testing.assert_close(ga,g,atol=0,rtol=0)
            assert cache.hits==cache.misses==1
        assert runtime.compute_loss==loss
        # Caching past the edited layer must be rejected.
        if target<31:
            try:PrefixCache(runtime,[batch],target+1)
            except AssertionError:pass
            else:raise AssertionError('Cached an edited layer')
        handle.remove();results.append(target)
    return dict(status='PASS',targets=results,value_and_gradient_exact=True,hooked_prefix_rejected=True)
if __name__=='__main__':print(main())
