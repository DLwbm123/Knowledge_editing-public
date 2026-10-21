"""A capped first answer must not suppress the remaining frozen panel."""
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from .dual_execution import ExecutionBudget
from .rank_retention import consume

with tempfile.TemporaryDirectory() as d:
    root=Path(d);(root/'private').mkdir();master=root/'master';(master/'private').mkdir(parents=True);(master/'basequalify/private').mkdir(parents=True)
    def save(p,x):p.write_text(json.dumps(x))
    aliases=[f'L{i:03}' for i in range(1,7)]
    panel=[]
    for alias in aliases:
        panel.append(dict(alias=alias,base_score=True,base_answer='base',query=dict(question=alias,reference='gold',image_path='synthetic',image_sha256='bound')))
        save(master/f'basequalify/private/{alias}_GENERATION.json',dict(status='COMPLETE',raw_answer='base',raw_token_ids=[1,2]))
    save(master/'private/LOCALITY_PANEL.json',dict(rows=panel,based_only_on_Base=True))
    save(master/'private/BASE_SCORES.json',dict(scores=dict.fromkeys(aliases,True)))
    save(root/'private/binding.json',dict(model_snapshot='synthetic'))
    save(root/'bindings.json',dict(first=dict(generation={},runtime=dict(model_snapshot_sha='synthetic'))))
    save(root/'private/SOURCE_QUALIFICATION.json',dict(arms=dict(ADAPT=dict(final_correct=True,protection_feasible=True))))
    expected=dict(native_nll=[.02,3.7]);save(root/'private/EXPECTED_FINAL.json',expected)
    c=dict(panel_aliases=aliases,max_new_tokens=1024,model_forward_max=100,work_forward_max=90,work_seconds=20,KILL_seconds=30,per_generation_forward_max=2,complete_generation_requests_max=6)
    budget=ExecutionBudget(c,time.time());layer=torch.nn.Linear(3,2,bias=False);before=layer.weight.detach().clone()
    class Adapter:
        def prepare_inputs(self,image,question,target):return dict(image_sha256='bound',question=question)
        def generate_prepared_with_result(self,raw,config):
            for _ in range(3 if raw['question']=='L001' else 1):budget.before_forward(None,None)
            return SimpleNamespace(raw_token_ids=[1,2],decoded_text='output')
    runtime=SimpleNamespace(model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2)),generation_config={},adapter=Adapter())
    # Native weight checks require the model's real parameter registry.
    runtime.model=torch.nn.Sequential(layer);runtime.model.generation_config=SimpleNamespace(eos_token_id=2)
    old={k:os.environ.get(k) for k in ['RUN_LOCALITY_MASTER','RUN_OLD_BINDINGS']}
    os.environ.update(RUN_LOCALITY_MASTER=str(master),RUN_OLD_BINDINGS=str(root/'bindings.json'))
    result=dict(arms=dict(ADAPT=dict(final=expected)))
    try:consume(runtime,layer,dict(ADAPT=(None,(None,expected))),result,lambda _:before+.1,c,budget,root)
    finally:
        for k,v in old.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    assert result['retention'][0]['status']=='FORWARD_CAP_NO_RETRY'
    assert all(x['status']=='COMPLETE' for x in result['retention'][1:])
    assert budget.requests==6 and budget.forwards==7 and torch.equal(layer.weight,before)
    print(json.dumps(dict(status='PASS',case='first request capped, other five complete, original weights restored',real_model_calls=0)))
