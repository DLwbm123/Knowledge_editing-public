"""Build the fixed operator control from the verified earlier execution chain."""
from pathlib import Path

here = Path(__file__).resolve().parent
source = here.parent / 'med_trainable_bridge_v1'

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('upstream execution chain changed')
    return text.replace(old, new)

worker = (source / 'eval_worker.py').read_text()
worker = replace_once(worker, '    import bridge_train\n', '    import operator_bridge\n')
old = """    for n in ('TRAIN_GUARD_EXIT.json',):
        if json.loads((root/n).read_text()).get('code') != 0:
            raise ValueError('training guard did not succeed')
    if any(json.loads((root/f'TRAIN_EXIT_{i}.json').read_text()).get('code')!=0 for i in range(4)):
        raise ValueError('training shard failed')
"""
new = """    prior = Path(os.environ['PREVIOUS_TRAIN_ROOT'])
    if json.loads((prior/'TRAIN_GUARD_EXIT.json').read_text()).get('code') != 0:
        raise ValueError('previous expert binding run did not finish')
    if json.loads((prior/'TRAIN_FROZEN.json').read_text()).get('owners') != 101:
        raise ValueError('previous expert binding incomplete')
    if json.loads((root/'ROUTES_FROZEN.json').read_text()).get('generated') != 1513:
        raise ValueError('candidate routes not frozen')
"""
worker = replace_once(worker, old, new)
worker = replace_once(worker, """                learned=json.loads((root/'weights'/f'{owner}_mix.json').read_text())
                if learned['owner']!=owner or learned['steps']!=40: raise ValueError('trained scalar missing')
""", "                endpoint_checks=[]\n")
old = """                    mix=bridge_train.TrainableBridge(a,z).to(device)
                    with torch.no_grad():mix.logit.fill_(learned['logit'])
                    return mix.requires_grad_(False)
"""
new = """                    mix=operator_bridge.OperatorBridge(a,z,0.5).to(device)
                    layer=runtime.get_module('model.layers.21.mlp.down_proj')
                    hook_class=r.q.scoped.replay.MedTraceLayerHook
                    def verify(module,args,output):
                        if endpoint_checks: return
                        mask=torch.ones(args[0].shape[:-1],device=args[0].device,dtype=torch.bool)
                        def apply(expert):
                            h=hook_class(module,expert);h.set_request_routing(mask)
                            return h._forward_hook(module,args,output)
                        with torch.no_grad():
                            for coefficient,expert in ((0.0,a),(1.0,z)):
                                endpoint=operator_bridge.OperatorBridge(a,z,coefficient).to(device)
                                if not torch.equal(apply(endpoint),apply(expert)):
                                    raise ValueError('real layer endpoint mismatch')
                            middle=apply(mix)
                            ar=a.residual(args[0]).to(output.dtype)
                            zr=z.residual(args[0]).to(output.dtype)
                            if not torch.equal(middle,output+(ar*0.5+zr*0.5)):
                                raise ValueError('real layer midpoint mismatch')
                        endpoint_checks.append(True)
                    handles.append(layer.register_forward_hook(verify))
                    return mix.requires_grad_(False)
"""
worker = replace_once(worker, old, new)
worker = replace_once(worker, '                factory=r.p.expert\n', '                handles=[]\n                factory=r.p.expert\n')
worker = replace_once(worker, '                finally:r.p.expert=factory\n', "                finally:\n                    r.p.expert=factory\n                    for handle in handles: handle.remove()\n                if len(endpoint_checks)!=1: raise ValueError('real layer endpoints not checked')\n")

guard = (source / 'gpu_guard.py').read_text()
start = guard.index("    if phase=='EVAL':\n        while not (root/'TRAIN_GUARD_EXIT.json').exists():")
end = guard.index('    gpu=[]\n', start)
guard = guard[:start] + """    if phase != 'EVAL': raise ValueError('fixed control has no training')
    prior=Path(os.environ['PREVIOUS_TRAIN_ROOT'])
    if json.loads((prior/'TRAIN_GUARD_EXIT.json').read_text()).get('code') != 0:
        raise ValueError('prior binding run failed')
    if json.loads((prior/'TRAIN_FROZEN.json').read_text()).get('owners') != 101:
        raise ValueError('prior binding incomplete')
    if json.loads((root/'ROUTES_FROZEN.json').read_text()).get('generated') != 1513:
        raise ValueError('candidate routes not frozen')
""" + guard[end:]
score = (source / 'score_driver.py').read_text()
score = replace_once(score, "Path('/private/tmp/.r.20261102.03/DECISIONS.json'),Path('/private/tmp/.r.20261101.09/DECISIONS.json')):", "Path('/private/tmp/.r.20261102.03/DECISIONS.json'),Path('/private/tmp/.r.20261101.09/DECISIONS.json'),\n                 Path('/private/tmp/.r.20261102.05/DECISIONS.json'),Path('/private/tmp/.r.20261102.06/DECISIONS.json'),\n                 Path('/private/tmp/.r.20261102.08/DECISIONS.json'),Path('/private/tmp/.r.20261103.01/REAGG_DECISIONS.private.json')):")
score = replace_once(score, "worktree/'reports/med_trainable_bridge_v1_20261103'/'arm_trainable'", "worktree/'reports/med_operator_bridge_v1_20261103'/'arm_fixed'")
score = score.replace('可训练算子桥接：150专家库开发评估', '固定算子桥：150专家库开发评估').replace('trainable bridge', 'fixed operator bridge')
for name, text in [('eval_worker.py',worker),('gpu_guard.py',guard),('score_driver.py',score)]:
    compile(text, name, 'exec')
    (here/name).write_text(text)
