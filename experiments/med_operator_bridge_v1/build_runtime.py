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

endpoint_worker = (source/'train_worker.py').read_text()
endpoint_worker = endpoint_worker.replace('TRAIN_', 'ENDPOINT_').replace('training guard stale', 'endpoint guard stale')
endpoint_worker = endpoint_worker.replace('ENDPOINT_SOURCE_BOUND_PRIVATE.json', 'TRAIN_SOURCE_BOUND_PRIVATE.json')
endpoint_worker = replace_once(endpoint_worker, "    if counts['updates'] > 40 * 101 or counts['generations'] != 0:",
    "    if counts['updates'] != 0 or counts['generations'] != 0 or counts['forwards'] > 101:")
endpoint_worker = replace_once(endpoint_worker, '    import bridge_train as train\n', '    import endpoint\n')
endpoint_worker = replace_once(endpoint_worker,
    "            completed.append(train.train(runtime, r, task, entry['source_rows'], config,\n                {'status':'SOURCE_101_ROLES_ADMITTED','owners':admission['owners']}, med, du, root/'weights'/f'{owner}_mix.json', budget))",
    "            completed.append(endpoint.check(runtime, r, task, entry['source_rows'], config, med, du, budget))")
endpoint_worker = replace_once(endpoint_worker, "        if counts['updates'] != 40 * len(selected):\n            raise ValueError('training update count mismatch')",
    "        if counts['updates'] != 0 or counts['generations'] != 0 or counts['forwards'] != len(selected):\n            raise ValueError('endpoint forward counts differ')")
endpoint_worker = endpoint_worker.replace('SHARD_TRAINING_COMPLETE', 'SHARD_ENDPOINT_COMPLETE')

endpoint_guard = guard.replace("if phase not in ('TRAIN','EVAL'):raise ValueError('invalid phase')", "if phase != 'ENDPOINT':raise ValueError('invalid phase')")
start = endpoint_guard.index("    if phase=='TRAIN' and arm=='D':")
end = endpoint_guard.index('    gpu=[]\n', start)
endpoint_guard = endpoint_guard[:start] + """    prior=Path(os.environ['PREVIOUS_TRAIN_ROOT'])
    if json.loads((prior/'TRAIN_GUARD_EXIT.json').read_text()).get('code') != 0:
        raise ValueError('prior binding run failed')
    if json.loads((prior/'TRAIN_FROZEN.json').read_text()).get('owners') != 101:
        raise ValueError('prior binding incomplete')
    if json.loads((root/'ROUTES_FROZEN.json').read_text()).get('generated') != 1513:
        raise ValueError('candidate routes not frozen')
""" + endpoint_guard[end:]
start = endpoint_guard.index("        if phase=='TRAIN':\n            totals=")
end = endpoint_guard.index("        write(f'{phase}_GUARD_EXIT.json'",start)
endpoint_guard = endpoint_guard[:start] + """        totals=[json.loads((root/f'ENDPOINT_COMPLETE_{i}.json').read_text()) for i in range(4)]
        owners=[row['owner'] for shard in totals for row in shard['results']]
        if len(owners)!=101 or len(set(owners))!=101:
            raise ValueError('endpoint owner coverage differs')
        if sum(shard['counts']['forwards'] for shard in totals)!=101:
            raise ValueError('endpoint forward counts differ')
        if any(row['endpoint_checks'] != 2 or row['midpoint_checks'] != 1 for shard in totals for row in shard['results']):
            raise ValueError('endpoint output checks missing')
        write('ENDPOINT_FROZEN.json',{'status':'REAL_LAYER_ENDPOINTS_PASS','owners':101,'forwards':101})
""" + endpoint_guard[end:]

guard = replace_once(guard, "    gpu=[]\n", """    while not (root/'ENDPOINT_GUARD_EXIT.json').exists():
        if (root/'ENDPOINT_GUARD_FAILURE.json').exists() or (root/'ENDPOINT_STOP.json').exists():
            raise RuntimeError('endpoint guard failed')
        if time.time() >= deadline-660: raise TimeoutError('original deadline')
        time.sleep(5)
    if json.loads((root/'ENDPOINT_GUARD_EXIT.json').read_text()).get('code') != 0:
        raise ValueError('endpoint guard failed')
    endpoint=json.loads((root/'ENDPOINT_FROZEN.json').read_text())
    if endpoint.get('status') != 'REAL_LAYER_ENDPOINTS_PASS' or endpoint.get('owners') != 101:
        raise ValueError('pre-prediction endpoint coverage missing')
    gpu=[]
""")
for name, text in [('eval_worker.py',worker),('gpu_guard.py',guard),('score_driver.py',score),
                   ('endpoint_worker.py',endpoint_worker),('endpoint_guard.py',endpoint_guard)]:
    compile(text, name, 'exec')
    (here/name).write_text(text)
