"""Locked BalancEdit full-linear writer and explicitly labelled shared-data CE adaptation."""
import time
import torch
from dataclasses import replace
from cache_v18 import PrefixCache
from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor,finite_gradients,record_seed
from m3bench_repro.editors.llava_runtime import seed_everything
from m3bench_repro.editors.routing import balanced_radius

def train(runtime,r,template,task,source_batch,retention_batches,partition,folder,counts,budget,save,write):
    arm='BALANCEDIT' if partition==0 else 'BALANCEDIT_CE'
    from pathlib import Path
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/('ONCE_'+arm)).open('x'):pass
    record=replace(r.c.record(template),record_id=task['key'],question=task['source']['question'],target=task['source']['reference'],image_path=Path(r.c.local_path(task['source']['image_path'])),official_rephrase='Please answer this question: '+task['source']['question'],router_positive_source='native-only fixed prefix, heldout rephrase not accessed')
    editor=BalanceEditPaperSpecEditor(runtime);original_run=runtime.run_root;runtime.run_root=folder
    try:
        lock=editor.config_lock();assert lock['alpha']==.2 and lock['steps_per_edit']==50 and lock['learning_rate']==.01
        seed=record_seed(record.record_id,editor.method);seed_everything(seed)
        key,positive,negative,black=editor._anchors(record,folder/'black_images')
        radius=balanced_radius(key,positive,negative,alpha=.2,distance='euclidean')
        editor.router.add(record.record_id,key,radius,source_batch.target_token_ids)
        editor.wrapper.add_edit(record.record_id);parameters=editor.wrapper.train_only(record.record_id)
        optimizer=torch.optim.Adam(parameters,lr=.01);curve=[]
        batches=[source_batch]+(retention_batches[::2] if partition else [])
        with editor._activated(record.record_id),PrefixCache(runtime,batches):
            for clock in range(1,51):
                budget();optimizer.zero_grad(set_to_none=True)
                loss=runtime.compute_loss(source_batch);assert torch.isfinite(loss);loss.backward();counts['backwards']+=1
                extra=[]
                if partition:
                    for batch in retention_batches[::2]:
                        retention=runtime.compute_loss(batch);assert torch.isfinite(retention)
                        (retention/len(retention_batches[::2])).backward();counts['backwards']+=1;extra.append(float(retention.detach()))
                assert finite_gradients(parameters);torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();counts['updates']+=1
                curve.append(dict(step=clock,source_loss=float(loss.detach()),retention_losses=extra))
        editor.wrapper.train_only(None);editor.edit_history.append(record.record_id)
        state=dict(method=editor.method,target=editor.target,router=editor.router.export_state(),wrapper=editor.wrapper.export_state(),edit_history=editor.edit_history)
        # Save exactly one final copy; optimizer can be discarded after fixed 50 steps.
        path=folder/(arm+'.pt');save(path,state);loaded=r.p.load_state(path)
        from m3bench_repro.editors.routing import MemoryRouter
        editor.router=MemoryRouter.from_state(loaded['router'],device=runtime.device);editor.wrapper.load_exported_state(loaded['wrapper'])
        actual=editor.wrapper.export_state()
        def equal(a,b):
            if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a:equal(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b)
                for x,y in zip(a,b):equal(x,y)
            else:assert a==b
        equal(state['wrapper'],actual)
        write(folder/(arm+'_TRAIN.json'),dict(status='FROZEN',arm=arm,curve=curve,steps=50,seed=seed,save_load_exact=True,parameter_count=sum(v.numel() for v in parameters),config=lock,same_retention_inputs=bool(partition),forced_on_primary=True))
        return arm
    finally:
        editor.wrapper.set_active(None);runtime.replace_module(editor.target,editor.wrapper.base);runtime.run_root=original_run

def evaluate(runtime,r,template,task,partition,folder,evaluation,counts,budget,emit,write):
    arm='BALANCEDIT' if partition==0 else 'BALANCEDIT_CE';editor=BalanceEditPaperSpecEditor(runtime)
    try:
        editor.load_editor_state(folder/(arm+'.pt'))
        for row,_,raw,expanded,binding in evaluation:
            budget()
            record=r.s.record(template,row);decision=editor._route(record)
            from m3bench_repro.editors.routing import decision_as_json
            write(folder/'routing'/arm/(row['query_id']+'.json'),dict(route=decision_as_json(decision),natural_output_reuses='FORCED' if decision.logical_edit_id==task['key'] else 'BASE',no_additional_generation=True))
            with editor._activated(task['key']):emit(folder/'outputs'/arm/(row['query_id']+'.json'),None,raw,expanded,binding)
    finally:editor.wrapper.set_active(None);runtime.replace_module(editor.target,editor.wrapper.base)
