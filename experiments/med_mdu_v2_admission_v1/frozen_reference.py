"""Compute paired frozen pilot references and save unscored greedy outputs."""
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import os
import sys
import time

import torch


def read(path):
    return json.loads(path.read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def main(root):
    plan = read(root/'protocol.json')
    lane = int(os.environ['Q_LANE'])
    assert lane in (0,1)
    assert torch.cuda.device_count()==1
    torch.cuda.set_per_process_memory_fraction(plan['limit_bytes'][lane] /
        torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    outputs=root.parent
    parent=outputs/'medtrace-curriculum146-v1-20261017/run/private'
    stages=[outputs/name/'run/MDU' for name in (
        'med-mdu-boundary-v1-20261105','med-medinit-101-v1-20261104','med-mdu150-v1-20261105')]
    sys.path[:0]=[str(p) for p in (stages[-1]/'tools',outputs/'med-fixed-depth-v1-20261030/run/tools',
        parent/'tools',parent/'source',Path(os.environ['M3BENCH_LLAVA_SOURCE']))]
    from llava.model import LlavaMistralForCausalLM
    from m3bench_repro.editors.llava_runtime import EditorRecord,seed_everything
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
    from methods.medtrace.core import MedTraceLayerHook
    from scripts.editor_paperspec_formal import assert_official_llavamed_source,load_runtime
    from structures import TT4
    from train_pair import state_digest

    manifest=read(root/'panel.json')
    all_items=manifest['entries']
    calibration = plan.get('kl_calibration', False)
    formal = plan.get('formal_pilot', False)
    if calibration:
        assert manifest['status'] == 'DEVELOPMENT_KL_ONLY' and len(all_items) == 8
    elif formal:
        assert manifest['status'] == 'ASTRA62_REVIEW_PASSED' and len(all_items) == 30
    else:
        assert manifest['status'] == 'MACHINE_READY_BLIND_REVIEW_PENDING' and len(all_items) == 30
    assert len({i['source']['source_rows'][0]['image_sha256'] for i in all_items}) == len(all_items)
    items=all_items[lane::2]
    rows=[]
    for item in ([] if calibration else items):
        src=item['source']
        def add(role,q,reference,base=None):
            rows.append({'n':len(rows),'owner':item['owner'],'role':role,'question':q,'reference':reference,
                         'base':base,'source':src,'image':src['source_rows'][0]['image_path'],
                         'image_sha':src['source_rows'][0]['image_sha256']})
        for row in src['source_rows']:
            add(row['audit_role'].casefold(),row['question'],row['reference'])
        positives=[r for r in item['views'] if r['role'] in ('holdout_modal','holdout_context')]
        assert len(positives)==2 and len(item['negative']['holdout'])==1
        for row in positives:
            add(row['role'],row['question'],row['reference'])
        for row in item['negative']['holdout']:
            b=row['base_continuation']
            assert row['label']=='independent' and b['sequence_contract']=='continuation_only' and b['raw_token_ids']
            assert row['image_sha256']==src['source_rows'][0]['image_sha256']
            add('holdout_independent',row['question'],row['gold_answer'],b)
        if formal:
            for row in item['historical_t2l']:
                add('historical_t2l', row['question'], row['reference'])
    n_rows = len(rows)
    assert n_rows == (0 if calibration else 120 + sum(len(i['historical_t2l']) for i in items) if formal else 120)
    bank=load(outputs/'med150-v1-20261027/run/retained/MED150.pt')
    med=bank['states']
    tasks={r['edit_id']:r for r in bank['tasks']}
    assert len(tasks)==len(med)==len(bank['R0_entries'])==150
    by_order={r['order']:r['edit_id'] for r in tasks.values()}
    mdu={}
    first={}
    for stage in stages:
        for path in stage.glob('TRAIN_COMPLETE_*.json'):
            for info in read(path)['results']:
                o=info['owner']
                edit=by_order[o]
                ck=load(stage/'weights'/f'{o}_MDU.pt')
                assert edit not in mdu and ck['steps']==info['steps']==160
                assert ck['initial_digest']==state_digest(med[edit]) and state_digest(ck['state'])==info['final_digest']
                mdu[edit]=ck['state']
                first[edit]=ck['curve'][0]['ce']
    assert mdu.keys()==med.keys()

    assert_official_llavamed_source()
    original=LlavaMistralForCausalLM.from_pretrained
    old=LlavaMistralForCausalLM.__dict__.get('from_pretrained')
    @classmethod
    def on_cpu(cls,name,*a,**kw):
        kw.update(low_cpu_mem_usage=True,device_map={'':'cpu'})
        return original(name,*a,**kw)
    LlavaMistralForCausalLM.from_pretrained=on_cpu
    try:
        runtime=load_runtime(parent/'cpu_gate','cuda:0')
    finally:
        if old is None:del LlavaMistralForCausalLM.from_pretrained
        else:LlavaMistralForCausalLM.from_pretrained=old
    runtime.model.eval().half().to('cuda:0')
    assert not any(p.requires_grad for p in runtime.model.parameters())
    runtime.run_root=root
    layer=runtime.get_module('model.layers.21.mlp.down_proj')
    routing_layer=runtime.target_lock['balancedit']['targets'][0]
    router=MemoryRouter.from_state({'distance':'euclidean','entries':bank['R0_entries']},device=runtime.device)
    assert runtime.generation_config['do_sample'] is False and runtime.generation_config['max_new_tokens']==1024
    counts=Counter()
    eos=runtime.adapter.tokenizer.eos_token_id
    states={'MED':med,'MDU':mdu}

    def record(row, answer):
        src=row['source']
        file=Path(row['image'])
        if not file.is_file() and '/DataP/' in row['image']:
            file=Path(os.environ['FROZEN_IMAGE_ROOT'])/row['image'].split('/DataP/',1)[1]
        assert file.is_file()
        return EditorRecord(src['edit_id'],src['dataset'],row['question'],answer,row['question'],
            file,src['source_rows'][0]['original_image_path'],row['owner'],'FROZEN_PILOT_REFERENCE')

    print('MODEL_READY',flush=True)
    prefill=(runtime.model.model.embed_tokens,runtime.model.model.mm_projector,runtime.model.get_vision_tower())

    def prepare(r,q,raw_ids=None):
        for module in prefill:module.half()
        try:
            if raw_ids is None:
                b=runtime.build_edit_batch(q)
                assert b.image_sha256==r['image_sha']
                k=b.forward_kwargs()
                return {n:(v.float() if n=='inputs_embeds' else v) for n,v in k.items()}
            raw=runtime.adapter.prepare_inputs(q.image_path,q.question,None)
            assert raw['image_sha256']==r['image_sha']
            target=torch.tensor([raw_ids],dtype=torch.long,device=raw['input_ids'].device)
            assert target.numel()<=1024 and target.min()>=0
            full=torch.cat((raw['input_ids'],target),-1)
            labels=torch.cat((torch.full_like(raw['input_ids'],-100),target),-1)
            attention=torch.cat((raw['attention_mask'],torch.ones_like(target)),-1)
            emb,att,pos,lab=runtime._expand_multimodal(raw_input_ids=full,attention_mask=attention,
                                                       labels=labels,images=raw['images'])
            assert emb.dtype==torch.float16 and torch.equal(lab[lab!=-100],target[0])
            return {'inputs_embeds':emb.float(),'attention_mask':att,'position_ids':pos,'labels':lab,
                    'use_cache':False,'return_dict':True}
        finally:
            for module in prefill:module.float()

    if plan.get('gradient_preflight') or calibration or formal:
        from p import gradient_preflight
        runtime.model.float()
        gradient_preflight(root, plan, lane, items, runtime, layer, med, mdu, TT4,
                           MedTraceLayerHook, prepare, record, seed_everything)
        if not formal:
            return
        waiting_for = root / f'train{1 - lane}.json'
        while not waiting_for.is_file():
            assert time.time() < plan['deadline'] - 300
            time.sleep(15)
        arms = list(manifest['arms'])
        assert len(arms) == 4
        for arm in arms:
            states[arm] = dict(mdu)
            for item in all_items:
                src = item['source']
                child = load(root / 'weights' / f"{item['owner']}_{arm}.pt")
                assert child['edit'] == src['edit_id'] and child['seed'] == src['seed'] and child['steps'] == 160
                states[arm][src['edit_id']] = child['state']
        runtime.model.half()

    with torch.no_grad():
        routes=[]
        for row in rows:
            if not (row['role'].startswith('holdout') or row['role'] == 'historical_t2l'):
                continue
            seed_everything(row['source']['seed'])
            batch=runtime.build_question_batch(record(row,''))
            assert batch.image_sha256==row['image_sha']
            key=runtime.extract_layer_input_key(batch,module_path=routing_layer,pooling='mean')
            decision=router.route(key.to(runtime.device))
            route=decision_as_json(decision)
            selected=decision.logical_edit_id if decision.activated else None
            assert selected is None or selected in tasks
            row['route']=route
            row['routed']=selected
            counts['routes']+=1
            routes.append({'n':row['n'],'route':route,'own':selected==row['source']['edit_id']})
    assert counts['routes'] == 45 + (n_rows - 120 if formal else 0)
    (root/f'routes{lane}.json').write_text(json.dumps(routes,allow_nan=False))
    runtime.model.float()

    def attach(arm,edit):
        if edit is None or arm=='Base':return None,None
        expert=TT4(tasks[edit]['seed'],8,8)
        expert.load_state_dict(states[arm][edit],strict=True)
        expert.to(layer.weight.device).eval().requires_grad_(False)
        hook=MedTraceLayerHook(layer,expert)
        hook.attach()
        return expert,hook

    def teacher(kwargs, selections):
        labels=kwargs['labels']
        mask=labels[0,1:]!=-100
        tokens=labels[0,1:][mask]
        active=tokens!=eos
        assert active.any()
        pos=int(mask.nonzero()[0])
        result={}
        decoded={n:v for n,v in kwargs.items() if n!='labels'}
        current=[]
        tap=layer.register_forward_pre_hook(lambda m,a:current.append(a[0][0,pos].detach()))
        try:
            for name,arm,edit in selections:
                expert,hook=attach(arm,edit)
                try:
                    if hook:hook.set_teacher_routing(labels)
                    logits=runtime.model(**decoded).logits[0,:-1][mask].float()
                    counts['teacher']+=1
                    assert counts['teacher']<=plan['teacher_cap_per_lane']
                    lp=logits.log_softmax(-1)
                    nll=-lp.gather(1,tokens[:,None]).squeeze(1)
                    values={'nll':float(nll[active].mean()),'ce_with_eos':float(nll.mean()),
                            'first_nll':float(nll[active][0])}
                    if arm=='Base':
                        old=lp[active]
                    else:
                        kl=(old.exp()*(old-lp[active])).sum(-1).mean()
                        assert float(kl)>-2e-6
                        values['kl']=max(0.,float(kl))
                        if expert:
                            b,a=expert.factors()
                            x=current[-1].float()
                            x=x/(x.square().mean().sqrt()+expert.epsilon)
                            values['projection']=float((x@torch.linalg.qr(a.T).Q).norm())
                            values['residual']=float(((x@a.T)@b.T).norm())
                    result[name]=values
                finally:
                    if hook:hook.detach()
            assert all(torch.equal(current[0],v) for v in current[1:])
            return result
        finally:
            tap.remove()

    def generation_inputs(row,rec):
        for m in prefill:m.half()
        try:
            raw=runtime.adapter.prepare_inputs(rec.image_path,rec.question,None)
            assert raw['image_sha256']==row['image_sha']
            raw_ids=raw['input_ids']
            model=runtime.llava_model()
            expanded=model.prepare_inputs_labels_for_multimodal(
                raw_ids,None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
            assert expanded[4].dtype==torch.float16
            return raw,expanded
        finally:
            for m in prefill:m.float()

    def greedy(row,arm,edit,raw,expanded):
        _,hook=attach(arm,edit)
        model=runtime.llava_model()
        original_prepare=model.prepare_inputs_labels_for_multimodal
        called=0
        def reuse(*args,**kw):
            nonlocal called
            if args[0] is not None and torch.equal(args[0],raw['input_ids']):
                assert args[3] is None and args[4] is None
                called+=1
                assert called==1
                return tuple(x.to(device=runtime.device,dtype=torch.float32 if x.is_floating_point() else x.dtype)
                             if isinstance(x,torch.Tensor) else x for x in expanded)
            return original_prepare(*args,**kw)
        model.prepare_inputs_labels_for_multimodal=reuse
        try:
            counts['generation']+=1
            assert counts['generation']<=plan['generation_cap_per_lane']
            with (hook.generation_request() if hook else nullcontext()):
                answer=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
        finally:
            model.prepare_inputs_labels_for_multimodal=original_prepare
            if hook:hook.detach()
        tokens=list(answer.raw_token_ids)
        assert called==1 and tokens
        return {'answer':answer.decoded_text,'tokens':tokens,'at_cap':len(tokens)>=1024,
                'eos':tokens[-1]==eos}

    with torch.no_grad(),(root/f'metrics{lane}.jsonl').open('x',buffering=1) as out:
        for row in rows:
            assert time.time()<plan['deadline']
            seed_everything(row['source']['seed'])
            edit=row['source']['edit_id']
            role=row['role']
            measured_arms = ['MED', 'MDU'] + (arms if formal else [])
            sel = [('Base', 'Base', None)] + [(arm, arm, edit) for arm in measured_arms]
            if 'routed' in row and row['routed'] != edit:
                for arm in measured_arms:
                    sel.append(('bank_' + arm, arm, row['routed']))
            q=record(row,row['reference'])
            ref=teacher(prepare(row,q),sel)
            if role=='native':
                assert abs(ref['MED']['ce_with_eos']-first[edit][0])<=5e-4
            data={k:row[k] for k in ('n','owner','role')}
            data['reference_prefix']=ref
            if row['base']:
                data['base_prefix']=teacher(prepare(row,q,row['base']['raw_token_ids']),sel)
            if formal or role.startswith('holdout'):
                data['route'] = row.get('route')
                raw, expanded = generation_inputs(row, q)
                data['bank_generation'] = {arm: greedy(row, arm, row.get('routed', edit), raw, expanded)
                                           for arm in measured_arms}
                data['reference'] = row['reference']
                if formal:
                    data['question'] = row['question']
            print(json.dumps(data, ensure_ascii=False, allow_nan=False), file=out)
            if row['n'] % 10 == 0:
                print('DONE_EVENTS', row['n'] + 1, n_rows, flush=True)
    if formal:
        assert counts['generation'] == 6 * n_rows
    else:
        assert counts['teacher'] >= 405 and counts['generation'] == 90
    (root / f'done{lane}.json').write_text(json.dumps(
        {'events': n_rows, 'counts': dict(counts), 'training': len(items) * 640 if formal else 0, 'judge': 0}))


if __name__=='__main__':
    main(Path.cwd())
