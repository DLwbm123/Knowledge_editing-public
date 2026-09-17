"""Seven-configuration V3 entrypoint. No launch without exact cohort/budget approval.

Judge is a separate receipt-bound consumer; no in-process semantic evaluation.
Controls are separately registered and are not silently added by this entrypoint.
"""
import argparse,fcntl,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage17_single import setup


def validate(cfg,packet,tasks,extensions):
    if cfg.get('mode')!='STAGE19_APPROVED_SEVEN' or cfg.get('gpu')!='3' or cfg.get('gpu_uuid')!='GPU-43e3d478-7979-ea29-8130-64a467b48a5c':raise ValueError('Only approved seven-config GPU3 campaign')
    if not 1<=len(tasks['tasks'])<=48:raise ValueError('Cohort size outside target bound')
    if digest(tasks['tasks'])!=tasks['freeze_id'] or tasks['freeze_id']!=cfg.get('approved_training_freeze'):raise ValueError('Unapproved training cohort')
    if digest({k:v for k,v in packet.items() if k!='freeze_id'})!=packet['freeze_id'] or packet['freeze_id']!=cfg.get('approved_evaluation_freeze'):raise ValueError('Evaluation changed')
    if digest({k:v for k,v in extensions.items() if k!='freeze_id'})!=extensions['freeze_id'] or extensions['freeze_id']!=cfg.get('approved_extension_freeze'):raise ValueError('Extensions changed')
    if cfg.get('train_seconds')!=cfg.get('approved_budget_seconds') or not 0<cfg.get('approved_budget_seconds',0)<=5*3600:raise ValueError('Core budget outside proposal')
    if [t['canonical_edit_id'] for t in tasks['tasks']]!=[p['candidate_id'] for p in packet['candidate_packages']]:raise ValueError('Task/order mismatch')
    if cfg.get('freeze_id')!=tasks['freeze_id']:raise ValueError('Dispatch freeze mismatch')
    expected={t['canonical_edit_id']:digest(t) for t in tasks['tasks']}
    if cfg.get('approved_task_bindings')!=expected:raise ValueError('Task binding changed')
    if any(p['training']!=t for p,t in zip(packet['candidate_packages'],tasks['tasks'])):raise ValueError('Train/eval packet mismatch')
    ids=list(expected)
    if set(packet['orders'])!={'canonical','18001','18002'} or packet['orders']['canonical']!=ids or any(sorted(v)!=sorted(ids) for v in packet['orders'].values()):raise ValueError('Order contract changed')
    if packet['early_anchors']!=ids[:10] or packet['prefixes']!=sorted({n for n in (1,10,25,50,100,len(ids)) if n<=len(ids)}):raise ValueError('Prefix/anchor contract changed')
    return expected


def run(cfg):
    root=Path(cfg['run']);packet=read(root/'private/DEV_EVALUATION_V3.json');taskpack=read(root/'private/TRAINING_TASKS_V3.json');extensions=read(root/'private/EXTENSIONS_V3.json')
    validate(cfg,packet,taskpack,extensions);setup(cfg)
    source={str(r['qid']):r for r in read(Path(cfg['source_train']))}
    for t in taskpack['tasks']:
        for r in [t['native']]+t['H_fit']+t['G_fit']+t['U_fit']:
            author=source[str(r['source_qid'])]
            if author['question']!=r['question'] or author['answer']!=r['reference'] or Path(author['img_name']).parent.name!=Path(r['image_path']).parent.name:raise ValueError('Author source binding changed')
    import torch
    from dataclasses import replace,asdict
    from m3bench_repro.editors.llava_runtime import EditorRecord,write_json_atomic
    from m3bench_repro.editors.routing import MemoryRouter,balanced_radius
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    from methods.medtrace.hsic import bound_hook
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from scripts.medtrace.stage19_train import train_triplet
    from scripts.medtrace.stage18_cfact import assert_base_off
    from scripts.medtrace.run_selective_write import save
    from scripts.medtrace import stage15
    runtime=load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['cpu_gate'])));runtime.run_root=root/'private/work';runtime.model.eval();assert_base_off(runtime)
    if runtime.generation_config!=cfg['generation_lock'] or next(runtime.model.parameters()).dtype!=torch.float16:raise ValueError('Runtime changed')
    frozen=[(p,p._version,p.data_ptr()) for p in runtime.model.parameters()]
    tasks=taskpack['tasks'];byid={t['canonical_edit_id']:t for t in tasks};packages={p['candidate_id']:p for p in packet['candidate_packages']}
    records={}
    for t in tasks:
        n=t['native'];records[t['canonical_edit_id']]=EditorRecord(t['canonical_edit_id'],n['dataset'],n['question'],n['reference'],t['fit_questions'][0],Path(n['image_path']),n['image_path'],t['order'],'AI_REVIEWED_DEV','NATIVE_ONLY_FIT')
    from scripts.medtrace.stage18_cfact import source_batch
    for t in tasks:
        def stratum(row):
            n=len(source_batch(runtime,records[t['canonical_edit_id']],row).target_token_ids)
            return 0 if n<=4 else 1 if n<=8 else 2 if n<=16 else 3
        if any(stratum(h)!=stratum(g) for h,g in zip(t['H_fit'],t['G_fit'])):raise ValueError('H/G tokenizer length strata differ before training')
    record=next(iter(records.values()));target=runtime.target_lock['balancedit']['targets'][0]
    basefile=read(root/'private/BASE_OUTPUTS.json');extbase=read(root/'private/EXTENSION_BASE_OUTPUTS.json')
    if digest(basefile)!=cfg['approved_base_binding'] or digest(extbase)!=cfg['approved_extension_base_binding']:raise ValueError('Base qualification binding changed')
    baseline={r['query_id']:r['output'] for source in (basefile,extbase) for r in source['records']};cached={};routes={}
    def check():
        stage15.budget(root,cfg)
        if any(p._version!=v or p.data_ptr()!=ptr or p.requires_grad for p,v,ptr in frozen):raise ValueError('Base parameters changed')
    def key(r):
        assert_base_off(runtime)
        with torch.inference_mode():return runtime.extract_layer_input_key(runtime.build_question_batch(r),module_path=target,pooling='mean')
    # Base-only keys are collected before BE temporarily wraps its fixed target.
    for t in tasks:
        r=records[t['canonical_edit_id']];k=key(r);pos=key(replace(r,question=t['fit_questions'][0]));black=runtime.make_black_image(r,root/'private/black');neg=key(replace(r,image_path=black))
        router=MemoryRouter('euclidean');router.add(t['canonical_edit_id'],k,balanced_radius(k,pos,neg,alpha=.2,distance='euclidean'));entry=router.export_state()['entries'][0];entry['label']=[];routes[t['canonical_edit_id']]=entry
    def panel(ids):
        rows={}
        for edit in ids:
            p=packages[edit]
            for r in [p['training']['native']]+p['evaluation']:rows.setdefault(digest([r['image_sha256'],r['question']]),r)
        for link in extensions['text_links']:
            if link['edit'] in ids:rows.setdefault(link['query_id'],extensions['text_queries'][link['query_id']])
        for link in extensions['independent_image_positive']:
            if link['edit'] in ids:
                r=link['source'];rows.setdefault(digest([r['image_sha256'],r['question']]),dict(r,role=link['role']))
        return rows
    for q,row in panel(list(byid)).items():
        check();base,raw,_,_=stage15.base_output(runtime,root,row,record)
        if q not in baseline or base['binding']!=baseline[q]['binding'] or base['raw_token_ids']!=baseline[q]['raw_token_ids']:raise ValueError('Base drift or unqualified query')
        cached[q]=(base,raw,key(replace(record,question=row['question'],target='',image_path=Path(row['image_path']))))
    output_file=root/'private/CAMPAIGN_OUTPUTS.jsonl'
    if output_file.exists():raise ValueError('Existing output requires explicit resume audit')
    import json
    def evaluate(name,states,editor=None):
        expert=None
        if editor is None:
            cp=AsymmetricCPExpert(14336,4096,4).to(runtime.device);expert=LowRankExpert(cp,20260912,rank=4).to(runtime.device).requires_grad_(False);del cp
        count=0
        def one(mode,order_name,inserted,rows):
            nonlocal count
            router=MemoryRouter.from_state(dict(distance='euclidean',entries=[routes[e] for e in inserted]),device=runtime.device)
            for q,row in rows.items():
                check();base,raw,k=cached[q];decision=router.route(k);selected=decision.nearest_logical_edit_id;out=base;forced=None
                if decision.activated or mode=='single':
                    state=states[selected]
                    if editor is None:
                        if state['binding']['task']!=byid[selected] or state['binding']['code']!=cfg['code_commit'] or state['step']!=320:raise ValueError('Checkpoint binding changed')
                        expert.load_state_dict(state['expert']);binding=dict(layer_id=state['binding']['layer_id'],expert_id=selected,W0_id=state['binding']['W0'])
                        with bound_hook(runtime,expert,binding) as hook:forced=stage15.generate(runtime,raw,base['binding'],hook)
                    else:
                        editor.wrapper.clear();editor.wrapper.load_exported_state(state['wrapper'])
                        with editor._activated(selected):forced=stage15.generate(runtime,raw,base['binding'])
                    if decision.activated:out=forced
                result=dict(method=name,mode=mode,order=order_name,prefix=len(inserted),inserted=inserted,query_id=q,source=row,route=asdict(decision),Base=base,R0=out,FORCED_OWN=forced if mode=='single' else None,code=cfg['code_commit'])
                with output_file.open('a') as f:f.write(json.dumps(result)+'\n')
                count+=1
        for edit in byid:one('single','canonical',[edit],panel([edit]))
        for order_name,order in packet['orders'].items():
            for n in packet['prefixes']:one('sequential',order_name,order[:n],panel(order[:n]))
        write_new(root/'public'/(name.replace('@','_')+'_GENERATED.json'),dict(outputs=count,status='GENERATED_NOT_SCORED'))
    # Fixed C => original BE => HSIC C is the dependency order.
    for mode in ('fixed21','HSIC_Top1'):
        lane=root/mode;(lane/'private').mkdir(parents=True,exist_ok=True);(lane/'public').mkdir(exist_ok=True)
        states={m:{} for m in ('C_NO_H','C_EXTRA','C_FACT')}
        for t in tasks:
            check();receipt=train_triplet(runtime,lane,dict(cfg,initialization_root=str(root/'shared_initialization')),t,records[t['canonical_edit_id']],selection_mode=mode)
            for item in receipt['branches']:states[item['branch']][t['canonical_edit_id']]=torch.load(item['path'],map_location='cpu',weights_only=False)
        for name,bank in states.items():evaluate(name+('@L21' if mode=='fixed21' else '@HSIC'),bank)
        del states
        if mode=='fixed21':
            editor=BalanceEditPaperSpecEditor(runtime);base_layer=editor.wrapper.base;bank={}
            try:
                lock=editor.config_lock()
                if lock['steps_per_edit']!=50 or lock['learning_rate']!=.01 or lock['alpha']!=.2:raise ValueError('BE recipe changed')
                for t in tasks:
                    check();editor.reset_editor_state();r=records[t['canonical_edit_id']];receipt=editor.apply_edit(r)
                    if not receipt['finite_losses'] or not receipt['finite_gradients']:raise ValueError('BE execution failed')
                    re=editor.router.export_state()['entries'][0];expected=routes[t['canonical_edit_id']]
                    if not torch.equal(re['key'],expected['key']) or re['radius']!=expected['radius']:raise ValueError('BE/C router mismatch')
                    state=dict(task=t,wrapper=editor.wrapper.export_state(),training=receipt,code=cfg['code_commit']);path=root/'private/BE'/f"e{t['order']:03d}.pt";save(path,state);bank[t['canonical_edit_id']]=torch.load(path,map_location='cpu',weights_only=False)
                editor.reset_editor_state();evaluate('BalancEdit',bank,editor)
            finally:
                editor.reset_editor_state();runtime.replace_module(target,base_layer)
            del bank,editor;assert_base_off(runtime)
    check();write_new(root/'public/CAMPAIGN_GENERATED.json',dict(status='SEVEN_GENERATED_NOT_SCORED',N=len(tasks),configurations=7,controls_run=False,cleanup_permitted=False,seconds=time.time()-cfg['campaign_epoch']))

if __name__=='__main__':
    cfg=read(os.environ['JOB_CONFIG']);root=Path(cfg['run'])
    with (root/'private/worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(cfg)
