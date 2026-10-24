"""One preregistered six-forward prefill check, then the finite relative-hinge study."""
import os,json,time,math
from pathlib import Path
import torch
from prefix_cache import PrefixCache

def run(runtime,r,out,partition,counts):
    shared=out.parent;parent=Path(os.environ['PARENT_RUN'])
    def read(p):return json.loads(Path(p).read_text())
    def write(p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
    if partition==1:
        with (shared/'PREFILL_REPAIR_ONCE').open('x'):pass
        template=read(parent/'private/M13_SOURCE_PACKET.json')['original_tasks'][4];bindings=read(parent/'private/EVAL_BINDINGS.json');task=read(shared/'DATA_ADMISSION.json')['tasks'][0]
        row=task['retention'];answers=[]
        for form in ('{}.', 'The answer is {}.', 'It is {}.'):
            answers.extend([form.format(task['retention_correct'].rstrip('.')),form.format(task['retention_wrong'].rstrip('.'))])
        r.old.fp32(runtime)
        backend=dict(flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled(),tf32=torch.backends.cuda.matmul.allow_tf32)
        assert backend==dict(flash=False,efficient=False,math=True,tf32=False)
        runtime.model.half();items=r.s.prepare(runtime,bindings,template,[dict(row,query_id='PREFILL_'+str(i),reference=t) for i,t in enumerate(answers)],True);r.old.fp32(runtime)
        batches=[];tokens=[]
        for expr,value,raw,_,binding in items:
            kwargs,labels,ids=value;prefix=raw['input_ids'][0].tolist();full=runtime._tokenize_prompt(raw['prompt']+' '+expr['reference'])[0].tolist();assert full[:len(prefix)]==prefix
            content=tuple(full[len(prefix):]);assert tuple(ids[:len(content)])==content and tuple(ids[len(content):])==(28705,2)
            labels=labels.clone();positions=torch.where(labels[0]!=-100)[0];labels[0,positions[len(content):]]=-100
            batches.append(r.old.batch_on(runtime,(dict(kwargs,labels=labels),labels,content)));tokens.append(list(content))
        initial=r.p.tr.TT4(template['seed'],8,8).state_dict();expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(r.mm.canonicalize(initial)[0]);hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
        before=counts['forwards'];losses=[];hook.attach()
        try:
            with PrefixCache(runtime,batches),torch.no_grad():
                for i,batch in enumerate(batches):
                    r.c.budget();p=shared/'prefill_attempts'/(str(i)+'.json');assert not p.exists();write(p,dict(attempt=1,time=time.time()))
                    hook.set_teacher_routing(batch.labels);losses.append(float(runtime.compute_loss(batch)))
        finally:hook.detach()
        assert counts['forwards']-before==6
        import owner5_semantic_loss as semantic
        summary=semantic.summary(losses);old=read(shared/'OLD_PROXY_REFERENCE.json');assert tokens==old['content_tokens']
        difference=abs(summary['score']-old['score']);passed=math.isfinite(summary['score']) and difference<=1e-6
        write(shared/'PREFILL_CHECK_COMPLETE.json',dict(status='PASS' if passed else 'FAIL',score=summary['score'],legacy_score=old['score'],absolute_difference=difference,summary=summary,content_tokens=tokens,backend=backend,additional_language_forwards=6,negative_score_preserved=summary['score']<0))
    while not (shared/'PREFILL_CHECK_COMPLETE.json').exists():r.c.budget();time.sleep(2)
    check=read(shared/'PREFILL_CHECK_COMPLETE.json')
    if check['status']!='PASS':
        write(out/'STUDY_COMPLETE.json',dict(status='PREFILL_CHECK_FAILED',counts=counts));return
    from relative_study import run as experiment
    experiment(runtime,r,out,partition,counts)
