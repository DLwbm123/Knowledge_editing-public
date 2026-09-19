"""Bounded numeric acceptance. No semantic judging, selection, or formal bank."""
import sys,os,time,io,gc,random,platform,hashlib,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from scripts.medtrace.stage19_fasttrack_budget import read,write,check
from scripts.medtrace.stage17_prepare import digest


def run(cfg):
    from scripts.medtrace.stage17_single import setup
    setup(cfg)
    import torch,numpy as np
    if cfg.get("deterministic_localization",False):torch.use_deterministic_algorithms(True)
    from dataclasses import replace
    from scripts.medtrace.stage19_fasttrack import load,record_for,query_id
    from scripts.medtrace.stage18_cfact import source_batch,teachers_for,state_hash,assert_base_off
    from scripts.medtrace import stage15
    from methods.medtrace import MedTraceLayerHook,AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert,optimizer_for
    from m3bench_repro.editors.llava_runtime import seed_everything
    import training,regularizer,scaled_regularizer
    import importlib.util
    spec=importlib.util.spec_from_file_location('original22',ROOT/'reports/medtrace_stage22_20260919/training.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    root=Path(cfg['run']);prior=Path(cfg['common_run']);reg=read(root/'public/PREREGISTRATION.json');stream=read(prior/'private/STREAM.json');assert digest(stream)==cfg['stream_binding']
    def clock():torch.cuda.synchronize();return time.perf_counter()
    start=time.perf_counter();rt=load(cfg);load_seconds=clock()-start
    frozen=[(p,p._version,p.data_ptr()) for p in rt.model.parameters()]
    bank=torch.load(Path(cfg['stage20_run'])/'private/BANKS.pt',map_location='cpu',weights_only=False)
    baseline={r['query_id']:r['output'] for r in read(prior/'private/HISTORICAL_BASE.json')['records']}
    def stats(x):
        x=x.detach();d=x.double();return dict(dtype=str(x.dtype),L2=float(d.norm()),maxabs=float(d.abs().max()),nonzero=int(torch.count_nonzero(x)),elements=x.numel(),finite=bool(torch.isfinite(x).all()))
    def vec(xs):return torch.cat([x.detach().double().flatten() for x in xs])
    def comparison(a,b):
        a=a.double().flatten();b=b.double().flatten();den=float(a.norm()*b.norm());return dict(difference=stats(a-b),relative_difference=float((a-b).norm()/b.norm()) if float(b.norm()) else None,cosine=float(a@b/den) if den else None)
    def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
    def rng_id(r):
        b=io.BytesIO();torch.save(r,b);return hashlib.sha256(b.getvalue()).hexdigest()
    imports={m.__name__:dict(file=m.__file__,sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()) for m in (training,regularizer,scaled_regularizer,old)}
    runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,model_class=str(type(rt.model)),backbone_class=str(type(rt.model.model)),imports=imports,model_eval=not rt.model.training,training_dropout_modules=[n for n,m in rt.model.named_modules() if isinstance(m,torch.nn.Dropout) and m.training],deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,stream=int(torch.cuda.current_stream().cuda_stream),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    (root/'private/BACKBONE_FORWARD_SOURCE.txt').write_text(inspect.getsource(type(rt.model.model).forward))
    results=[];numerics=[];repeats=[];costs=[];private=[]
    for n in reg['mechanical_positions']:
        check(cfg);case_start=clock();t=stream['tasks'][n-1];record=record_for(t);assert_base_off(rt)
        raw,_,binding=stage15.prepared(rt,t['native'],record);stamp=clock();before=stage15.generate(rt,raw,binding);generation_seconds=clock()-stamp
        assert before['raw_token_ids']==baseline[query_id(t['native'])]['raw_token_ids']
        with torch.inference_mode():key=rt.extract_layer_input_key(rt.build_question_batch(record),module_path='model.layers.31.mlp.up_proj',pooling='mean').cpu()
        assert torch.equal(key,bank['routes'][t['canonical_edit_id']]['key'])
        w0=torch.load(Path(cfg['stage21A_run'])/'private/W0'/f'e{n:03d}.pt',map_location='cpu',weights_only=True);assert w0['origin_layer']==30 and w0['task']==digest(t)
        expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4),20260912,rank=4).to(rt.device)
        batches=[source_batch(rt,record,t['native'])]+[rt.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
        assert len(batches)==5 and all(rt.adapter.tokenizer.eos_token_id in b.target_token_ids for b in batches)
        extra=(source_batch(rt,record,t['H_fit'][0]),t['H_fit'][0]);stamp=clock();teachers=teachers_for(rt,prior,cfg,t,record);teacher_seconds=clock()-stamp
        setup_seconds=clock()-case_start-generation_seconds
        seed_everything(t['seed']);expert.load_state_dict(w0['expert']);expert.requires_grad_(True);params=list(expert.parameters());hook=MedTraceLayerHook(rt.get_module(training.LAYER),expert);hook.attach();boundary=[]
        try:
            f,m=regularizer.capture(rt,hook,batches,diagnostic=True,boundaries=boundary);r,dx,dy=regularizer.loss(f)
            gx=vec(torch.autograd.grad(.001*dx,params,retain_graph=True));gy=vec(torch.autograd.grad(-.001*dy,params,retain_graph=True))
            gradients={};layers={}
            targets=[f['patch'],f['final']]+[o[k] for o in boundary for k in ('patch_tokens','final_tokens')]+params
            for scale in reg['numeric_scales']:
                check(cfg);gs=torch.autograd.grad(r*scale,targets,retain_graph=True)
                gradients[scale]=vec(gs[-len(params):])/scale
                layers[str(scale)]=dict(pooled=[dict(raw=stats(g),unscaled=stats(g.double()/scale)) for g in gs[:2]],token_activations=[dict(raw=stats(g),unscaled=stats(g.double()/scale)) for g in gs[2:-len(params)]],writer=[dict(raw=stats(g),unscaled=stats(g.double()/scale)) for g in gs[-len(params):]])
            zl=f['patch'].detach().double().requires_grad_();fl=f['final'].detach().double().requires_grad_();reference=.001*regularizer.dependence(f['input'].double(),zl)-.001*regularizer.dependence(zl,fl)
            refg=torch.autograd.grad(reference,(zl,fl));plateau=comparison(gradients[1048576],gradients[16777216]);stable=plateau['relative_difference'] is not None and plateau['relative_difference']<=.01 and plateauau_cos(plateau)
            numeric=dict(position=n,D_x=float(dx.detach()),D_y=float(dy.detach()),reg=float(r.detach()),gx=stats(gx),gy=stats(gy),gx_plus_gy=stats(gx+gy),component_sum_vs_true=comparison(gx+gy,gradients[1]),scaled_recovery_vs_component_sum=comparison(gradients[16777216],gx+gy),combined={str(s):stats(g) for s,g in gradients.items()},boundaries=layers,pooled_FP64_reference=[stats(g) for g in refg],reference_scope='partial derivatives at identical rounded pooled values; not a full CUDA Jacobian reference',scale_plateau=plateau,scale_stable=stable,kernels=regularizer.kernel_diagnostics(f))
            full_values={k:v.detach().clone() for k,v in f.items()};hidden_grad=gradients[16777216].clone();actual=torch.cat([o['writer_inputs'][0] for o in boundary]).detach().float()
            private.append(dict(position=n,masks=m));del targets,gs,f,r,dx,dy,boundary,refg,zl,fl,reference
            parity={}
            for path in ('full','hidden'):
                stamp=clock();pf,_=regularizer.capture(rt,hook,batches,path=path);pr,*_=regularizer.loss(pf);pg=vec(scaled_regularizer.scaled_regularizer_gradients(pr,params,scale=16777216.,retain_graph=False));duration=clock()-stamp
                parity[path]=dict(features_exact=all(torch.equal(pf[k].detach(),full_values[k]) for k in pf),gradient=comparison(pg,hidden_grad),reg=float(pr.detach()),seconds=duration)
                del pf,pr,pg
            numeric['hidden_full_parity']=parity;numerics.append(numeric)
        finally:hook.detach();expert.zero_grad(set_to_none=True)
        write(root/'public/NUMERIC_PATH_DIAGNOSTIC.json',dict(runtime=runtime,cases=numerics))
        # Multiple fixed probes plus all current five-forward writer activations.
        grid=torch.arange(14336,device=rt.device,dtype=torch.float32);fixed=torch.stack([torch.sin(grid*(i+1)/1000) for i in range(4)])
        probes=torch.cat([fixed,actual]);del actual
        def train(kind):
            seed_everything(t['seed']);expert.load_state_dict(w0['expert']);expert.requires_grad_(True);assert state_hash(expert)==w0['W0'];opt=optimizer_for(expert,rt.model)
            initial_rng=rng();torch.save(initial_rng,root/'private'/f'RNG_{n}_{kind}.pt')
            hook=MedTraceLayerHook(rt.get_module(training.LAYER),expert);module=rt.get_module(training.LAYER);before_hooks=len(module._forward_hooks);hook.attach();curve=[];times=[];snaps={};stages={}
            try:
                for step in range(20):
                    check(cfg);diagnostic=step in (0,2,19);stage_tensors={}
                    def observe(label,ps):stage_tensors[label]=dict(parameters=vec(ps).cpu(),gradients=vec([p.grad for p in ps]).cpu())
                    stamp=clock();kw=dict(extra=extra,extra_weight=.25,U_weight=.01,diagnostic=diagnostic)
                    fn=old.update if kind.startswith('original') else training.update
                    if not kind.startswith('original'):kw.update(regularize=kind=='on',batches=batches,observer=observe if diagnostic else None)
                    out=fn(rt,hook,expert,opt,batches[0],batches[1+step%4],teachers[step%len(teachers)],**kw)
                    times.append(dict(step=step+1,diagnostic=diagnostic,seconds=clock()-stamp));curve.append(out)
                    if diagnostic and stage_tensors:stages[str(step+1)]=stage_tensors
                    if step in (2,19):
                        with torch.no_grad():delta=expert.residual(probes).cpu()
                        snaps[str(step+1)]=dict(parameters=vec(params).cpu(),delta=delta)
                b=io.BytesIO();torch.save(expert.state_dict(),b);b.seek(0);expert.load_state_dict(torch.load(b,weights_only=True))
                with torch.no_grad():roundtrip=torch.equal(expert.residual(probes).cpu(),snaps['20']['delta'])
                return dict(snaps=snaps,curve=curve,times=times,stages=stages,rng=rng_id(initial_rng),save_load_exact=roundtrip)
            finally:
                hook.detach();opt.zero_grad(set_to_none=True)
                if len(module._forward_hooks)!=before_hooks:raise RuntimeError('Writer hook leak')
        runs={}
        for kind in ('original1','zero1','original2','zero2','on'):
            runs[kind]=train(kind);print('MECHANICAL_RUN',n,kind,flush=True)
        rep={};effect={}
        for step in ('3','20'):
            rep[step]={}
            for a,b in (('original1','zero1'),('original2','zero2'),('zero1','zero2'),('original1','original2')):
                rep[step][a+'_vs_'+b]={k:comparison(runs[a]['snaps'][step][k],runs[b]['snaps'][step][k]) for k in ('parameters','delta')}
            effect[step]={k:comparison(runs['on']['snaps'][step][k],runs['zero1']['snaps'][step][k]) for k in ('parameters','delta')}
        noise=max(x['delta']['difference']['L2'] for x in rep['20'].values());signal=effect['20']['delta']['difference']['L2'];exact=all(x['parameters']['difference']['nonzero']==0 for s in rep.values() for x in s.values())
        stages={}
        for step,row in runs['on']['stages'].items():
            stages[step]={label:dict(parameters=stats(x['parameters']),gradients=stats(x['gradients']),vs_OFF={k:comparison(x[k],runs['zero1']['stages'][step][label][k]) for k in x}) for label,x in row.items()}
        increments=[x['regularization'] for x in runs['on']['curve'] if x['regularization'] and 'accumulated_gradient_increment_norm' in x['regularization']]
        accumulation=all(x['accumulated_gradient_increment_norm']>0 for x in increments)
        assert all(p._version==v and p.data_ptr()==ptr and not p.requires_grad and p.grad is None for p,v,ptr in frozen)
        assert_base_off(rt);after=stage15.generate(rt,raw,binding)
        with torch.inference_mode():afterkey=rt.extract_layer_input_key(rt.build_question_batch(record),module_path='model.layers.31.mlp.up_proj',pooling='mean').cpu()
        checks=dict(Base_OFF_exact=before['raw_token_ids']==after['raw_token_ids'],router_exact=torch.equal(key,afterkey),base_unchanged=True,save_load_exact=all(x['save_load_exact'] for x in runs.values()),hook_cleanup=True)
        recovered=numeric['combined']['16777216']['finite'] and numeric['combined']['16777216']['nonzero']>0
        status='PASS' if recovered and stable and accumulation and exact and signal>noise and all(checks.values()) else ('GRADIENT_RECOVERED_BUT_UPDATE_NEGLIGIBLE' if recovered and stable and accumulation and exact and signal<=noise else 'FAIL_MECHANICAL_GATES')
        results.append(dict(position=n,status=status,checks=checks,repeatability_exact=exact,gradient_recovered=recovered,scale_stable=stable,accumulation_nonzero=accumulation,effects=effect,OFF_noise_L2=noise,ON_signal_L2=signal,update_stages=stages,accumulation=[{k:v for k,v in x.items() if k not in ("masks",)} for x in increments],actual_activation_probe_rows=len(probes)-4,fixed_probes=4))
        repeats.append(dict(position=n,comparisons=rep,initial_RNG={k:v['rng'] for k,v in runs.items()},historical_transient_OFF_failure='Preserved; old run did not persist maxdiff; current repeats do not erase it'))
        costs.append(dict(position=n,setup_seconds=setup_seconds,teacher_seconds=teacher_seconds,base_generation_seconds=generation_seconds,on=runs['on']['times'],off=runs['zero1']['times'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),valid_tokens=[int(b.attention_mask.sum()) for b in batches],target_tokens=[len(b.target_token_ids) for b in batches]))
        torch.save(dict(position=n,runs=runs),root/'private'/f'TRACES_{n}.pt')
        write(root/'private/MASK_DETAILS.json',private);write(root/'public/REPEATABILITY.json',dict(runtime=runtime,cases=repeats));write(root/'public/MECHANICAL_GATE_STAGE24R.json',dict(status='PASS' if all(r['status']=='PASS' for r in results) else 'NOT_PASS',cases=results,formal_training=False,new_Judge=0));write(root/'public/COST_PROFILE_STAGE24R.json',dict(load_seconds=load_seconds,cases=costs,estimate_status='CPU_CLOSEOUT_REQUIRED',prefix_cache='NOT_IMPLEMENTED_PENDING_HIDDEN_PATH_COST',includes_no_formal_320_training=True))
        print('CASE_COMPLETE',n,status,flush=True)
        del runs,probes,teachers,batches,extra,expert;gc.collect();torch.cuda.empty_cache()


def plateauau_cos(d):return d['cosine'] is not None and d['cosine']>=.999

if __name__=='__main__':run(read(os.environ['JOB_CONFIG']))
