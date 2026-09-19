"""Pre-semantic mechanical and cost gate; two canonical-ID-selected cases."""
import sys,os,time,copy,io,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from scripts.medtrace.stage19_fasttrack_budget import read,write,check
from scripts.medtrace.stage17_prepare import digest


def run(cfg):
    from scripts.medtrace.stage17_single import setup
    setup(cfg)
    import torch
    from dataclasses import replace
    from scripts.medtrace.stage19_fasttrack import load,record_for,query_id
    from scripts.medtrace.stage18_cfact import source_batch,teachers_for,state_hash,assert_base_off
    from scripts.medtrace import stage15
    from methods.medtrace import MedTraceLayerHook,AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert,optimizer_for
    from m3bench_repro.editors.llava_runtime import seed_everything
    from training import update
    from regularizer import capture,loss,kernel_diagnostics
    import importlib.util
    spec=importlib.util.spec_from_file_location('original22',ROOT/'reports/medtrace_stage22_20260919/training.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    root=Path(cfg['run']);prior=Path(cfg['common_run']);reg=read(root/'public/PREREGISTRATION.json');stream=read(prior/'private/STREAM.json');assert digest(stream)==cfg['stream_binding']
    rt=load(cfg);frozen=[(p,p._version,p.data_ptr()) for p in rt.model.parameters()];results=[]
    bank=torch.load(Path(cfg['stage20_run'])/'private/BANKS.pt',map_location='cpu',weights_only=False)
    baseline={r['query_id']:r['output'] for r in read(prior/'private/HISTORICAL_BASE.json')['records']}
    for n in reg['mechanical_positions']:
        check(cfg);t=stream['tasks'][n-1];record=record_for(t);assert_base_off(rt)
        raw,_,binding=stage15.prepared(rt,t['native'],record);before=stage15.generate(rt,raw,binding);assert before['raw_token_ids']==baseline[query_id(t['native'])]['raw_token_ids']
        with torch.inference_mode():key=rt.extract_layer_input_key(rt.build_question_batch(record),module_path='model.layers.31.mlp.up_proj',pooling='mean').cpu()
        assert torch.equal(key,bank['routes'][t['canonical_edit_id']]['key'])
        w0=torch.load(Path(cfg['stage21A_run'])/'private/W0'/f'e{n:03d}.pt',map_location='cpu',weights_only=True);assert w0['origin_layer']==30 and w0['task']==digest(t)
        expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4),20260912,rank=4).to(rt.device)
        batches=[source_batch(rt,record,t['native'])]+[rt.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
        assert len(batches)==5 and all(rt.adapter.tokenizer.eos_token_id in b.target_token_ids for b in batches)
        extra=(source_batch(rt,record,t['H_fit'][0]),t['H_fit'][0]);teachers=teachers_for(rt,prior,cfg,t,record)
        def train(kind):
            seed_everything(t['seed']);expert.load_state_dict(w0['expert']);expert.requires_grad_(True);assert state_hash(expert)==w0['W0'];opt=optimizer_for(expert,rt.model)
            hook=MedTraceLayerHook(rt.get_module('model.layers.30.mlp.down_proj'),expert);hook.attach();curve=[];times=[]
            try:
                for step in range(3):
                    check(cfg);torch.cuda.synchronize();start=time.perf_counter()
                    kw=dict(extra=extra,extra_weight=.25,U_weight=.01,diagnostic=step==0)
                    fn=old.update if kind=='original' else update
                    if kind!='original':kw.update(regularize=kind=='on',batches=batches)
                    r=fn(rt,hook,expert,opt,batches[0],batches[step+1],teachers[step%len(teachers)],**kw)
                    torch.cuda.synchronize();times.append(time.perf_counter()-start);curve.append(r)
                state={k:v.detach().cpu().clone() for k,v in expert.state_dict().items()}
                probe=torch.linspace(-1,1,14336,device=rt.device)[None,:]
                delta=expert.residual(probe).detach().cpu()
                if kind=='on':
                    saved=io.BytesIO();torch.save(expert.state_dict(),saved);saved.seek(0);restored=torch.load(saved,weights_only=True);expert.load_state_dict(restored);assert torch.equal(delta,expert.residual(probe).detach().cpu())
                    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state();f,m=capture(rt,hook,batches);assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state());del f,m
                return state,delta,curve,times
            finally:hook.detach();opt.zero_grad(set_to_none=True)
        original,_,_,_=train('original');zero,zero_delta,_,off_times=train('zero');lambda0_exact=all(torch.equal(original[k],zero[k]) for k in original);lambda0_max_difference=max(float((original[k]-zero[k]).abs().max()) for k in original)
        on,on_delta,curve,on_times=train('on');diag=curve[0]['regularization'];assert diag['x_gradient_norm']>0 and diag['y_gradient_norm']>0
        parameter_difference=sum(int((on[k]!=zero[k]).sum()) for k in on);delta_difference=float((on_delta-zero_delta).norm());effective_change_pass=parameter_difference>0 and delta_difference>0
        assert all(p._version==v and p.data_ptr()==ptr and not p.requires_grad and p.grad is None for p,v,ptr in frozen)
        assert_base_off(rt);after=stage15.generate(rt,raw,binding);assert before['raw_token_ids']==after['raw_token_ids']
        results.append(dict(position=n,lambda0_exact=lambda0_exact,lambda0_max_difference=lambda0_max_difference,Base_OFF_exact=True,router_key_exact=True,base_unchanged=True,save_load_exact=True,parameter_elements_changed=parameter_difference,effective_delta_difference_norm=delta_difference,effective_change_pass=effective_change_pass,on_step_seconds=on_times,off_step_seconds=off_times,curve=curve,peak_allocated_bytes=torch.cuda.max_memory_allocated(),teacher_rows=len(teachers)))
        write(root/'private/MECHANICAL_DETAILS.json',dict(results=results));print('CASE_COMPLETE',n,flush=True)
        del teachers,batches,extra,expert;gc.collect();torch.cuda.empty_cache()
    per_step=max(max(r['on_step_seconds']) for r in results);conservative_writer_seconds=320*per_step*1.2
    report=dict(status='PASS' if all(r['effective_change_pass'] and r['lambda0_exact'] for r in results) else 'FAIL_MECHANICAL_GATES',cases=[{k:v for k,v in r.items() if k!='curve'} for r in results],regularizer_components=[dict(position=r['position'],**{k:v for k,v in r['curve'][0]['regularization'].items() if k!='masks'}) for r in results],conservative_writer_seconds=conservative_writer_seconds,DEV19_training_seconds=19*conservative_writer_seconds,full45_training_seconds=45*conservative_writer_seconds,regularizer_every_update=True,precision='backbone float16/writer float32; kernel float64',no_new_semantic_judgments=True)
    write(root/'public/MECHANICAL_GATE.json',report)

if __name__=='__main__':run(read(os.environ['JOB_CONFIG']))
