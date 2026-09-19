"""Additional mechanical localization only; same 900-second ledger, no judge."""
import sys,os,time,io,gc
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
    from scripts.medtrace.stage18_cfact import source_batch,teachers_for,assert_base_off
    from scripts.medtrace import stage15
    from methods.medtrace import MedTraceLayerHook,AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert,optimizer_for
    from m3bench_repro.editors.llava_runtime import seed_everything
    import training,regularizer,scaled_regularizer
    root=Path(cfg['run']);common=Path(cfg['common_run']);stream=read(common/'private/STREAM.json');rt=load(cfg)
    results=[];strict=cfg.get('deterministic_localization',False)
    if strict:torch.use_deterministic_algorithms(True)
    def clk():torch.cuda.synchronize();return time.perf_counter()
    def vector(xs):return torch.cat([x.detach().double().flatten() for x in xs])
    def diff(a,b):
        a=a.double().flatten();b=b.double().flatten();d=a-b;den=float(a.norm()*b.norm());return dict(maxabs=float(d.abs().max()),L2=float(d.norm()),nonzero=int(torch.count_nonzero(d)),relative=float(d.norm()/b.norm()) if float(b.norm()) else None,cosine=float(a@b/den) if den else None)
    versions=[(p,p._version,p.data_ptr()) for p in rt.model.parameters()]
    for n in (8,1):
        t=stream['tasks'][n-1];record=record_for(t);seed_everything(t['seed']);w0=torch.load(Path(cfg['stage21A_run'])/'private/W0'/f'e{n:03d}.pt',map_location='cpu',weights_only=True)
        ex=LowRankExpert(AsymmetricCPExpert(14336,4096,4),20260912,rank=4).to(rt.device);ex.load_state_dict(w0['expert']);ex.requires_grad_(True);params=list(ex.parameters())
        batches=[source_batch(rt,record,t['native'])]+[rt.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
        extra=(source_batch(rt,record,t['H_fit'][0]),t['H_fit'][0]);teachers=teachers_for(rt,common,cfg,t,record)
        hook=MedTraceLayerHook(rt.get_module(training.LAYER),ex);hook.attach();r=dict(position=n,strict_deterministic_diagnostic=strict,CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),attention_implementation=getattr(rt.model.config,'_attn_implementation',None))
        try:
            # Repeated same forward/backward distinguishes scale drift from GPU VJP noise.
            f,_=regularizer.capture(rt,hook,batches,path='hidden');v,*_=regularizer.loss(f)
            g=[vector(scaled_regularizer.scaled_regularizer_gradients(v,params,retain_graph=True)) for _ in range(3)]
            r['same_graph_same_scale_repeat']=[diff(g[i],g[0]) for i in (1,2)]
            g4=vector(scaled_regularizer.scaled_regularizer_gradients(v,params,scale=4096,retain_graph=True));r['plateau']=diff(g4,g[0]);del f,v,g,g4
            if not strict:
                stamp=clk();regularizer.ACTIVE_CACHE=regularizer.build_prefix_cache(rt,batches);r['cache_build_seconds']=clk()-stamp
                r['cache_contract']=dict(boundary='full L30 block input',items=5,kwargs=[list(x['kwargs']) for x in regularizer.ACTIVE_CACHE],bytes=sum(x['hidden'].numel()*x['hidden'].element_size() for x in regularizer.ACTIVE_CACHE),lifetime='current edit only; cleared before next edit',writer_independent=True)
                parity={};reference=None
                for path in ('full','hidden','cached'):
                    stamp=clk();f,_=regularizer.capture(rt,hook,batches,path=path);v,*_=regularizer.loss(f);g=vector(scaled_regularizer.scaled_regularizer_gradients(v,params));fv=vector(list(f.values()))
                    if reference is None:reference=(fv,g,float(v.detach()))
                    parity[path]=dict(features=diff(fv,reference[0]),gradient=diff(g,reference[1]),loss=float(v.detach()),loss_difference=float(v.detach())-reference[2],seconds=clk()-stamp)
                    del f,v,g,fv
                r['path_parity']=parity
                # Full 20-step comparison; do not reset/erase the first mechanical attempt.
                tracks={}
                for label,enabled,on in [('hidden_ON',False,True),('cached_ON',True,True),('hidden_OFF',False,False),('cached_OFF',True,False)]:
                    cache=regularizer.ACTIVE_CACHE;regularizer.ACTIVE_CACHE=cache if enabled else None
                    seed_everything(t['seed']);ex.load_state_dict(w0['expert']);opt=optimizer_for(ex,rt.model);times=[];snap={}
                    for step in range(20):
                        check(cfg);stamp=clk();diagnostic=step in (0,2,19)
                        training.update(rt,hook,ex,opt,batches[0],batches[1+step%4],teachers[step%len(teachers)],extra,extra_weight=.25,U_weight=.01,diagnostic=diagnostic,regularize=on,batches=batches)
                        times.append(dict(step=step+1,diagnostic=diagnostic,seconds=clk()-stamp))
                        if step in (2,19):snap[str(step+1)]=vector(params).cpu()
                    tracks[label]=dict(times=times,snap=snap);regularizer.ACTIVE_CACHE=cache;opt.zero_grad(set_to_none=True)
                r['update_path_parity']={s:{k:diff(tracks['cached_'+k]['snap'][s],tracks['hidden_'+k]['snap'][s]) for k in ('ON','OFF')} for s in ('3','20')}
                r['times']={k:v['times'] for k,v in tracks.items()};del tracks,reference
            else:
                # Separate deterministic-backend diagnostic, never replacing the frozen runtime gate.
                states=[];losses=[]
                for repeat in range(3):
                    seed_everything(t['seed']);ex.load_state_dict(w0['expert']);opt=optimizer_for(ex,rt.model);curve=[]
                    for step in range(20):
                        check(cfg);out=training.update(rt,hook,ex,opt,batches[0],batches[1+step%4],teachers[step%len(teachers)],extra,extra_weight=.25,U_weight=.01,diagnostic=False,regularize=False);curve.append({k:x['unweighted'] for k,x in out['terms'].items()})
                    states.append(vector(params).cpu());losses.append(curve);opt.zero_grad(set_to_none=True)
                r['strict_OFF_repeats']=[diff(x,states[0]) for x in states[1:]];r['strict_loss_curves_equal']=[x==losses[0] for x in losses[1:]]
        except RuntimeError as exc:
            r['runtime_error']=str(exc);r['status']='DIAGNOSTIC_STOPPED';print(str(exc),flush=True)
        finally:hook.detach();regularizer.ACTIVE_CACHE=None;ex.zero_grad(set_to_none=True)
        assert_base_off(rt);raw,_,b=stage15.prepared(rt,t['native'],record);out=stage15.generate(rt,raw,b);baseline={x['query_id']:x['output'] for x in read(common/'private/HISTORICAL_BASE.json')['records']};r['Base_OFF_exact']=out['raw_token_ids']==baseline[query_id(t['native'])]['raw_token_ids'];r['base_unchanged']=all(p._version==v and p.data_ptr()==ptr for p,v,ptr in versions)
        results.append(r);write(root/'public'/('DETERMINISM_LOCALIZATION.json' if strict else 'PREFIX_CACHE_DIAGNOSTIC.json'),dict(cases=results,code=cfg['code_commit'],frozen_main_runtime_modified=False,new_Judge=0));print('LOCALIZED',n,strict,flush=True)
        del teachers,ex,batches,extra;gc.collect();torch.cuda.empty_cache()
    if not strict:
        # Length profile for all19 from their actual expanded batches; no optimizer work.
        lengths=[]
        for t in stream['tasks'][:19]:
            check(cfg);record=record_for(t);bs=[source_batch(rt,record,t['native'])]+[rt.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
            lengths.append(dict(position=t['order'],valid_tokens=[int(b.attention_mask.sum()) for b in bs]));del bs
        write(root/'public/LENGTH_PROFILE.json',dict(rows=lengths))

if __name__=='__main__':run(read(os.environ['JOB_CONFIG']))
