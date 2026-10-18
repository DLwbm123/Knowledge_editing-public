"""One fixed pass over 10 training candidates: six forced sequences each, no generation."""
import os,sys,json,time,traceback,subprocess,math
from pathlib import Path
from types import SimpleNamespace
P=Path(os.environ['PARENT_RUN']);OUT=Path(os.environ['OUTPUT_RUN']);ROOT=OUT.parent
sys.path[:0]=[str(OUT/'tools'),str(P/'private/tools'),str(P/'private/source'),str(P/'private/official_llava')]
import torch
import module_run as r
from proxy_audit_math import selfcheck,token_metrics
from scripts.medtrace.run_realmodel_core import load_real_runtime

def read(p):return json.loads(Path(p).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)
START=time.time();counts=dict(forwards=0,vision_forwards=0,updates=0,backwards=0,operator_backwards=0,generations=0);LAST=0.
def budget():
    global LAST
    import shutil
    assert not (ROOT/'STOP').exists() and time.time()<float(os.environ['DEADLINE_EPOCH'])
    assert shutil.disk_usage(ROOT).free>8*1024**3
    assert counts['forwards']<=400060 and counts['updates']<=1060 and counts['backwards']<=10000 and counts['operator_backwards']<=1100 and counts['generations']<=256
    if time.time()-LAST>60:
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<2*1024**3
        LAST=time.time()
r.c.budget=budget

def main():
    with (OUT/'ATTEMPT').open('x') as f:f.write(str(os.getpid()))
    write(OUT/'CPU_MATH_CHECK.json',selfcheck());torch.use_deterministic_algorithms(True)
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=P/'private/cpu_gate'));runtime.model.eval();assert not any(p.requires_grad for p in runtime.model.parameters())
    handle=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
    vision=runtime.llava_model().get_vision_tower();vh=vision.register_forward_pre_hook(lambda *_:counts.__setitem__('vision_forwards',counts['vision_forwards']+1))
    write(OUT/'MODEL_READY.json',dict(time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))
    write(OUT/'PROCESS.json',dict(commands=subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','pid=,args='],text=True),gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True)))
    bindings=read(P/'private/EVAL_BINDINGS.json');template=read(P/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    initial=r.p.tr.TT4(template['seed'],8,8).state_dict();base=r.mm.canonicalize(initial)[0]
    expert=r.p.expert(initial,template['seed'],runtime.device);expert.load_state_dict(base);assert torch.count_nonzero(expert.G1)==0
    hook=r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER),expert)
    rows=read(OUT/'CANDIDATES.json');assert len(rows)==10
    try:
        for entry in rows:
            budget();row=entry['row'];qid=row['query_id'];assert not (OUT/'items'/(qid+'.json')).exists()
            texts=[]
            for form in ('{}.', 'The answer is {}.', 'It is {}.'):
                texts.extend([form.format(row['reference'].rstrip('.')),form.format(entry['wrong_choice'].rstrip('.'))])
            expressions=[dict(row,query_id=qid+'_sem_'+str(i),reference=text) for i,text in enumerate(texts)]
            runtime.model.half();items=r.s.prepare(runtime,bindings,template,expressions,True);r.old.fp32(runtime)
            old=read(entry['prior_base'])['binding'];assert old['runtime']['actual_precision']=='MATH_FP32_FROZEN_FP16_PREFILL'
            prepared=[];content_ids=[];prefix=None
            for expr,value,raw,_,binding in items:
                assert all(binding[k]==old[k] for k in ('question','image_sha256','prompt_ids','generation'))
                if prefix is None:prefix=raw['input_ids'][0].tolist()
                assert raw['input_ids'][0].tolist()==prefix
                kwargs,labels,tokens=value;full=runtime._tokenize_prompt(raw['prompt']+' '+expr['reference'])[0].tolist()
                assert full[:len(prefix)]==prefix
                content=tuple(full[len(prefix):]);assert content and tuple(tokens[:len(content)])==content and tuple(tokens[len(content):])==(28705,2)
                original_positions=torch.where(labels[0]!=-100)[0];assert len(original_positions)==len(tokens)
                labels=labels.clone();labels[0,original_positions[len(content):]]=-100
                assert tuple(labels[labels!=-100].tolist())==content
                batch=r.old.batch_on(runtime,(dict(kwargs,labels=labels),labels,content));prepared.append((batch,binding,content));content_ids.append(content)
            divergences=[]
            for j in (0,2,4):
                a,b=content_ids[j:j+2];same=0
                while same<min(len(a),len(b)) and a[same]==b[same]:same+=1
                assert same<min(len(a),len(b)),'Contrast differs only by stopping; preserve diagnostic failure'
                divergences.append((same,(a[same],b[same])))
            measured=[];hook.attach()
            try:
                for j,(batch,binding,content) in enumerate(prepared):
                    budget();assert counts['forwards']<60
                    attempt=OUT/'attempts'/(qid+'_'+str(j)+'.json');assert not attempt.exists();write(attempt,dict(attempt=1,time=time.time()))
                    hook.set_teacher_routing(batch.labels)
                    with torch.no_grad():
                        output=runtime.model(**batch.forward_kwargs());nll,targets=token_metrics(output.logits,batch.labels)
                        assert tuple(targets.tolist())==content
                        torch.testing.assert_close(nll.mean(),output.loss,rtol=1e-6,atol=1e-6)
                        common,tokens_at_divergence=divergences[j//2];positions=torch.where(batch.labels[0]!=-100)[0]
                        logp=output.logits[0,int(positions[common])-1].float().log_softmax(-1)
                        first=[float(logp[k]) for k in tokens_at_divergence]
                    measured.append(dict(expression_index=j,content_token_ids=list(content),token_nll=nll.cpu().tolist(),legacy_mean_nll=float(output.loss),mean_nll=float(nll.mean()),sum_nll=float(nll.sum()),token_count=len(content),masked_tail=[28705,2],first_divergence=dict(content_index=common,candidate_tokens=list(tokens_at_divergence),both_log_probs=first),binding=binding,preprocessing_seed=binding.get('preprocessing_seed')))
                    write(OUT/'sequences'/(qid+'_'+str(j)+'.json'),measured[-1]);del output,nll,logp
            finally:hook.detach()
            margins_mean=[];margins_sum=[];margins_first=[]
            for j in (0,2,4):
                a,b=measured[j:j+2]
                torch.testing.assert_close(torch.tensor(a['first_divergence']['both_log_probs']),torch.tensor(b['first_divergence']['both_log_probs']),rtol=1e-5,atol=1e-5)
                margins_mean.append(b['legacy_mean_nll']-a['legacy_mean_nll']);margins_sum.append(b['sum_nll']-a['sum_nll']);margins_first.append(a['first_divergence']['both_log_probs'][0]-a['first_divergence']['both_log_probs'][1])
            result=dict(query_id=qid,score_mean=sum(margins_mean)/3,score_sum=sum(margins_sum)/3,score_first_divergence=sum(margins_first)/3,pair_margins_mean=margins_mean,pair_margins_sum=margins_sum,pair_margins_first=margins_first,positive_lengths=[v['token_count'] for v in measured[::2]],negative_lengths=[v['token_count'] for v in measured[1::2]],mask_and_causal_shift_pass=True,shared_context_pass=True,old_Base_input_binding_pass=True,all_sequences=6)
            write(OUT/'items'/(qid+'.json'),result);write(OUT/'PROGRESS.json',dict(last_query=qid,completed=len(list((OUT/'items').glob('*.json'))),counts=counts))
        assert counts['forwards']==60 and counts['updates']==counts['backwards']==counts['generations']==0
        write(OUT/'AUDIT_COMPLETE.json',dict(status='COMPLETE',rows=10,sequences=60,counts=counts,seconds=time.time()-START))
        while not (ROOT/'AUDIT_DECISION.json').exists():
            budget();write(OUT/'RESIDENCY.json',dict(time=time.time(),status='AUDIT_COMPLETE_WAITING_REGISTERED_DECISION'));time.sleep(10)
        decision=read(ROOT/'AUDIT_DECISION.json')
        if decision['status']=='PROCEED_RELATIVE_HINGE_V5':
            vh.remove();counts.pop('vision_forwards')
            from study import run
            run(runtime,r,OUT,int(os.environ['CUDA_VISIBLE_DEVICES']),counts)
            write(OUT/'FINAL_COMPLETE.json',read(OUT/'STUDY_COMPLETE.json'))
        else:write(OUT/'FINAL_COMPLETE.json',dict(status='AUDIT_ONLY_COMPLETE',counts=counts))
    finally:hook.detach();handle.remove()

code=0
try:main()
except BaseException:
    code=1;write(OUT/'FAILURE.json',dict(error=traceback.format_exc(),counts=counts,automatic_retry=False));raise
finally:write(OUT/'EXIT.json',dict(code=code,counts=counts,seconds=time.time()-START))
