"""Fixed teacher reconstruction bindings; no new qualification or tuning."""
import math
MODULE='model.layers.6.mlp.down_proj'
AUTH=dict(samples=['E01','E02','E03'],layer=MODULE,forward_max=3200,work_forward_max=3100,fit_forward_max=26,fit_forward_total_max=78,per_generation_forward_max=128,seconds=1800,disk_max=251658240,generation_max=21,judge_payload_max=18,checkpoint=0,retries=0,native_null_gate=.01,trust=.001)
def fit_matches(entry,old):
 return entry['backward_calls']==old['backward_calls'] and entry['fit_forwards']==old['fit_forwards'] and len(entry['steps'])==len(old['steps']) and len(entry['teacher_native_nll'])==len(old['teacher_native_nll'])==2 and all(math.isfinite(x) and abs(x-y)<=1e-5 for x,y in zip(entry['teacher_native_nll'],old['teacher_native_nll'])) and math.isfinite(entry['steps'][-1]['loss_before']) and abs(entry['steps'][-1]['loss_before']-old['steps'][-1]['loss_before'])<=1e-5

def source_matches(new,old):
 return new['status']==old['status']=='COMPLETE' and bool(new['raw_answer'].strip()) and new['raw_answer']==old['raw_answer'] and new['raw_token_ids']==old['raw_token_ids']

def run_binding_checks():
 from copy import deepcopy
 v=dict(backward_calls=14,fit_forwards=15,steps=[dict(loss_before=.009)],teacher_native_nll=[.009,.5]);assert fit_matches(v,deepcopy(v))
 for field in ['backward_calls','fit_forwards']:
  b=deepcopy(v);b[field]+=1;assert not fit_matches(b,v)
 for index in [0,1]:
  b=deepcopy(v);b['teacher_native_nll'][index]+=.001;assert not fit_matches(b,v)
 b=deepcopy(v);b['steps'][-1]['loss_before']=float('nan');assert not fit_matches(b,v)
 g=dict(status='COMPLETE',raw_answer='same',raw_token_ids=[1,2]);assert source_matches(g,dict(g))
 for b in [dict(g,raw_answer='changed'),dict(g,raw_token_ids=[1,3]),dict(g,status='INCOMPLETE')]:assert not source_matches(b,g)
 return ['fixed_fit_steps_dual_native_and_FP32_loss_binding','source_complete_text_and_tokens_exact_no_rejudge']

def run_checks():
 from .ftmq_ops import run_checks as fit_checks
 from .layer6N3_cpu_check import run_checks as layer_checks
 fit_checks();layer_checks();checks=run_binding_checks()
 return dict(status='PASS',checks=['original_fit_exception_restore','explicit_layer6_endpoint_gates_and_legacy_compatibility',*checks])
