"""Freeze Stage21 contracts and CPU-only diagnostics from immutable Stage20."""
import json,sys,math,hashlib,shutil,itertools,os
from pathlib import Path
from collections import Counter
from statistics import mean
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage17_prepare import digest,PROTOCOL
from scripts.medtrace.astra_judge_bundle import read,write_new
REPORT=Path(__file__).resolve().parent
OLD=Path(os.environ['STAGE20_LOCAL_RUN'])


def ranks(v):
    return [1+sum(x<y for x in v)+(sum(x==y for x in v)-1)/2 for y in v]


def corr(a,b):
    a=ranks(a);b=ranks(b);ma=mean(a);mb=mean(b)
    den=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/den if den else None


def run():
    root=REPORT/'private/run';p=root/'private';public=root/'public'
    names=['METHOD_LOCK_V1.json','STREAM_MANIFEST.json','RESULTS_PURE.json','NEW_SOURCE_RESULTS.json','ABLATION_DEV11.json','HSIC_REBUILD_SCORE_AUDIT.json','ROUTING_EXPOSURE_AND_TRANSITIONS.json','TRAINING_ACCEPTANCE.json']
    evidence={name:read(OLD/'public'/name) for name in names}
    assert evidence['STREAM_MANIFEST.json']['N']==45
    formal={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (OLD/'public').iterdir() if x.is_file()}
    write_new(p/'STAGE20_FORMAL_BASELINE.json',dict(files=formal,source_directory=str(OLD/'public'),reason='Explicit immutable Stage20 validation; small formal files only'))
    stream=read(OLD/'private/STREAM.json');binding=digest(stream)
    for name in ['STREAM.json','FRESH_BASE_OUTPUTS.json','QUALIFIED_SCORE_CACHE.json']:
        shutil.copy2(OLD/'private'/name,p/name)
    for src,dst in [('OUTPUTS.jsonl','STAGE20_OUTPUTS.jsonl'),('ablation/private/OUTPUTS.jsonl','STAGE20_NOH11_OUTPUTS.jsonl')]:shutil.copy2(OLD/'private'/src,p/dst)
    auth=dict(stage=21,approved=True,gpu_seconds_limit=12600,new_judgment_limit=800,physical_gpu=0,gpu_uuid='GPU-ccc69848-6553-ad18-65b4-09227b59cfa0',previous_stage_balance_inherited=False)
    method=dict(stage=21,inherited=evidence['METHOD_LOCK_V1.json'],stream_binding=binding,runtime=read(OLD/'private/DISPATCH_train.json')['runtime_lock'],A=dict(name='NO_H_HSIC',only_change='H slot absent, H weight 0',reuse_complete_NO_H11=True,HSIC='Fresh native-only five observations, unchanged rule, selected layer recorded for all45'),B=dict(status='WAITING_LAYER_CLARIFICATION',requested_writer='model.layers.21.mlp.down_proj',actual_frozen_BE_writer='model.layers.31.mlp.up_proj',router_feature='model.layers.31.mlp.up_proj input mean',issue='L21 is not a same-layer BE control; no B GPU until resolution'),initialization='Exact Stage20 CP W0 converted on CPU to identical FP32 R4 tensor state, checked against each original continuation W0 hash; no CP retraining',judging=dict(protocol=PROTOCOL,model='gpt-6-astra',reasoning='high',semantic_retries=0,exact_reuse='Exact image/question/reference/output under same protocol only'))
    prereg=dict(stage=21,title='MECHANISM CLOSURE',source_public_commit='41d5bc0ea6b7a702e28ad7acc5920c9aa35c0892',stream_binding=binding,registered_prefixes=[11,19,32,45],N=45,priority=['NO_H_HSIC','B after layer clarification','mechanism closeout'],evaluation_banks={str(n):dict(native=n,old_core=26,new_panel=38 if n==45 else 0,positive_image=4 if n==45 else 0) for n in [11,19,32,45]},missing_early_new_panel='NA: exact Stage20 prefix banks, not zero',statistics=dict(Wilson='Descriptive marginal QA-binomial intervals only; same-source dependence prevents independent-sample interpretation',source_macro=True,leave_one_source_out=True,p_values=False,old_H_sources=2),stopping=dict(GPU_process_seconds=12600,new_judgments=800,budget_includes='loads, selection, failures, restoration, training, generation',no_score_based_stopping=True),checkpoint_consumers=['all saved prefix and insertion outputs','save/load OFF parity','resource profiling','registered Stage21 comparison and bounded follow-on design'],EXTRA='EXTRA_FULL11_UNSUPPORTED',future_authorization=dict(shared_across_all_later_stages=True,GPU_seconds=28800,new_judgments=2000,preregister_before_execution=True,no_automatic_100_146=True),Stage20_immutable=True)
    for name,val in [('RUN_AUTHORIZATION.json',auth),('METHOD_LOCK_STAGE21.json',method),('PREREGISTRATION.json',prereg)]:
        write_new(REPORT/name,val);write_new(public/name,val)
    write_new(public/'GPU_BUDGET_LEDGER.json',dict(limit_seconds=12600,sessions=[]))
    write_new(public/'BUDGET_LEDGER.json',dict(limit=800,new_judgment_items_dispatched=0,exact_reuse_count=0,judgment_batches=[],actual_cost=None,cost_status='Provider actual billing unavailable; not zero'))
    write_new(REPORT/'FOLLOW_ON_BUDGET_LEDGER.json',dict(scope='All stages after Stage21 combined',gpu_limit_seconds=28800,new_judgment_limit=2000,gpu_sessions=[],judgment_batches=[],gpu_seconds_used=0,new_judgments_used=0,authorization='Explicit user reply: later stages together 8 GPU hours / 2000 items',per_stage_reset_allowed=False))
    audit=evidence['HSIC_REBUILD_SCORE_AUDIT.json'];rows=[];hist=Counter();hist2=Counter();curves=[]
    for r in audit['rows']:
        scores={int(k):v for k,v in r['scores'].items()};ranked=sorted(scores,key=lambda k:(-scores[k],k));layers=sorted(scores);v=[scores[k] for k in layers];curves.append(v);a,b=ranked[:2];hist[a]+=1;hist2[b]+=1
        assert r['layer_id']==a and len(r['diagnostics']['rows'])==len(scores)
        rows.append(dict(position=r['position'],selected_layer=a,second_layer=b,margin=scores[a]-scores[b],ratio=scores[a]/scores[b] if scores[b] else None,depth_spearman=corr(layers,v),adjacent_non_decreasing=sum(y>=x for x,y in zip(v,v[1:]))/(len(v)-1),invalid=r['invalid'],score_components=r['diagnostics']['rows']))
    stability=[corr(a,b) for a,b in itertools.combinations(curves,2)];valid=[x for x in stability if x is not None]
    diag=dict(edits=len(rows),top1_histogram=dict(hist),top2_histogram=dict(hist2),selection_entropy_bits=-sum(c/len(rows)*math.log2(c/len(rows)) for c in hist.values()),rank_stability=dict(pair_count=len(stability),mean_spearman=mean(valid),min_spearman=min(valid),max_spearman=max(valid)),mean_score_by_depth={str(k+1):mean(v[k] for v in curves) for k in range(30)},rows=rows,input_specific_layer_diversity=len(hist)>1,claim='CURRENT HSIC DYNAMIC-LAYER CLAIM NOT SUPPORTED' if len(hist)==1 else 'Diversity observed; mechanism not established',new_GPU_training=False,observations='Five correlated native-only wrappers, Base OFF, no H/G/eval')
    write_new(REPORT/'HSIC_DEPTH_BIAS_DIAGNOSTIC.json',diag)
    (REPORT/'HSIC_DEPTH_BIAS_DIAGNOSTIC.md').write_text(f'# HSIC depth diagnostic\n\n{diag["claim"]}\n\n45/45 select L30; selection entropy={diag["selection_entropy_bits"]:.3f} bits. Pairwise layer-score rank Spearman mean={mean(valid):.6f}, range={min(valid):.6f}–{max(valid):.6f}. Scores/components, margins, ratios, depth correlations and adjacent monotonicity are in the JSON. This is an audit of saved Base-only scores, not selector optimization or retraining.\n')
    cfg=read(OLD/'private/DISPATCH_train.json')
    for k in ['gpu_deadline_epoch','campaign_epoch','train_seconds','base_queue_binding']:cfg.pop(k,None)
    cfg.update(stage=21,run='/root/rivermind-data/job-521/run',stage20_run='/root/rivermind-data/job-520/run',arm='NO_H_HSIC',phase='NO_H_HSIC',python='/root/rivermind-data/job-521/env/bin/python',entrypoint='/root/rivermind-data/job-521/entry.py',worker='/root/rivermind-data/job-521/code/reports/medtrace_stage21_20260918/runtime.py',authorization_binding=digest(auth),approved_predecessor_commits=['0fab76c4700c7730be106499eff2e3846c3bb592','d3a377d1caf947742ec49a429797adf1e5d09df7'])
    write_new(p/'CONFIG_TEMPLATE.json',cfg)
    write_new(REPORT/'ARTIFACT_INVENTORY.json',dict(Stage20=dict(main_FACT=45,main_BE=45,CP_W0=45,NO_H_complete=11,EXTRA_incomplete=1,original_outputs_and_score_bindings=True),Stage21=dict(CPU_W0_reconstruction_exact=45,additional_NO_H_writers=34,B_pending=True),storage=dict(observed_free_bytes=13759184896,reserve_bytes=8*1024**3,planned_new_bytes_upper_bound=1024**3,cleanup_needed=False),retention='Keep active resume state, final banks and bindings for registered consumers; Stage20 protected'))
    write_new(REPORT/'RUN_MANIFEST.json',dict(stage=21,status='PREPARING_NO_H; B_LAYER_CLARIFICATION_PENDING',local_run=str(root),remote_run=cfg['run'],remote_code='/root/rivermind-data/job-521/code',ssh_host='root@hb01-ssh.gpuhome.cc',ssh_port=30177,ssh_socket='/tmp/job-519ft/socket',arm='NO_H_HSIC',follow_on_ledger='FOLLOW_ON_BUDGET_LEDGER.json',Stage20_protected=True))


if __name__=='__main__':run()
