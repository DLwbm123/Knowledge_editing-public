"""Read-only scientific-stop closeout; no generation, Judge or queue mutation."""
from collections import Counter
import json,os,sqlite3,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/source')
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import astra_report as ar
from panels import protection_ids
from logic import transitions
R,c,r=ar.RUN,ar.common,ar.r

def binding_key(query,judge,output,epoch):
    assert all(query[k]==judge[k] for k in ('question','reference','image_sha256'))
    full=dict(query_id=query['query_id'],question=query['question'],reference=query['reference'],image_sha256=query['image_sha256'],
        image_path=judge['image_path'],prompt_ids=judge['prompt_ids'],attention_mask=judge['attention_mask'],runtime=judge['runtime'],
        generation=judge['generation'],output=output,judge={k:epoch[k] for k in ('model','reasoning_effort','protocol','prompt')})
    return c.digest(full),full

def main():
    review=c.read(R/'private/STOP_REVIEWED.json');assert review['reason']=='SCIENTIFIC_MARGIN_ADMISSION_REJECTED'
    ledger=c.read(R/'RESOURCE_LEDGER.json');assert not any(not x.get('ended_epoch') for x in ledger['gpu_sessions'])
    assert ledger['Judge_attempts']==0 and not (R/'private/GENERATION_COMPLETE.json').exists()
    marker=R/'private/REPORT_COMPLETE.json'
    if marker.exists():assert c.read(marker)['status']=='PARTIAL_NOT_COMPLETE';print('PARTIAL_NOT_COMPLETE');return
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    payloads={x['key']:x for x in db.execute('SELECT * FROM payload')}
    scores={k:x['correct'] for k,x in payloads.items()};statuses={k:x['status'] for k,x in payloads.items()}
    epoch=c.read(R/'private/judge_purew146/EPOCH_MANIFEST.json');tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ev=c.read(R/'private/EVAL_LEDGER.json')
    records=[];lookup={};coverage={a:Counter() for a in ('A','B','C','D')}
    files=[('A',Path(x['path']),x['output_binding']) for x in db.execute("SELECT * FROM consumer WHERE method='A'")]
    files.extend(('B',p,None) for p in (R/'private/outputs/B').glob('*/*/*/*.json') if p.name!='COMPLETE.json' and not p.name.endswith('.STARTED.json'))
    for arm,path,locked in files:
        d=c.read(path)
        if locked:assert c.digest(d)==locked
        bb=d['binding'];phase=bb['phase'];query=ev['queries'][bb['input']['query_id']];assert bb['input']==query
        key,full=binding_key(query,bb['judge_input'],d['R0'],epoch)
        if key in payloads:
            assert json.loads(payloads[key]['binding'])==full
            status=statuses[key]
        else:
            scores[key]=None;status='UNATTEMPTED'
        coverage[arm][status]+=1
        prefix=phase['prefix'];mode=phase['mode'];owner=phase['owner']
        active={(t['native']['image_sha256'],t['native']['question']) for t in tasks[:prefix]} if mode=='bank_R0' else set()
        records.append((dict(binding=dict(phase=dict(arm=arm,node=0,prefix=prefix),input=query,mode=mode,owner_order=owner),
            active_target=(query['image_sha256'],query['question']) in active,effective_expert=None),key))
        lookup[arm,mode,prefix,owner,query['query_id']]=key
    assert sum(coverage['A'].values())==sum(coverage['B'].values())==161
    panels=[];coefs={};groups={}
    for arm in ('A','B'):
        for prefix in (1,3):
            for task in ('T0','T1G','T2G','T1L','T2L'):
                metric,co,g=r.panel(records,scores,'RETRO146',arm,0,'bank_R0',task,tasks[:prefix],prefix)
                metric.pop('route_ON',None);metric['unknown']=metric.pop('missing')
                keys=r.combine(list(co.values()));metric['permanent_missing_payloads']=sum(statuses.get(k)=='MISSING' for k in keys)
                metric['unattempted_payloads']=sum(k not in payloads for k in keys)
                n=metric['observations'];metric['micro_bounds']=[100*metric['known_correct']/n,100*(metric['known_correct']+metric['unknown'])/n] if n else None
                panels.append(metric);coefs[arm,prefix,task]=co;groups.update(g)
    ca,cb=coefs['B',3,'T2G'],coefs['A',3,'T2G'];assert set(ca)==set(cb)
    delta={e:r.combine([ca[e],{k:-v for k,v in cb[e].items()}]) for e in ca}
    bounds=r.score_bounds(r.combine([{k:v/len(delta) for k,v in x.items()} for x in delta.values()]),scores)
    contrast=dict(comparison='B-A',delta_macro_bounds_pp=[100*x for x in bounds],negative_edit_lower_bounds=sum(r.score_bounds(x,scores)[0]<-1e-12 for x in delta.values()),**ar.bootstrap(delta,scores,groups))
    protections=[];endpoints=[]
    for arm in ('A','B'):
        for task in ('T1L','T2L'):
            ids=protection_ids(tasks,ev,task);keys=[lookup[arm,'protect100_R0',100,0,q] for q in ids]
            correct=sum(scores[k]==1 for k in keys);unknown=sum(scores[k] is None for k in keys)
            protections.append(dict(arm=arm,task=task,observations=len(keys),known_correct=correct,unknown=unknown,
                permanent_missing=sum(statuses.get(k)=='MISSING' for k in keys),unattempted=sum(k not in payloads for k in keys),
                Base_correct_to_wrong_bounds=[len(keys)-correct-unknown,len(keys)-correct],zero_damage_gate=correct==len(keys)))
            if arm=='B':endpoints.append(dict(task=task,comparison='A-to-B',**transitions([(lookup['A','protect100_R0',100,0,q],lookup['B','protect100_R0',100,0,q]) for q in ids],scores)))
        for task in ('T0','T2G'):
            pairs=[]
            for t in tasks[:3]:
                ids=[t['edit_id']] if task=='T0' else [q for e in t['events'] if e['task']=='T2G' for q in e['all_probe_query_ids'] if not ev['Base_correctness'][q]]
                pairs.extend((lookup[arm,'insertion_R0',t['order'],t['order'],q],lookup[arm,'bank_R0',3,0,q]) for q in dict.fromkeys(ids))
            endpoints.append(dict(arm=arm,task=task,comparison='actual-insertion-to-prefix3',**transitions(pairs,scores)))
    diagnostics={}
    for arm in ('B','C','D'):
        rows=[]
        for owner in (1,2,3):
            es=sorted([c.read(p) for p in (R/'private/steps'/arm/str(owner)).glob('*.json')],key=lambda e:e['slot'])
            if not es:continue
            flags=Counter(z for e in es if not e['accepted'] for z in set([e['rejection_reason']] if e.get('rejection_reason') else [x for a in e['candidates'] for x in a['reasons']]))
            source=R/'public'/('SOURCE_'+arm+'_'+str(owner)+'.json')
            source=c.read(source) if source.exists() else None
            fixed_cycle=len(es)>=4 and all(e['zero_native'] for e in es[-4:]) and len({(e['slot']-1)%4 for e in es[-4:]})==4
            ce=source['final']['weighted_all_FIT_CE'] if source else sum(e['source_CE'] for e in es[-4:])/4 if fixed_cycle else None
            rows.append(dict(source_position=owner,slots=len(es),accepted=sum(e['accepted'] for e in es),zero_fraction=sum(e['zero_native'] for e in es)/len(es),
                rejected_slot_flags=dict(flags),complete_source_CE=ce,CE_origin='direct_final_measurement' if source else 'derived_four_cyclic_CE_logs_at_unchanged_native_W' if fixed_cycle else 'NA',
                high_loss_frozen=ce is not None and ce>=.1 and len(es)>=10 and all(e['zero_native'] for e in es[-10:]),
                numeric_domination_slots=sum(any(v['numeric_envelope_dominates'] for a in e['candidates'] for v in a['risks'].values()) for e in es),
                source_measurement=source))
        diagnostics[arm]=rows
    counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ('committed_updates','attempt_slots','forwards','vision_forwards','backwards','generation_attempts')}
    result=dict(status='PARTIAL_NOT_COMPLETE',terminal_reason=review['reason'],scientific_success=False,full_protocol_complete=False,
        formal_slots=counts['attempt_slots'],generated_consumers=dict(A=161,B=161,C=0,D=0),planned_total_consumers=644,
        immutable_cache_consumer_coverage={a:dict(v) for a,v in coverage.items()},panels=panels,B_minus_A=contrast,
        C_minus_B='NA_C_OUTPUTS_NOT_GENERATED',D_minus_B='NA_D_NOT_STARTED',
        protection=protections,endpoints=endpoints,diagnostics=diagnostics,
        C_margin_gate=dict(status='REJECTED',all_ten_positive_finite=False,invalid_count='NA_NOT_PERSISTED',individual_values='NA_NOT_PERSISTED',no_model_replay=True),
        C_D_protection=dict(status='UNKNOWN_NOT_GENERATED',observations_per_arm=36,damage_identification_bounds=[0,36],unknown_fails=True),
        actual_counts=counts,new_Judge=ledger['Judge_attempts'],GPU_process_hours=ledger['gpu_seconds_used']/3600,
        owned_matrix_peak_bytes=ledger['weights_observed_peak_bytes'],owned_GPU_peak_MiB=ledger['GPU_memory_observed_peak_MiB'],
        all_GPU_leases_closed=True,generation_barrier_not_overridden=True,C_checkpoint_retained=True,
        B_native_reload='PASS',C_D_final_reload='NOT_RUN',full_68_row_D_peak='NOT_MEASURED_D_NOT_STARTED',
        reporting_engineering_receipts=[c.read(p) for p in (R/'public').glob('ENGINEERING_PUBLICATION_*.json')],failure_cost_included=True,failed_stage_GPU_seconds=sum(s['resident_seconds'] for s in ledger['gpu_sessions'] if s['action']=='learn_C_1'),
        failures=[dict(kind=p.stem,epoch=c.read(p)['epoch']) for p in (R/'private/failures').glob('*.json')],
        unattempted_not_permanent_missing=True,bootstrap_seed=20260912,bootstrap_draws=10000,automatic146=False,
        global_nonlinear_clinical_privacy_certificate=False,original146='PAUSED',M='PAUSED',OL='WAIT_ORIGINAL_FULL_PREDECESSOR')
    c.write(R/'public/RESULTS.json',result)
    text='# B/C/D科学准入止步：partial交付\n\nC第1源20槽后，十个概念/提示间隔未能全部有限且严格为正，触发冻结科学止步。具体不合格数与数值未在异常前落盘，不能进一步细分，不重放模型补取。B完成60槽、161输出和新进程2题精确重载；C仅20槽、无评价输出；D未启动。本轮不是完整644消费者实验，不是预算耗尽或可自动绕过的工程故障。\n\n## 同前三项prefix3，仅读历史全绑定评分\n\n|臂|任务|已知正确/分母|未知|微平均界%|\n|---|---|---:|---:|---|\n'
    for p in panels:
        if p['prefix']==3:text+='|'+ '|'.join(map(str,[p['arm'],p['task'],str(p['known_correct'])+'/'+str(p['observations']),p['unknown'],p['micro_bounds']]))+'|\n'
    text+='\n## 固定36个Base正确保护题\n\n|臂|任务|正确/分母|Base正确→错界|\n|---|---|---:|---|\n'
    for z in protections:text+='|'+ '|'.join(map(str,[z['arm'],z['task'],str(z['known_correct'])+'/'+str(z['observations']),z['Base_correct_to_wrong_bounds']]))+'|\n'
    text+='\nC/D所有评价指标NA（未生成），不能写为0分或永久missing。B未新请求Judge，只在完整绑定与已有终态payload一致时复用；未知区分永久missing与未尝试，详见RESULTS.json。B-A共享缺分精确界及10000次原seed配对edit/source区间保留；C-B/D-B不可评估。\n\n## 训练诊断\n\n|臂|源|槽|接受|零更新比例|完整源CE|CE依据|过早冻结|\n|---|---:|---:|---:|---:|---:|---|---|\n'
    for a,rs in diagnostics.items():
        for z in rs:text+='|'+ '|'.join(map(str,[a,z['source_position'],z['slots'],z['accepted'],z['zero_fraction'],z['complete_source_CE'],z['CE_origin'],z['high_loss_frozen']]))+'|\n'
    text+='\nC仅前三槽接受，后17槽为零，拒绝涉及逐视图实际removed KL硬门；数值包络未达到预算1%支配标记。C完整源CE由固定同W最后四个cyclic CE日志代数平均得到，不是重新调用模型。采样代理可通过而实际视图KL拒绝，不能据此放宽阈值，也不能把线性/采样风险当全局非线性证书。\n\n固定36保护逐题损伤和A→B新增/修复、实际T0/T2G插入到prefix3端点见RESULTS.json；平均/修复不能抵消损伤，未知不通过零损伤门。C/D没有保护输出，损伤只能识别为[0,36]，不是观察到36损伤。所有驻留、失败、准入、保存与调用成本计入；原预算和激活时间未重置，C恢复点及故障现场保留。无新模型/Judge/数据重放，不自动D或146，不恢复原146/M，不启动O/L。\n'
    (R/'public/REPORT_ZH.md').write_text(text)
    c.write(marker,dict(status='PARTIAL_NOT_COMPLETE',scientific_stop=True,epoch=time.time()));print('PARTIAL_NOT_COMPLETE')

if __name__=='__main__':main()
