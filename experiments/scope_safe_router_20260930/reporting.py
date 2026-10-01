"""Only aggregate routing outcomes are public; inputs/tokens/scores stay private."""
from collections import Counter
import json,time
from resources import ROOT,read,write
from judge_protocol import read_scores,NAMESPACE
from phases import references,candidates,methods
from decision import transitions,stats
def text(name,value):
    from storage import Store
    Store(ROOT).write('public/'+name,value.encode())
def anonymize(summary,phase):
    tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];mapping={t['canonical_edit_id']:phase+'e'+str(t['order'] if phase=='DEV' else t['order']-24) for t in tasks};mapping[None]='OFF'
    for cell in summary['expert_switch_matrix']:
        for field in ['from_expert','to_expert']:cell[field]=mapping[cell[field]]
    return summary
def report(phase):
    rows=candidates(phase);refs=references(phase);scores=read_scores(ROOT);panels={};results={}
    for method in methods(phase):
        rs=[r for r in rows if r['arm']==method];panels[method]={};results[method]={}
        for mode,prefix in [('single',1)]+[('sequential',p) for p in [4,8,12,24]]+[('EXPOSED_REGRESSION',p) for p in [4,8,12,24]]:
            a=[r for r in rs if r['mode']==mode and r['prefix']==prefix];b=[r for r in refs if r['mode']==mode and r['prefix']==prefix]
            assert a and b;key=mode+'/'+str(prefix);panels[method][key]=anonymize(transitions(a,b,scores),phase)
            tasknames=['EXPOSED_REGRESSION'] if mode=='EXPOSED_REGRESSION' else ['T0','T1G','T2G','T1L','T2L','T2L_PRESSURE']
            results[method][key]={task:stats([r for r in a if r['task']==task],scores) for task in tasknames}
        for left,right in [(8,12),(12,24)]:
            for mode in ['sequential','EXPOSED_REGRESSION']:
                a=[r for r in rs if r['mode']==mode and r['prefix']==right];b=[r for r in rs if r['mode']==mode and r['prefix']==left]
                key=f'{mode}/bank{left}_to_bank{right}';panels[method][key]=anonymize(transitions(a,b,scores,shared_only=True),phase);panels[method][key]['fixed_shared_inputs_only']=True
    path=ROOT/'public/ROUTE_TRANSITION_ANALYSIS.json';allphases=read(path) if path.exists() else {};allphases[phase]=panels;write(path,allphases)
    write(ROOT/f'public/{phase}_ROUTER_AGGREGATES.json',results)
    s=f'# {phase} Router-only 结果\n\n所有方法 expert weights 冻结；训练/backward/optimizer 为0。已完成 single、bank4/8/12/24 和 EXPOSED_REGRESSION old47。完整分母保留 missing；原 Base 资格不重定义。\n\n|Router|bank24 T2G已知正确/完整分母|bank24 Pressure已知正确/完整分母|缺失(T2G/Pressure)|\n|---|---|---|---|\n'
    for method in methods(phase):
        x=results[method]['sequential/24'];a=x['T2G'];b=x['T2L_PRESSURE'];s+=f'|{method}|{a["known_correct"]}/{a["n"]}|{b["known_correct"]}/{b["n"]}|{a["missing"]}/{b["missing"]}|\n'
    s+='\nroute buckets、switch matrix、Positive Rejection、Negative Rescue、Negative Damage、NetRescue，以及bank8→12/12→24固定共享输入面板见 ROUTE_TRANSITION_ANALYSIS.json。NetRescue不替代联合门槛。DEV为暴露诊断，REG为暴露回归，不是独立确认。T1L/T2L资格以原冻结 BASE_MASKS 决定，未合格输入不充入正式locality分母。\n'
    text(phase+'_ROUTER_RESULTS_ZH.md',s)
def cal_audit():
    tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];pilot=read(ROOT/'private/PILOT_SELECTION.json')['orders'];checks=read(ROOT/'private/CHECK_POS.json');g=read(ROOT/'private/G_SUPPORTS.json');peredit=[]
    for order in pilot:
        t=tasks[order-1];rows=checks[str(order)];same=[r for r in rows if r['image_sha256']==t['native']['image_sha256']];cross=[r for r in rows if r['image_sha256']!=t['native']['image_sha256']]
        inherited=next(r for r in g if r['order']==order)
        assert inherited['native_only'] and len(cross)==0
        peredit.append(dict(edit_order=order,inherited_same_image_CHECK_candidates=len(same),fit_same_image_supports_excluded=len(t['semantic_fit_questions']),formal_cross_image_positive_excluded=sum(r['task']=='T2G' and r['image_sha256']!=t['native']['image_sha256'] for r in t['evaluation']),verified_isolated_same_image_CAL=0,verified_isolated_cross_image_CAL=0,minimum_same_image=4,minimum_cross_image=2,verification='No separate scope verification/provenance evidence supplied; source-answer-derived paraphrases are candidate proposals only',status='CAL_SCOPE_UNAVAILABLE'))
    audit=dict(status='CAL_SCOPE_UNAVAILABLE',pilot_edits=peredit,existing_scope_support_native_only=True,cross_image_existing_candidates=0,formal_positive_examples_excluded_from_new_CAL=True,fit_U_bg_DEV_REG_old47_not_repurposed=True,automatic_rewrites_not_verified_cross_image=True,data_created=0,router_retuned=False,required_next_data='Independently verified, correctly role-isolated same-image and cross-image scope annotations',epoch=time.time())
    write(ROOT/'public/CAL_SCOPE_DATA_AUDIT.json',audit)
def close(status,reason,reg=None):
    decision=read(ROOT/'public/JOINT_ROUTER_DECISION.json');ledger=read(ROOT/'RESOURCE_LEDGER.json');queue=read(ROOT/'QUEUE.json');scores=read_scores(ROOT);missing=read(ROOT/NAMESPACE/'JUDGE_MISSING_LOCK.json')['keys'];attempts=ledger['judge_attempts'][ledger['historical_judge_attempt_count']:];costs=[];reuse=Counter()
    for j in queue:
        assert j['status']=='COMPLETE'
        costs.append(read(ROOT/'jobs'/j['id']/'COMPUTE_COUNTS.json'))
        for p in (ROOT/'jobs'/j['id']).rglob('GENERATION_REUSE.json'):reuse.update(read(p))
    assert all(c['training_steps']==c['backward_calls']==c['optimizer_steps']==0 and c['expert_weights_unchanged'] for c in costs)
    assert all(s.get('ended_epoch') for s in ledger['gpu_sessions'])
    allrows=[r for phase in ['DEV','REG'] if any(j['phase']==phase for j in queue) for r in candidates(phase)]
    unresolved={r[k] for r in allrows for k in ['judge_key','base_judge_key']}-set(scores)-set(missing);assert not unresolved
    audit=dict(status=status,reason=reason,training_steps=0,backward_calls=0,optimizer_steps=0,experts_frozen=True,new_adapters_created=0,checkpoints_deleted=0,historical_assets_readonly=True,all_jobs_complete=True,jobs_by_phase=dict(Counter(j['phase'] for j in queue)),REG_status='COMPLETE' if status=='COMPLETE' else 'NOT_ADMITTED',REG_methods=decision['REG_methods'],GPU_only=[5,6,7],all_GPU_leases_ended=True,GPU_hours=ledger['current_gpu_seconds']/3600,formal_backbone_calls=sum(c['model_calls'] for c in costs),generation_reuse=dict(reuse),inherited_Judge_attempts=ledger['historical_judge_attempts'],new_Judge_attempts=ledger['current_judge_attempts'],cumulative_Judge_attempts=ledger['judge_submission_attempt_items'],new_physical_requests=len(attempts),successful_payload_cache=len(scores),permanent_missing_payloads=len(missing),unresolved=0,no_transport_retries=True,CAL_SCOPE='CAL_SCOPE_UNAVAILABLE',independent_CONFIRM=False,public_delivery='PENDING_GITHUB',epoch=time.time())
    write(ROOT/'public/FINAL_EXECUTION_AUDIT.json',audit)
    if not decision['REG_allowed']:text('REG_ROUTER_RESULTS_ZH.md','# REG Router 结果\n\nNOT_ADMITTED：没有明确 DEV PASS，未启动 REG，没有训练或调参。\n')
    s='# Scope-safe Router 最终结果\n\n'+reason+'。专家权重全程冻结，训练/backward/optimizer 为0。\n\n|Candidate|联合判定|bank24 Pressure已知增益|精确差界|已知route rescue出现次数|\n|---|---|---|---|---|\n'
    for method,d in decision['candidates'].items():
        gate=d['gates']['bank24/Pressure'];s+=f'|{method}|{d["status"]}|{gate["known_correct_gain"]:+d}|[{gate["delta_min"]*100:.3f}, {gate["delta_max"]*100:.3f}] pp|{d["known_route_rescue_occurrences"]}|\n'
    hazard=read(ROOT/'public/INSERTION_HAZARD_AUDIT.json')['expert12'];s+=f'\nREG expert12仅使用U_bg时H_count={hazard["H_count"]}，H_source={hazard["H_source"]}，H_switch={hazard["H_switch"]}，排名{hazard["rank"]}。hazard预先冻结后才读取暴露damage；非空capture可以识别latent capture风险，不能证明它是可部署的损伤分类器。\n'
    s+='\n完整机械验证见 ROUTER_MECHANICAL_TESTS.json；科学门槛及shared-key missing界见 JOINT_ROUTER_DECISION.json；按完整分母的主结果见 DEV_ROUTER_RESULTS_ZH.md。没有加权总分；路由救援不替代T0/T1G/T2G/Pressure/old35/T2L门槛。原冻结T1L资格为空，明确不可评估，不当作accuracy通过。\n\nCAL_SCOPE_UNAVAILABLE：继承支持及CHECK为native图像改写，未提供>=4独立verified同图和>=2隔离verified跨图scope-positive。未复制formal面板或自动改写充作新cross-image独立样本；本轮参数不变。若联合门槛未通过，现有native/S_fit+U_bg信息不足以支持本轮部署结论，优先获取合法scope校准标注。\n'
    s+=f'\n本轮已收口，REG={audit["REG_status"]}；GPU驻留{audit["GPU_hours"]:.4f}小时（含latent构造、机械检查及推理，不是训练时间）。新Judge {audit["new_Judge_attempts"]}项，累计{audit["cumulative_Judge_attempts"]}，无重试，积压0。复用生成统计{dict(reuse)}。本轮无adapter/optimizer或权重清理，历史资产不动。\n\nDEV与old47为暴露面板；REG即便执行也为EXPOSED_REGRESSION，不是independent confirmation。不重新训练expert，不搜索阈值，不自动进入下一阶段。公开仅source/config/tests/脱敏聚合与限制；私有输入、raw responses、逐题评分、weights及凭证不公开。\n'
    text('FINAL_RESULTS_ZH.md',s)
