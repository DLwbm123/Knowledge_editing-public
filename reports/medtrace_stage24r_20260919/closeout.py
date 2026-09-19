"""CPU-only report assembly from immutable mechanical receipts; no scoring."""
import json,sys
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def main(report):
    report=Path(report);p=report/'private/run/public';s=report/'private/run/strict_mechanical/public'
    mech=read(p/'MECHANICAL_GATE_STAGE24R.json');numeric=read(p/'NUMERIC_PATH_DIAGNOSTIC.json');rep=read(p/'REPEATABILITY.json');cache=read(p/'PREFIX_CACHE_DIAGNOSTIC.json');det=read(p/'DETERMINISM_LOCALIZATION.json');cost=read(p/'COST_PROFILE_STAGE24R.json');length=read(p/'LENGTH_PROFILE.json');gen=read(report/'private/HISTORICAL_GENERATION_COST.json')
    strict=read(s/'MECHANICAL_GATE_STAGE24R.json');strictrep=read(s/'REPEATABILITY.json')
    for name in ['NUMERIC_PATH_DIAGNOSTIC.json','REPEATABILITY.json','MECHANICAL_GATE_STAGE24R.json','PREFIX_CACHE_DIAGNOSTIC.json','DETERMINISM_LOCALIZATION.json','LENGTH_PROFILE.json']:
        write(report/name,read(p/name))
    write(report/'DETERMINISTIC_MECHANICAL_DIAGNOSTIC.json',dict(scope='Separate deterministic-backend diagnostic, not replacement for original frozen runtime',mechanical=strict,repeatability=strictrep))
    ordinary=max(x['seconds'] for r in cost['cases'] for x in r['on'] if not x['diagnostic']);diagnostic=max(x['seconds'] for r in cost['cases'] for x in r['on'] if x['diagnostic'])
    cached_ordinary=max(x['seconds'] for r in cache['cases'] for x in r['times']['cached_ON'] if not x['diagnostic']);cached_diagnostic=max(x['seconds'] for r in cache['cases'] for x in r['times']['cached_ON'] if x['diagnostic']);off=max(x['seconds'] for r in cache['cases'] for x in r['times']['hidden_OFF'])
    sampled_len=max(max(r['valid_tokens']) for r in cost['cases']);population_len=max(max(r['valid_tokens']) for r in length['rows']);ratio=max(1,(population_len/sampled_len)**2)
    setup=max(r['setup_seconds'] for r in cost['cases']);cache_build=max(r['cache_build_seconds'] for r in cache['cases'])
    # Empirical conservative projection, not a proof that future outputs have bounded length.
    estimate=lambda a,b,extra:1.2*(cost['load_seconds']+19*((316*a+4*b)*ratio+setup+extra)+3*gen['per_query_max_seconds_sum'])
    hidden_est=estimate(ordinary,diagnostic,0);cache_est=estimate(cached_ordinary,cached_diagnostic,cache_build)
    cost.update(status='COST_REDUCED_BUT_NOT_A_START_AUTHORIZATION',ordinary_step_max_seconds=ordinary,diagnostic_step_max_seconds=diagnostic,cached_ordinary_step_max_seconds=cached_ordinary,cached_diagnostic_step_max_seconds=cached_diagnostic,n_diagnostic_forecast=4,formal_steps=320,sampled_max_valid_tokens=sampled_len,population19_max_valid_tokens=population_len,length_factor_squared=ratio,setup_max_seconds=setup,cache_build_max_seconds=cache_build,generation=gen,generation_forecast='All79 queries for3 arms, per-query maximum historical cost; method-dependent future lengths unknown',safety_margin=1.2,hidden_R_H19_plus3x79_seconds=hidden_est,cached_R_H19_plus3x79_seconds=cache_est,cached_plus_one_rebuilt_control19_seconds=cache_est+1.2*19*320*off*ratio,phase24C_cap=4800,optional_R0='Not scheduled or included',prefix_cache='same-state feature/loss/gradient exact; frozen-runtime20-step trajectories drift alongside OFF noise; not fully accepted',projection_limits=['Only2 measured training cases, native/wrapper length distribution19 measured','H/U teacher/setup cost sampled; length factor is empirical, not a strict runtime bound','Historical generation may underpredict new method answer lengths','Old bank/outputs cannot be silently treated as compatible with a newly selected deterministic training lane'])
    write(report/'COST_PROFILE_STAGE24R.json',cost)
    ledger=read(report/'private/GPU_BUDGET_LEDGER.json');judge=read(report/'private/JUDGE_BUDGET_SNAPSHOT.json');used=sum(x['seconds'] for x in ledger['sessions']);sessions=[x for x in ledger['sessions'] if x['phase']=='24R'];new=sum(x['seconds'] for x in sessions)
    assert new<=900 and used<=28800 and all(x['exit_code']==0 for x in sessions)
    bill=dict(GPU_seconds_Stage24R=new,Stage24R_cap=900,GPU_seconds_cumulative=used,GPU_seconds_remaining=28800-used,GPU_total_limit=28800,new_Judge=0,Judge_cumulative=judge['new_judgment_items_dispatched'],Judge_remaining=2000-judge['new_judgment_items_dispatched'],sessions=sessions,method_or_semantic_retries=0,scope='mechanical and preregistered numerical/backend/cost localization only')
    write(report/'RESOURCE_LEDGER.json',bill)
    status=dict(status='COMPLETE_WITH_FAILED_ACCEPTANCE',numeric='FP16 return-boundary underflow confirmed; isolated scaling recovers finite nonzero writer gradients',scale_plateau='FAIL_BOTH_CASES',repeatability='Original frozen runtime fails; separate deterministic-backend diagnostic reported without overwriting original gate',Stage24C='NOT_LAUNCHED',Stage25='NOT_LAUNCHED',formal_new_writers=0,new_semantic_judgments=0,no_automatic_continuation=True)
    write(report/'STATUS.json',status)
    rows=[]
    for n,m,d in zip(numeric['cases'],mech['cases'],strict['cases']):
        rows.append(f"| {n['position']} | {n['combined']['1']['L2']:.3g} → {n['combined']['65536']['L2']:.6g} | {n['scale_plateau']['relative_difference']:.2%} / {n['scale_plateau']['cosine']:.6f} | {m['OFF_noise_L2']:.6g} / {m['ON_signal_L2']:.6g} | {d['OFF_noise_L2']:.6g} / {d['ON_signal_L2']:.6g} |")
    text=f'''# Stage24R：数值修复有恢复，验收未通过

已执行包内补丁、现有PyTorch2.6 CPU测试、两例真实GPU分层诊断、3/20步机械更新、重复性与成本定位。未新增正式bank、未做语义评分、未启动24C/25。历史Stage20–24正式工件保持原样。

| 原位置 | combined writer梯度L2（S1→S65536） | S4096/65536相对差 / cosine | 原环境OFF噪声 / ON-OFF信号L2 | 独立确定性诊断OFF噪声 / ON-OFF信号L2 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

上表有效补丁在4个固定输入及五观测的全部真实writer输入上测量；两例实际输入行数分别为{mech['cases'][0]['actual_activation_probe_rows']}、{mech['cases'][1]['actual_activation_probe_rows']}。OFF噪声是多组重复差异的最大值，ON/OFF为20步结果。完整3步、20步参数差、非零数、pre/post clip、Adam及归一化记录见JSON；原始张量和RNG只私有保存。

## 断点与重复性

两例在FP64 pooled叶节点处梯度非零；S=1时返回FP16 token激活的梯度全部归零，FP32 writer也归零。固定S=65536后激活及writer梯度恢复，加入既有CE/H/U FP32梯度的实际增量非零。缩放只作用于组合reg的VJP，在FP32 writer处除回；未缩放既有梯度、未二次backward、未改变科学系数。

但预设尺度平台要求相对差≤1%、cosine≥.999，两例均失败。同一图、同一S65536反复VJP完全一致，而不同S的差异仍存在；不是一次随机波动可以忽略。FP64参考只覆盖相同已舍入pooled值的局部导数，不是完整CUDA Jacobian参考。两项梯度相加与真正组合反传分别记录，未拿前者替代实际梯度。

原冻结环境的Python/NumPy/Torch/CUDA初始RNG相同、模型eval、无训练态dropout、hook清理正常，仍出现OFF差异。独立进程同时设置CUBLAS_WORKSPACE_CONFIG=:4096:8与deterministic_algorithms=True后，两例三次OFF20步及loss曲线一致。该证据将问题定位到后端执行确定性设置相关范围，尚不能单独归因某个CUDA算子；这不是已批准的正式环境替换。历史Stage24未留数值的lambda0瞬时失败及本轮原环境失败全部保留。

## 成本

按实际底层API去除五次HSIC前向的LM head后，全前向和hidden-only特征、loss、writer梯度精确一致。普通步最大约{ordinary:.3f}秒；只去LM head仍不足。

冻结prefix边界为整个L30 block的输入，保留实际attention mask、position IDs、cache positions及RoPE；当前编辑五输入建立，结束释放，不缓存补丁后的状态。同一状态下full/hidden/cached特征、loss、梯度完全一致，普通步最大降到{cached_ordinary:.3f}秒。原环境20步轨迹仍有差异，OFF也漂移，因此不能把单步一致写成完整训练兼容通过。

按316普通步+4诊断步、原19实际长度最大{population_len}对已测{sampled_len}的平方长度因子、加载/teacher/setup、3×79完整DEV生成历史逐题最大耗时、20%余量：hidden-only约{hidden_est:.0f}秒；缓存路径约{cache_est:.0f}秒。缓存估算低于4800秒仅是资源预测，**不是启动许可或机械PASS**。若需另重建一个同环境控制bank，估算约{cost['cached_plus_one_rebuilt_control19_seconds']:.0f}秒。新方法输出长度未知、H/U长度及setup仅两例实测，故不能承诺严格上界或纯45按此速度完成。

## 收口与下一步

当前结论是梯度恢复但数值稳定性门槛未通过，原环境有效更新信号不能超越OFF噪声；独立确定性对照只能作为定位证据。不会自动增大lambda、改变两项比例、提高精度、搜索scale或启动正式19。下一步需先明确稳定VJP实现及正式确定性环境合同，再按同两例重验；若改为ACTIVE_TOKEN_HSIC，必须另立科学变体，不能覆盖当前全valid-token版本。

原190QA/19来源保守规则扫描的零候选结论保留，不能称所有语义候选穷尽。可另提source-disjoint语义关系审阅合同，但本轮不审新参考、不生成新gold、不重命名旧CONFIRM、不下载材料。

本轮GPU {new:.2f}/900秒；后续共享累计{used:.2f}/28800秒，余{28800-used:.2f}秒。新增Judge0，累计{bill['Judge_cumulative']}/2000，未增加预算。所有新机械证据私有保留；公开仅源码与脱敏汇总。
'''
    (report/'ANALYSIS_ZH.md').write_text(text)
    (report/'ADVISOR_UPDATE_ZH.md').write_text(f'''# Stage24R 导师更新

**修复已执行，验收未通过；24C/25没有启动。**

- 固定loss scaling使两例原本为零的组合writer梯度恢复；CPU与真实GPU验证分开记录。
- S4096/65536梯度差{numeric['cases'][0]['scale_plateau']['relative_difference']:.2%}、{numeric['cases'][1]['scale_plateau']['relative_difference']:.2%}，均超过1%门槛，不放宽阈值。
- 原后端OFF不完全复现；单独确定性诊断中OFF噪声为0，ON实际补丁变化非零，但尺度门槛仍失败。不覆盖历史失败、不自动修改正式运行环境。
- 前缀缓存将普通步约{ordinary:.2f}秒降至{cached_ordinary:.2f}秒；单步特征/梯度一致，完整轨迹兼容尚未通过。
- 本轮GPU {new:.1f}秒，新增评分0；共享余{28800-used:.1f}秒。不能据此宣称HSIC有效性或进入正式实验。

详见ANALYSIS_ZH.md及分层数值、重复性、成本和资源JSON。
''')
    print(json.dumps(dict(status=status,resource=bill,cost_seconds=cache_est),ensure_ascii=False))

if __name__=='__main__':main(sys.argv[1])
