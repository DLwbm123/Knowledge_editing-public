"""Read-only reviewed calibration-stop delivery; no learner, generation or Judge replay."""
import json,os,sqlite3,time
from pathlib import Path
import common as c
R=c.RUN

def main():
    review=c.read(R/'private/ADMISSION_STOP_REVIEW.json');assert review['reason']=='BASE_DECISION_MARGIN_NOT_ADMITTED'
    cost=c.read(R/'RESOURCE_LEDGER.json');assert cost['Judge_attempts']==0 and all(x.get('ended_epoch') for x in cost['gpu_sessions'])
    counts={k:sum(c.read(p).get(k,0) for p in (R/'private/attempts').glob('*.json')) for k in ('forwards','vision_forwards','backwards','committed_updates','attempt_slots','generation_attempts')}
    assert counts['attempt_slots']==counts['committed_updates']==0
    assert not (R/'private/GENERATION_COMPLETE.json').exists() and not (R/'private/TRAINING_GENERATION_COMPLETE.json').exists()
    db=sqlite3.connect('file:'+str(R/'private/judge_purew146/queue.sqlite')+'?mode=ro',uri=True)
    assert db.execute('SELECT count(*) FROM consumer').fetchone()[0]==0
    assert not db.execute("SELECT 1 FROM payload WHERE status IN ('PENDING','RESERVED')").fetchone()
    assert not (R/'private/judge_purew146/ALL_WORKERS_COMPLETE.json').exists()
    assert all(not (R/'public'/('REAL_VLM_ADMISSION_'+a+'.json')).exists() for a in ('N','J','NJ'))
    outputs=list((R/'private/calibration_outputs').glob('*.json'));attempted=[p for p in outputs if p.name.endswith('.STARTED.json')];durable=[p for p in outputs if not p.name.endswith('.STARTED.json')]
    assert len(attempted)==len(durable)==counts['generation_attempts']==2
    assert c.read(R/'private/control/CAL/1.DONE.json')['source_buffers_retired']
    assert not (R/'private/control/CAL/2.DONE.json').exists()
    marker=R/'private/REPORT_COMPLETE.json'
    if marker.exists():assert c.read(marker)['status']=='PARTIAL_NOT_COMPLETE';print('PARTIAL_NOT_COMPLETE');return
    result=dict(status='PARTIAL_NOT_COMPLETE',terminal_reason=review['reason'],scientific_success=False,full_protocol_complete=False,
        activation=c.read(R/'RUN_MANIFEST.json')['starting_epoch'],deadline=c.read(R/'RUN_MANIFEST.json')['deadline_epoch'],physical_GPU=2,
        calibration=dict(historical_exact_durable_outputs=2,required=8,first_source_factor_window_finished=True,second_source_strict_decision_gap_rejected=True,normalized_and_plain_final_factors_saved=False,
            offending_token_count='NA_NOT_PERSISTED',minimum_gap='NA_NOT_PERSISTED',per_token_gaps='NA_NOT_PERSISTED',gate_is_finite_strictly_positive=True,token_or_source_subsets_removed=False,gap_floor_added=False,no_model_replay=True),
        actual_counts=counts,new_Judge=0,registered_new_consumers=0,generated_evaluation_consumers=0,
        new_arms={a:dict(status='NOT_STARTED_CALIBRATION_GATE',formal_slots=0,real_VLM_admission='NOT_RUN',T0='NA_NOT_GENERATED',T1G='NA_NOT_GENERATED',T2G='NA_NOT_GENERATED',T1L='NA_NOT_GENERATED',T2L='NA_NOT_GENERATED',fixed36_damage_identification_bounds=[0,36],unknown_protection_fails=True) for a in ('N','J','NJ')},
        contrasts={k:'NA_NEW_ARM_NOT_GENERATED' for k in ('N-B','J-B','NJ-J','NJ-N')},
        historical_B='Existing immutable161 outputs; prior159 valid+2 unattempted consumers unchanged; not rejudged or relabeled permanent missing',
        GPU_process_seconds=cost['gpu_seconds_used'],GPU_process_hours=cost['gpu_seconds_used']/3600,all_GPU_leases_closed=True,
        generation_and_scoring_complete_barriers_not_overridden=True,unattempted_not_permanent_missing=True,
        owned_matrix_peak_bytes=cost.get('weights_observed_peak_bytes',0),failures=[dict(kind=p.stem,epoch=c.read(p).get('epoch')) for p in (R/'private/failures').glob('*.json')],
        reporting_boundary='No new efficacy comparison; CPU checks do not establish native strict-margin admission',
        original146='PAUSED',M='PAUSED',OL='WAIT_ORIGINAL_FULL_PREDECESSOR',automatic146=False,clinical_or_privacy_claim=False)
    c.write(R/'public/RESULTS.json',result)
    text='''# N/J/NJ校准科学准入止步：partial交付

GPU2一次安装后，第二个Base校准源的原始token top1–top2概率间隔未能全部有限且严格为正，触发BASE_DECISION_MARGIN_NOT_ADMITTED。第一源完成两种初始因子的当前窗计算；第二源在形成统计前止步。共2个新鲜Base回答/tokens已持久化且与历史精确相等；没有保存完整plain/normalized因子，没有进入三臂真实VLM准入和正式训练。

实际消耗为0正式槽、0写入、4反传、48前向（其中4 vision，已包含）、2生成、0新Judge；GPU驻留/预留/失败时间全部计入原激活预算，详见RESULTS.json。原准入和controller异常、START和一次尝试账本均保留；未重复安装或重放。

本次没有N/J/NJ评价结果，T0/T1G/T2G/T1L/T2L均NA，不写0分，不写永久missing。每臂固定36保护的损伤只能识别为[0,36]，未知不通过零损伤。历史B161输出及159有效+2未尝试消费者状态不变，不单独补Judge；所有新差值不可评估。没有伪造GENERATION_COMPLETE、TRAINING_GENERATION_COMPLETE或ALL_WORKERS_COMPLETE。无新增消费者待评分，不能称完成完整科学评分。

严格正mu是1/mu²归一化的数学定义域；未通过时不能用floor/删token/删源或替换采样绕过。实际不合格token数与间隔值未在抛异常前保存，无法再细分；不重放补取。有限精度下分布并列/极小间隔说明原始argmax翻转率目标可能对决策歧义敏感，不能凭此次日志证明其确切发生数量，也不能说J方向机制被否定——J尚未开始。后续如改为集合决策/正间隔有条件证书，必须另行推导冻结，不能悄改本轮。

CPU的KKT/恢复/评分流程检查通过仍不等真实校准通过。当前是明确科学准入止步，不是GPU不足；原146/M继续暂停，O/L继续等待，无自动下一轮。源码、日志诊断、失败和成本匿名公开后暂停本轮半小时监测。
'''
    (R/'public/REPORT_ZH.md').write_text(text);c.write(marker,dict(status='PARTIAL_NOT_COMPLETE',scientific_admission_stop=True,epoch=time.time()));print('PARTIAL_NOT_COMPLETE')

if __name__=='__main__':main()
