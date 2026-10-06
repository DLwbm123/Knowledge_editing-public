"""Fixed masks, edit macro means, source sensitivity, and the preregistered gate."""
from collections import Counter,defaultdict
import csv
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
from audit import read,write,digest
from legacy_metrics import metric,paired,micro_delta
from reporting_parent import components


def load():
    phase=read(RUN/'private/SCORING_PHASE.json')['phase']
    assert read(RUN/'private/judge_common/SCORER_DONE.json')['status']=='COMMON_SCORING_COMPLETE_WITH_MISSING'
    l=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');H=read(RUN/'private/H_AVAILABLE.json');tasks=[t for t in l['tasks'] if t['edit_id'] in H]
    db=sqlite3.connect('file:'+str(RUN/'private/judge_common/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    payloads={r['key']:dict(r) for r in db.execute('SELECT * FROM payload')};scores={k:r['correct'] for k,r in payloads.items()};raw={};keys={}
    for c in db.execute('SELECT * FROM consumer'):
        d=read(c['path']);assert digest(d)==c['output_binding']
        key=(c['method'],c['mode'],c['edit_order'],c['query_id']);assert key not in keys
        raw[key]=d;keys[key]=c['payload_key'];b=d['binding']['judge_input'];q=d['binding']['input']
        full=dict(query_id=q['query_id'],question=q['question'],reference=q['reference'],image_sha256=q['image_sha256'],image_path=b['image_path'],prompt_ids=b['prompt_ids'],attention_mask=b['attention_mask'],runtime=b['runtime'],generation=b['generation'],output=d['R0'],judge=read(RUN/'private/judge_common/EPOCH_MANIFEST.json')['judge_identity'])
        assert digest(full)==c['payload_key'] and json.loads(payloads[c['payload_key']]['binding'])==full
    assert all(p['status'] in ('FORMAT_VALID','MISSING') for p in payloads.values());db.close()
    return phase,l,H,tasks,raw,keys,scores


def report():
    phase,l,H,tasks,raw,keys,scores=load();by={t['order']:t for t in tasks};groups=components(tasks,H);gnames={g:f'G{i+1:02d}' for i,g in enumerate(sorted(set(groups.values())))}
    arms=sorted({k[0] for k in raw});rows=[];route=[]
    for arm in arms:
        for t in tasks:
            for event in t['events']:
                for qid in event['all_probe_query_ids']:
                    task=event['task']
                    if task!='T0' and bool(l['Base_correctness'][qid])!=task.endswith('L'):continue
                    for mode in ('single','forced'):
                        k=(arm,mode,t['order'],qid)
                        if k not in keys:continue
                        rows.append(dict(arm=arm,mode=mode,task=task,edit=t['edit_id'],query_id=qid,key=keys[k]))
            for k,d in raw.items():
                if k[0]!=arm or k[2]!=t['order'] or k[1] not in ('H_fit','H_ON','H_OFF'):continue
                basekey=keys[arm,'H_OFF',t['order'],k[3]];base=scores[basekey]
                for task in ('H_fit','H_retain' if base==1 else 'H_repair' if base==0 else 'H_Base_UNKNOWN'):
                    rows.append(dict(arm=arm,mode=k[1],task=task,edit=t['edit_id'],query_id=k[3],key=keys[k]))
    panels=[];units={};eligible={}
    families=sorted({a.rsplit('_s',1)[0] for a in arms})
    for family in families:
        members=[a for a in arms if a.rsplit('_s',1)[0]==family]
        for task in ('T0','T1G','T2G','T1L','T2L','H_fit','H_retain','H_repair','H_Base_UNKNOWN'):
            modes=('single','forced') if not task.startswith('H_') else ('H_fit','H_ON','H_OFF')
            for mode in modes:
                rr=sorted([r for r in rows if r['arm'] in members and r['task']==task and r['mode']==mode],key=lambda r:(r['edit'],r['query_id'],int(r['arm'].rsplit('_s',1)[1])))
                m,u=metric(rr,scores);panels.append(dict(arm=family,seed_slots=[int(a.rsplit('_s',1)[1]) for a in members],task=task,mode=mode,**m))
                units[family,task,mode]=u;eligible[family,task,mode]=rr
    contrasts=[]
    comparisons=[('TT88_DIR',x) for x in ('TT88_FROZEN','TT88_NO_H','TT88_H1','TT88_SMALL_LR_H','TT88_GUARDED_H','TT88_MATCH')]+[('TT88_GUARDED_H','TT88_FROZEN')]+[('TT84_DIR',x) for x in ('TT84_FROZEN','TT84_NO_H','TT84_H1')]+[('MIDDLE_'+h,'OUTER_'+h) for h in ('NO_H','H1')]
    for a,b in comparisons:
        if a not in families or b not in families:continue
        for task in ('T0','T1G','T2G','H_fit','H_retain','H_repair'):
            mode='H_fit' if task.startswith('H_') else 'single';ua=units[a,task,mode];ub=units[b,task,mode]
            contrasts.append(dict(comparison=a+'-'+b,task=task,**paired(ua,ub,scores,groups),micro_delta_bounds=micro_delta(eligible[a,task,mode],eligible[b,task,mode],scores)))
    # Final descriptive source tables; use the same frozen support, never re-score.
    source_panels=[];source_deltas=[]
    for (family,task,mode),rr in eligible.items():
        for group in sorted(gnames):
            selected=[r for r in rr if groups[r['edit']]==group]
            m,_=metric(selected,scores)
            source_panels.append(dict(arm=family,task=task,mode=mode,source_group=gnames[group],**m))
    lookup={(p['arm'],p['task'],p['mode'],p['source_group']):p for p in source_panels}
    for contrast in contrasts:
        a,b=contrast['comparison'].split('-');task=contrast['task'];mode='H_fit' if task.startswith('H_') else 'single'
        for group in gnames.values():
            left=lookup[a,task,mode,group];right=lookup[b,task,mode,group]
            delta=left['macro']-right['macro'] if left['macro'] is not None and right['macro'] is not None else None
            source_deltas.append(dict(comparison=contrast['comparison'],task=task,source_group=group,macro_delta=delta,edits=left['edits']))
    write(RUN/'public/SOURCE_GROUP_RESULTS.json',dict(panels=source_panels,paired=source_deltas,interpretation='Descriptive within-source edit means; five groups, not independent confirmation'))
    # Private relation table joins scope evidence to Base/ON/R0 scores and NLL.
    scope=read(RUN/'private/SCOPE_AUDIT.json');private=[]
    for row in scope['rows']:
        t=by[row['order']];matches=[k for k,d in raw.items() if k[0]=='TT88_FROZEN_s0' and k[1]=='H_OFF' and k[2]==row['order'] and d['binding']['input']['evidence']['review_id']==row['review_id']]
        assert len(matches)==1;k=matches[0];base=scores[keys[k]];measurements=[]
        for kk,d in raw.items():
            if kk[2]!=row['order'] or kk[3]!=k[3] or kk[1] not in ('H_fit','H_ON','H_OFF'):continue
            measurements.append(dict(arm=kk[0],mode=kk[1],correct=scores[keys[kk]],NLL=d['reference_NLL'],route=d['route'],actual_hook=d['generation_diagnostic']['actual_hook_active'],output=d['R0'],binding=d['binding']))
        private.append(dict(row,Base_correct=base,stratum='H_retain' if base==1 else 'H_repair' if base==0 else 'UNKNOWN',source_component=gnames[groups[t['edit_id']]],measurements=measurements))
    write(RUN/'private/SCOPE_ROUTING_TABLE.json',dict(rows=private,original_denominators_unchanged=True))
    source_joint=[]
    for family in ('TT88_DIR','TT84_DIR'):
        frozen=family.split('_')[0]+'_FROZEN'
        if family not in families:continue
        for t in tasks:
            measurements={}
            for a in (family,frozen):
                result={}
                for task in ('T0','T1G','T2G','H_fit'):
                    mode='H_fit' if task=='H_fit' else 'single';rs=[r for r in eligible[a,task,mode] if r['edit']==t['edit_id']];ss=[scores[r['key']] for r in rs]
                    result[task]=sum(ss)/len(ss) if ss and None not in ss else None
                measurements[a]=result
            aa,bb=measurements[family],measurements[frozen]
            complete=all(v is not None for v in [*aa.values(),*bb.values()])
            joint=complete and aa['T0']==1 and aa['H_fit']>bb['H_fit'] and all(aa[x]>=bb[x] for x in ('T1G','T2G'))
            source_joint.append(dict(arm=family,anonymous_edit=f"E{t['order']:03d}",source_group=gnames[groups[t['edit_id']]],joint_success=bool(joint),definition='all-seed T0 correct; within-edit seed-mean T1G/T2G no lower than own FROZEN_W0; H_fit strictly better',scores=measurements))
    gate=None
    if 'TT88_DIR' in families:
        observed={};valid=True
        for task,needed,denom in [('T0',24,24),('T1G',90,93),('T2G',75,81),('H_fit',12,36)]:
            mode='H_fit' if task=='H_fit' else 'single';rr=eligible['TT88_DIR',task,mode];ss=[scores[r['key']] for r in rr]
            observed[task]=dict(correct=sum(s for s in ss if s is not None),expected_denominator=denom,observed=len(ss),missing=sum(s is None for s in ss),required_correct=needed)
            valid &= len(ss)==denom and None not in ss and sum(s or 0 for s in ss)>=needed
        successes={r['source_group'] for r in source_joint if r['arm']=='TT88_DIR' and r['joint_success']}
        valid &=len(successes)>=2
        gate=dict(status='PASS_DEV_ONLY' if valid else 'FAIL_DEV_ONLY',counts=observed,joint_gain_source_groups=len(successes),required_groups=2,next_phase='P2A' if valid else 'P2B',no_independent_confirmation=True)
        write(RUN/'private/P1_GATE.json',gate);write(RUN/'public/P1_GATE.json',gate)
    for family in families:
        for panel in ('T0','T1G','T2G','H_fit'):
            category='H_fit' if panel=='H_fit' else 'single'
            allowed={(r['arm'],r['edit'],r['query_id']) for r in eligible[family,panel,category]}
            selected=[(k,d) for k,d in raw.items() if k[0].rsplit('_s',1)[0]==family and k[1]==category and (k[0],by[k[2]]['edit_id'],k[3]) in allowed]
            if not selected:continue
            forced='forced' if category=='single' else 'H_ON';pairs=[(scores[keys[k]],scores[keys[k[0],forced,k[2],k[3]]]) for k,d in selected]
            route.append(dict(arm=family,panel=panel,n=len(selected),router_on=sum(bool(d['route']['activated']) for k,d in selected),raw_ON_R0_different=sum(d['R0']!=raw[k[0],forced,k[2],k[3]]['R0'] for k,d in selected),ON_correct_R0_wrong=sum(a==0 and b==1 for a,b in pairs),ON_wrong_R0_correct=sum(a==1 and b==0 for a,b in pairs),missing_pairs=sum(a is None or b is None for a,b in pairs)))
    mechanism=[]
    for p in sorted((RUN/'private/edits').glob('e*/*/TRAINING.json')):
        tr=read(p);curve=tr['curve'];mechanism.append(dict(arm=tr['binding']['arm'],anonymous_edit=p.parts[-3],status=tr['status'],steps=tr['steps'],accepted=tr['accepted'],nonzero=tr['nonzero'],H_descent=tr['H_descent'],function_path=tr['function_path'],net_function_displacement=tr['net_function_displacement'],match_relative_error=tr['match_relative_error'],seconds=tr['seconds'],parameters=tr['parameters'],TT_bytes=tr['TT_bytes'],temporary_AB_bytes=tr['temporary_AB_bytes'],peak_allocated=tr['peak_allocated'],peak_reserved=tr['peak_reserved'],projection_seconds=sum(c.get('projection',{}).get('seconds',0) for c in curve),protection_forwards=sum(c['protection_extra_forwards'] for c in curve),constraint_trigger_counts=dict(Counter(i for c in curve for trial in c.get('guard',{}).get('trials',[]) for i in trial['failed_indices'])),H_dot_Adam_negative=sum(c['actual_adam']['H_dot_dA'] is not None and c['actual_adam']['H_dot_dA']<0 for c in curve)))
        mechanism[-1]['fixed_nodes']=[dict(step=c['step'],before=c['diagnostic'],after=c.get('post_diagnostic'),spectrum=c['spectrum'],actual_adam=c['actual_adam'],projection=c.get('projection')) for c in curve if c['diagnostic'] is not None]
        mechanism[-1]['matching_steps_over_2pct']=sum(c.get('match',{}).get('relative_error',0)>.02 for c in curve)
    basecounts=Counter(r['stratum'] for r in private);latencies=[d['generation_diagnostic']['seconds'] for k,d in raw.items() if k[1] in ('forced','H_ON') and d['generation_diagnostic'].get('seconds') is not None]
    resources=read(RUN/'RESOURCE_LEDGER.json');summary=dict(phase=phase,panels=panels,paired=contrasts,source_joint=source_joint,routing=route,H_Base_relation_counts=dict(basecounts),H_scope=scope['summary'],mechanism=mechanism,gate=gate,H_eval='NA',T1L='NA_NO_QUALIFIED_EXTERNAL_DENOMINATOR',T2L='NA_NO_QUALIFIED_EXTERNAL_DENOMINATOR',full146=False,independent_edits=8,seed_repetitions=3,source_components=len(gnames),patient_independence='UNKNOWN',GPU_hours=resources['gpu_seconds_used']/3600,Judge_attempts=resources['Judge_attempts'],score_missing=sum(v is None for v in scores.values()),full_prepared_generation_mean_seconds=sum(latencies)/len(latencies) if latencies else None,inference_timing='complete prepared VQA generation; preprocessing and router measured separately or unavailable, not TT residual time')
    write(RUN/'public'/f'{phase}_RESULTS.json',summary);write(RUN/'public/PAIRED_RESULTS.json',summary)
    write(RUN/'public/FROZEN_W0_RESULTS.json',dict(panels=[p for p in panels if 'FROZEN' in p['arm']],paired=[p for p in contrasts if 'FROZEN' in p['comparison']],mask='original'))
    write(RUN/'public/MECHANISM_BASELINE.json',dict(resource=mechanism,routing=route,W0_candidate_diagnostics=[read(p) for p in sorted((RUN/'private/P0_geometry').glob('*.json'))],full_inference=[read(p) for p in sorted((RUN/'private/inference').glob('*.json'))],core_norms='coordinate dependent, not intrinsic update magnitude',loss='0.5 native+0.5 rotating fit+0.01 full-vocabulary KL(Base||student)+1 H CE',EOS='included; no target truncation <=128',learning_rates=dict(G1=.001,G2=.001,G3=.0001,G4=.0001)))
    with (RUN/'public/RESULTS.csv').open('w') as f:
        fields=sorted({k for p in panels for k in p});writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(panels)
    write(RUN/'public/BUDGET.json',dict(limit_GPU_hours=24,used_GPU_hours=resources['gpu_seconds_used']/3600,phases={a:sum(s.get('resident_seconds',0) for s in resources['gpu_sessions'] if s['action']==a)/3600 for a in sorted({s['action'] for s in resources['gpu_sessions']})},failures_counted=True,clock_reset=False))
    retained=read(RUN/'private/PARENT_REUSE_INVENTORY.json')['items'];banks={}
    for a in sorted({m['arm'] for m in mechanism}):
        items=[m for m in mechanism if m['arm']==a];banks[a]=dict(experts=len(items),total_parameters=sum(m['parameters'] for m in items),TT_bytes=sum(m['TT_bytes'] for m in items))
    write(RUN/'public/STORAGE_AND_BANKS.json',dict(new_expert_banks=banks,retained_parent_W0_count=sum(Path(x['path']).name=='W0.pt' for x in retained),retained_parent_W0_bytes=sum(x['bytes'] for x in retained if Path(x['path']).name=='W0.pt'),router_count=sum(Path(x['path']).name=='ROUTER.pt' for x in retained),router_bytes=sum(x['bytes'] for x in retained if Path(x['path']).name=='ROUTER.pt'),own_weight_peak_observed_bytes=resources.get('weights_observed_peak_bytes'),own_limit_bytes=2*1024**3,shared_base_not_copied=True))
    (RUN/'public/SCOPE_ROUTING_AUDIT.md').write_text('# H作用域、能力与路由审计\n\n原12关系保留，来源连通组5；患者标识UNKNOWN。H_scope不以Base正确为准。\n\nBase分层（关系数，非独立病例）：'+json.dumps(dict(basecounts),ensure_ascii=False)+'。逐关系原问答、图像、作用域证据、Base/ON/R0输出、NLL和hook轨迹仅存私有表。公开模式差异见PAIRED_RESULTS.json。历史来源审核不等于临床签核；未改旧标签或原分母。\n')
    def scoreline(a):
        pp=[p for p in panels if p['arm']==a and p['task'] in ('T0','T1G','T2G','H_fit') and p['mode'] in ('single','H_fit')]
        return '；'.join(f"{p['task']} {p['known_correct']}/{p['probes']}（缺失{p['missing_occurrences']}，编辑宏平均{p['macro']}）" for p in pp) or 'PENDING'
    dirs=[m for m in mechanism if m['arm'].startswith('TT88_DIR')];matched=[m for m in mechanism if m['arm'].startswith('TT88_MATCH')]
    text=['# 科学审阅草案：固定终点、开发性证据',f'阶段{phase}；尚需最终证据审阅和公开发布。8编辑、3优化种子、5来源连通组；不把24次优化当独立病例。CI仅描述少量来源下的不稳定性。','## 1. H低分的来源',f'Base按原12关系分层：{dict(basecounts)}。H_scope不由Base是否答对决定；H_repair是额外纠错，不能称保持。强制ON/R0差异见routing表，Base错误、专家能力与路由不是互斥可相加的因果百分比。','## 2. 是否在同一编辑保住编辑并学会H',scoreline('TT88_DIR'),f"DIR实际接受{sum(m['accepted'] for m in dirs)}，非零{sum(m['nonzero'] for m in dirs)}，训练H下降{sum(m['H_descent'] for m in dirs)}。门槛：{gate['status'] if gate else 'PENDING'}。来源组联合成功按固定W0比较定义；接受率或训练下降均不替代生成任务成功。",'## 3. 是否可由有效更新量解释',scoreline('TT88_MATCH'),f"功能匹配完成块{len(matched)}，MATCH_FAILED {sum(m['status']=='MATCH_FAILED' for m in matched)}。仅在匹配合格的预注册完整支持上解释方向差异；失败时不能排除小更新解释。固定配对差异和来源CI见PAIRED_RESULTS.json。",'## 4. 参数效率或中间秩诊断',scoreline('TT84_DIR') if 'TT84_DIR' in families else ' / '.join(a+': '+scoreline(a) for a in ('OUTER_NO_H','OUTER_H1','MIDDLE_NO_H','MIDDLE_H1')), 'P2只执行门槛选择的一支。中间秩分支为单seed且6400/7168参数不同，只能形成下一轮待验证假设；不自动追加。','## 5. 独立H与局部性', '当前P3为BLOCKED_DATA：已有来源隔离候选，但确认的同模态同部位困难H为0，训练外合格T1L/T2L分母未齐。H_eval/T1L/T2L为NA，患者独立性UNKNOWN。停止在DEV结论，旧H_fit不替代独立评测。',f"累计GPU小时{resources['gpu_seconds_used']/3600:.6f}，上限24；评分新请求{resources['Judge_attempts']}，缺失唯一键{sum(v is None for v in scores.values())}。权重、原始QA和逐题评分不公开。"]
    (RUN/'public/SCIENTIFIC_REVIEW_ZH.md').write_text('\n\n'.join(text)+'\n')
    write(RUN/'private'/f'REPORT_{phase}_COMPLETE.json',dict(status='COMPLETE',epoch=time.time(),phase=phase))
    print('REPORT_COMPLETE',phase,gate,flush=True)


if __name__=='__main__':report()
