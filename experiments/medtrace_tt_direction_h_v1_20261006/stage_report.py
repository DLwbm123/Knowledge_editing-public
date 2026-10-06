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
    comparisons=[('TT88_DIR',x) for x in ('TT88_FROZEN','TT88_NO_H','TT88_H1','TT88_GUARDED_H','TT88_MATCH')]+[('TT88_GUARDED_H','TT88_FROZEN')]+[('TT84_DIR',x) for x in ('TT84_FROZEN','TT84_NO_H','TT84_H1')]+[('MIDDLE_'+h,'OUTER_'+h) for h in ('NO_H','H1')]
    for a,b in comparisons:
        if a not in families or b not in families:continue
        for task in ('T0','T1G','T2G','H_fit','H_retain','H_repair'):
            mode='H_fit' if task.startswith('H_') else 'single';ua=units[a,task,mode];ub=units[b,task,mode]
            contrasts.append(dict(comparison=a+'-'+b,task=task,**paired(ua,ub,scores,groups),micro_delta_bounds=micro_delta(eligible[a,task,mode],eligible[b,task,mode],scores)))
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
        for category in ('single','H_fit'):
            selected=[(k,d) for k,d in raw.items() if k[0].rsplit('_s',1)[0]==family and k[1]==category]
            if not selected:continue
            forced='forced' if category=='single' else 'H_ON';pairs=[(scores[keys[k]],scores[keys[k[0],forced,k[2],k[3]]]) for k,d in selected]
            route.append(dict(arm=family,panel=category,n=len(selected),router_on=sum(bool(d['route']['activated']) for k,d in selected),raw_ON_R0_different=sum(d['R0']!=raw[k[0],forced,k[2],k[3]]['R0'] for k,d in selected),ON_correct_R0_wrong=sum(a==0 and b==1 for a,b in pairs),ON_wrong_R0_correct=sum(a==1 and b==0 for a,b in pairs),missing_pairs=sum(a is None or b is None for a,b in pairs)))
    mechanism=[]
    for p in sorted((RUN/'private/edits').glob('e*/*/TRAINING.json')):
        tr=read(p);curve=tr['curve'];mechanism.append(dict(arm=tr['binding']['arm'],anonymous_edit=p.parts[-3],status=tr['status'],steps=tr['steps'],accepted=tr['accepted'],nonzero=tr['nonzero'],H_descent=tr['H_descent'],function_path=tr['function_path'],net_function_displacement=tr['net_function_displacement'],match_relative_error=tr['match_relative_error'],seconds=tr['seconds'],parameters=tr['parameters'],TT_bytes=tr['TT_bytes'],temporary_AB_bytes=tr['temporary_AB_bytes'],peak_allocated=tr['peak_allocated'],peak_reserved=tr['peak_reserved'],projection_seconds=sum(c.get('projection',{}).get('seconds',0) for c in curve),protection_forwards=sum(c['protection_extra_forwards'] for c in curve),constraint_trigger_counts=dict(Counter(i for c in curve for trial in c.get('guard',{}).get('trials',[]) for i in trial['failed_indices'])),H_dot_Adam_negative=sum(c['actual_adam']['H_dot_dA'] is not None and c['actual_adam']['H_dot_dA']<0 for c in curve)))
    basecounts=Counter(r['stratum'] for r in private);latencies=[d['generation_diagnostic']['seconds'] for k,d in raw.items() if k[1] in ('forced','H_ON') and d['generation_diagnostic'].get('seconds') is not None]
    resources=read(RUN/'RESOURCE_LEDGER.json');summary=dict(phase=phase,panels=panels,paired=contrasts,source_joint=source_joint,routing=route,H_Base_relation_counts=dict(basecounts),H_scope=scope['summary'],mechanism=mechanism,gate=gate,H_eval='NA',T1L='NA_NO_QUALIFIED_EXTERNAL_DENOMINATOR',T2L='NA_NO_QUALIFIED_EXTERNAL_DENOMINATOR',full146=False,independent_edits=8,seed_repetitions=3,source_components=len(gnames),patient_independence='UNKNOWN',GPU_hours=resources['gpu_seconds_used']/3600,Judge_attempts=resources['Judge_attempts'],score_missing=sum(v is None for v in scores.values()),full_prepared_generation_mean_seconds=sum(latencies)/len(latencies) if latencies else None,inference_timing='complete prepared VQA generation; preprocessing and router measured separately or unavailable, not TT residual time')
    write(RUN/'public'/f'{phase}_RESULTS.json',summary);write(RUN/'public/PAIRED_RESULTS.json',summary)
    write(RUN/'public/FROZEN_W0_RESULTS.json',dict(panels=[p for p in panels if 'FROZEN' in p['arm']],paired=[p for p in contrasts if 'FROZEN' in p['comparison']],mask='original'))
    write(RUN/'public/MECHANISM_BASELINE.json',dict(resource=mechanism,routing=route,core_norms='coordinate dependent, not intrinsic update magnitude',loss='0.5 native+0.5 rotating fit+0.01 full-vocabulary KL(Base||student)+1 H CE',EOS='included; no target truncation <=128',learning_rates=dict(G1=.001,G2=.001,G3=.0001,G4=.0001)))
    with (RUN/'public/RESULTS.csv').open('w') as f:
        fields=sorted({k for p in panels for k in p});writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(panels)
    write(RUN/'public/BUDGET.json',dict(limit_GPU_hours=24,used_GPU_hours=resources['gpu_seconds_used']/3600,phases={a:sum(s.get('resident_seconds',0) for s in resources['gpu_sessions'] if s['action']==a)/3600 for a in sorted({s['action'] for s in resources['gpu_sessions']})},failures_counted=True,clock_reset=False))
    (RUN/'public/SCOPE_ROUTING_AUDIT.md').write_text('# H作用域、能力与路由审计\n\n原12关系保留，来源连通组5；患者标识UNKNOWN。H_scope不以Base正确为准。\n\nBase分层（关系数，非独立病例）：'+json.dumps(dict(basecounts),ensure_ascii=False)+'。逐关系原问答、图像、作用域证据、Base/ON/R0输出、NLL和hook轨迹仅存私有表。公开模式差异见PAIRED_RESULTS.json。历史来源审核不等于临床签核；未改旧标签或原分母。\n')
    write(RUN/'private'/f'REPORT_{phase}_COMPLETE.json',dict(status='COMPLETE',epoch=time.time(),phase=phase))
    print('REPORT_COMPLETE',phase,gate,flush=True)


if __name__=='__main__':report()
