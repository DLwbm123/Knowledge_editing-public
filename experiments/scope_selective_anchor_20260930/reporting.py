"""Aggregate only: no medical questions, answers, individual scores or activations."""
import collections,random,time,json
from resources import ROOT,read,write
from phases import candidates,references,arms
from judge_protocol import read_scores
from policy import stats,key

def paired(a,b,scores,cluster):
    aa={key(r):r for r in a};bb={key(r):r for r in b}
    if set(aa)!=set(bb):return dict(status='UNMATCHED')
    groups=collections.defaultdict(list);missing=0
    for k,r in aa.items():
        x,y=scores.get(r['judge_key']),scores.get(bb[k]['judge_key'])
        if x is None or y is None:missing+=1
        else:groups[r[cluster]].append(int(y)-int(x))
    n=len(aa);total=sum(sum(g) for g in groups.values());out=dict(n=n,missing=missing,clusters=len(groups),full_denominator_delta_bounds=[(total-missing)/n,(total+missing)/n] if n else None)
    if len(groups)>=2:
        rng=random.Random(20260929);vs=list(groups.values());samples=[]
        for _ in range(2000):
            z=[x for _ in vs for x in rng.choice(vs)];samples.append(sum(z)/len(z))
        samples.sort();out['CI95_complete_pairs']=[samples[49],samples[1949]]
    else:out['CI_status']='NOT_ESTIMABLE_FEWER_THAN_TWO_CLUSTERS'
    return out

def report(phase):
    rs=candidates(phase)+references(phase);scores=read_scores(ROOT);groups=collections.defaultdict(list)
    masks={(x['edit'],x['task'],x['query_id']):x['base_correct'] for x in read(ROOT/'private/BASE_MASKS.json')['rows']}
    for r in rs:
        if phase!='PILOT' and r['mode']=='CHECK_FORCED_ON':continue
        if r['task'] in ['T1L','T2L'] and masks.get((r['edit'],r['task'],r['query_id'])) is not True:continue
        groups[(r['mode'],r['prefix'],r['task'],r['arm'])].append(r)
    metrics={};comparisons={};fixed={};hold={}
    for (mode,prefix,task,arm),rows in groups.items():
        label=f'{mode}/{prefix}/{task}/{arm}';metrics[label]=stats(rows,scores)
        if mode=='sequential':
            fixedrows=[r for r in rows if r['order'] in (range(25,29) if phase=='REG' else range(1,5))]
            fixed[label]=stats(fixedrows,scores)
        if mode=='holdout':
            eligible=[r for r in rows if r.get('frozen_base_correct') is True]
            hold[label]=dict(unconditional=stats(rows,scores),frozen_Base_correct_retention=stats(eligible,scores),Base_correct_to_wrong=sum(scores.get(r['judge_key']) is False for r in eligible),wrong_to_correct=sum(r.get('frozen_base_correct') is False and scores.get(r['judge_key']) is True for r in rows),activation=sum(r['route']['activated'] for r in rows),activated_damage=sum(r['route']['activated'] and r.get('frozen_base_correct') is True and scores.get(r['judge_key']) is False for r in rows),missing=sum(scores.get(r['judge_key']) is None for r in rows),Sol_Base_correct=sum(scores.get(r['base_judge_key']) is True for r in rows))
        if arm.startswith('SP_'):
            for baseline in ['H','S','P_'+arm.split('_')[1],'L']+[a for a in {r['arm'] for r in rs} if a.startswith('G_')]+['A0','E_orig']:
                ref=groups.get((mode,prefix,task,baseline))
                if ref:
                    comparisons[label+'-vs-'+baseline]={c:paired(ref,rows,scores,c) for c in ['source_group','edit']}
    write(ROOT/f'public/{phase}_METRICS.json',dict(metrics=metrics,paired=comparisons,exploratory=True,source_not_patient=True,CI_complete_pairs_not_missing_bounds=True))
    write(ROOT/f'public/{phase}_FIXED_COHORT_PREFIX_METRICS.json',fixed);write(ROOT/f'public/{phase}_EXPOSED_HOLDOUT_REGRESSION.json',hold)
    text=f'# {phase} 暴露诊断/回归结果\n\nJudge 固定为 GPT‑6.1 sol/high；旧 Astra 分数未混入。正式 locality 使用冻结资格；来源不等同患者。置信区间仅完整配对，缺失上下界使用完整分母。所有比较均为探索性。\n\n'
    text+='|面板/前缀/任务/方法|正确/总数|缺失|\n|---|---|---|\n'
    text+='\n'.join(f'|{k}|{v["correct"]}/{v["n"]}|{v["missing"]}|' for k,v in sorted(metrics.items()))+'\n'
    from storage import Store
    Store(ROOT).write(f'public/{phase}_RESULTS_ZH.md',text.encode())

def close(status,reason,reg):
    from storage import Store
    store=Store(ROOT);ledger=read(ROOT/'RESOURCE_LEDGER.json');q=read(ROOT/'QUEUE.json')
    completed=[j for j in q if j['status']=='COMPLETE'];trains=sum(len(j['methods']) for j in completed if j['mode']=='train')
    gradients={};energy={}
    for p in (ROOT/'private/gradients').glob('*/*.json'):
        if p.parent.name.startswith('SMOKE'):continue
        d=read(p);gradients[p.parent.name+'/'+p.stem]=d;energy[p.parent.name+'/'+p.stem]=[dict(step=x['step'],negative=x.get('negative_energy'),positive=x.get('positive_energy')) for x in d['records']]
    write(ROOT/'public/GRADIENT_DIAGNOSTICS.json',dict(status='COMPLETE_EXECUTED_BLOCKS',by_method_edit_order=gradients))
    write(ROOT/'public/POSITIVE_NEGATIVE_ENERGY.json',dict(status='COMPLETE_EXECUTED_BLOCKS',values=energy,fit_only=True))
    write(ROOT/'public/SCALAR_SHRINK_CONTROL.json',read(ROOT/'public/BETA_GAMMA_SELECTION.json') if (ROOT/'public/BETA_GAMMA_SELECTION.json').exists() else dict(status='NOT_SELECTED'))
    deleted=[]
    for p in (ROOT/'checkpoints').glob('slot*/latest.pt'):
        rel=str(p.relative_to(ROOT));d=store.load_checkpoint(rel);assert d['step']==80
        with store.lock() as db:
            a=db['artifacts'][rel];assert not a['readers'] and not a['consumers'];a['pin']=False;size=a['bytes']
        store.delete(rel);deleted.append(dict(category='completed optimizer/RNG',bytes=size))
    selected=set(arms()[:3]) if (ROOT/'public/BETA_GAMMA_SELECTION.json').exists() and all(read(ROOT/'public/BETA_GAMMA_SELECTION.json')[k]['selected'] is not None for k in ['beta','gamma']) else set()
    for p in (ROOT/'adapters/s20260929').glob('*/*.pt'):
        if p.parent.name in selected:continue
        # All completed consumers and immutable responses are now scored or locked missing.
        rel=str(p.relative_to(ROOT))
        with store.lock() as db:
            a=db['artifacts'][rel];assert not a['readers'] and not a['consumers'];a['pin']=False;size=a['bytes']
        store.delete(rel);deleted.append(dict(category='unselected pilot final adapter',bytes=size))
    write(ROOT/'public/CHECKPOINT_LIFECYCLE.json',dict(status='COMPLETE_EXECUTED_BLOCKS',old_artifacts_untouched=True,deleted=deleted,retained='selected final adapters for reproducibility and requested comparisons',active_resume=0))
    actual=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    write(ROOT/'public/COST_AND_STORAGE.json',dict(continuations=trains,steps=trains*80,smoke_steps=320,GPU_seconds=ledger['current_gpu_seconds'],Judge_cumulative=ledger['judge_submission_attempt_items'],Judge_new=ledger['current_judge_attempts'],owned_bytes=actual,soft_bytes=10*1024**3,hard_bytes=20*1024**3,prefill='NOT_SEPARATELY_MEASURED',compute_by_job={j['id']:read(ROOT/'jobs'/j['id']/'COMPUTE_COUNTS.json') for j in completed}))
    if status!='COMPLETE':store.write('public/REG_RESULTS_ZH.md',('# REG 未运行\n\n'+(reg or reason)+'。不视为独立 CONFIRM。\n').encode())
    store.write('public/FINAL_RESULTS_ZH.md',('# 选择性正例锚定实验\n\n状态：'+status+'。'+reason+'。\n\n已完成 '+str(trains)+' 条正式续训，各80步；smoke额外320步单列。详见 PILOT/DEV/REG_RESULTS_ZH.md、JOINT_GAIN_DECISION.json、BETA_GAMMA_SELECTION.json。没有独立确认集；旧47输入面板保持暴露回归身份。无结果的阶段不推断正负效果。prefill未单独测量。\n').encode())
    store.write('public/NEXT_DECISION.md',('# 下一步\n\n'+reason+'。不自动扩展 seed、rank、门控或失败分支。\n').encode())
