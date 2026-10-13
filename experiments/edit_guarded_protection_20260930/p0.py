"""Read-only PR8 dependency and recorded-gradient audits; zero model/Judge."""
import os,json,math,collections,statistics
from pathlib import Path
from exact import delta,new_errors,gate
ROOT=Path(os.environ['RUN_ROOT']);PR8=Path(os.environ['PR8_ROOT'])
def read(p):return json.loads(Path(p).read_text())
def write(name,d):
    p=ROOT/'public'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def text(name,s):(ROOT/'public'/name).write_text(s)
def rows():
    q=read(PR8/'QUEUE.json');rs=[]
    for j in q:
        if j['phase'] not in ['PILOT','DEV']:continue
        if j['phase']=='PILOT' and j['mode']!='train':continue
        for p in (PR8/'jobs'/j['id']).rglob('CONSUMERS.json'):
            for r in read(p):
                if r['arm'] not in ['P_01','SP_01','L','G_0.75'] or r['mode']=='CHECK_FORCED_ON':continue
                if r['mode']=='holdout':r.update(edit='EXPOSED_HOLDOUT',task='EXPOSED_HOLDOUT')
                rs.append(r)
    refs=read(PR8/'private/references/DEV.json');frozen={r['input_id']:r['frozen_base_correct'] for r in refs if r['mode']=='holdout'}
    for r in rs:
        if r['mode']=='holdout':r['frozen_base_correct']=frozen[r['input_id']]
    masks={(x['edit'],x['task'],x['query_id']):x['base_correct'] for x in read(PR8/'private/BASE_MASKS.json')['rows']}
    keep=lambda r:r['task'] not in ['T1L','T2L'] or masks.get((r['edit'],r['task'],r['query_id'])) is True
    return [r for r in rs if keep(r)],[r for r in refs if keep(r)]

def missing_audit():
    scores={p.stem:read(p)['is_correct'] for p in (PR8/'private/judge_sol/scores').glob('*.json')};missing=set(read(PR8/'private/judge_sol/JUDGE_MISSING_LOCK.json')['keys'])
    rs,refs=rows();assert {r[k] for r in rs+refs for k in ['judge_key','base_judge_key']}<=set(scores)|missing
    old=read(PR8/'public/JOINT_GAIN_DECISION.json');out={};changed=[]
    for method,original in old['candidates'].items():
        c=[r for r in rs if r['arm']==method];gates={}
        for label,previous in original['gates'].items():
            if label.startswith('old47/'):
                prefix=int(label.split('/')[1]);panel=lambda a:[r for r in a if r['mode']=='holdout' and r['prefix']==prefix and r.get('frozen_base_correct') is True];a=panel(c);b=panel([r for r in refs if r['arm']=='H'])
            elif label=='pressure_gain':
                panel=lambda a:[r for r in a if r['mode']=='sequential' and r['prefix']==24 and r['task']=='T2L_PRESSURE'];a=panel(c);b=panel([r for r in refs if r['arm']=='H'])
            else:
                parts=label.split('/');mode,prefix,task=parts[:3];target=parts[3] if len(parts)==4 else 'H'
                wanted=['T1G','T2G'] if task=='generality' else [task]
                panel=lambda a:[r for r in a if r['mode']==mode and r['prefix']==int(prefix) and r['task'] in wanted];a=panel(c);b=panel([r for r in refs if r['arm']==target])
            if label.endswith('/T0'):
                d=new_errors(a,b,scores);d.update(delta_min=-d['new_error_max']/d['n'],delta_max=-d['new_error_min']/d['n'],status='PASS' if d['new_error_max']==0 else 'FAIL' if d['new_error_min']>0 else 'INCONCLUSIVE',quantity='negative count of new failures divided by fixed n')
            else:
                d=delta(a,b,scores);d.update(minimum=previous['minimum'],status=gate(d,previous['minimum']))
            d['original_status']=previous['status'];d['original_delta_bounds']=previous.get('delta_bounds');gates[label]=d
            if d['status']!=previous['status']:changed.append(dict(method=method,gate=label,old=previous['status'],exact=d['status']))
        statuses=[g['status'] for g in gates.values()];status='FAIL' if 'FAIL' in statuses else 'INCONCLUSIVE' if 'INCONCLUSIVE' in statuses else 'PASS'
        out[method]=dict(original_status=original['status'],exact_gate_status=status,gates=gates,dominance='No new dominance inferred; original candidates contained missing fields')
    result=dict(status='COMPLETE_ZERO_TRAIN_ZERO_JUDGE',frozen_PR8='06620956d9cea39936e80a03d49fc91f93c9290e',unique_missing=len(missing),model_calls=0,Judge_calls=0,definition='one shared Boolean z_j per UNIQUE permanent missing judge_key; exact affine bounds and exact error polynomials',PR8_original_unmodified=True,candidates=out,status_changes=changed)
    write('EXACT_MISSING_DEPENDENCY_AUDIT.json',result)
    s='# PR8 精确缺失依赖审计\n\n零训练、零模型前向、零新Judge；PR8原判定不变。同一缺失judge_key在任何方法或面板中均为同一z_j，系数先合并再计算精确上下界。\n\n| 方法 | PR8原状态 | 本阶段精确门槛状态 |\n|---|---|---|\n'
    s+='\n'.join(f'|{m}|{d["original_status"]}|{d["exact_gate_status"]}|' for m,d in out.items())+'\n\n'
    for x in changed:s+=f'- {x["method"]} / {x["gate"]}: {x["old"]} → {x["exact"]}\n'
    s+='\n这些是同一批暴露DEV结果的精确重算，未修复或重判永久缺失，不是新的模型效果。各门槛的精确界不意味着可以把不同最优赋值拼成一套模型；joint PASS仍要求所有门槛都对共享变量安全。\n'
    text('EXACT_MISSING_DEPENDENCY_AUDIT_ZH.md',s);return result

def simulate(a2,b2,dot,rho,projection):
    eps=1e-12;active=projection and dot<0;factor=dot/(a2+eps) if active else 0.
    projected2=max(0.,b2-2*factor*dot+factor*factor*a2);postdot=dot-factor*a2
    scale=min(1.,rho*math.sqrt(a2)/(math.sqrt(projected2)+eps));final2=max(0.,a2+scale*scale*projected2+2*scale*postdot)
    return dict(projection_active=active,cap_active=scale<1.,scale=scale,norm_ratio_after=scale*math.sqrt(projected2/a2) if a2 else None,projected_dot=postdot,global_clip=math.sqrt(final2)>1.,D_plus=-(a2+scale*postdot),raw_first_order_only=True)

def gram(record,beta,lam):
    norms=record['gradients']['norms'];cosines=record['gradients']['cosine'];ap={'CE':1.,'L_plus':beta};am={'U_KL':1.,'L_minus':lam}
    def dot(a,b):
        total=0.
        for i,w in a.items():
            for j,v in b.items():
                if not w or not v or not norms[i] or not norms[j]:continue
                cosine=1. if i==j else cosines.get(i+'__'+j,cosines.get(j+'__'+i));assert cosine is not None
                total+=w*v*cosine*norms[i]*norms[j]
        return total
    return max(0.,dot(ap,ap)),max(0.,dot(am,am)),dot(ap,am)

def summarize(rows):
    ratios=[x['norm_ratio'] for x in rows if x['norm_ratio'] is not None]
    return dict(points=len(rows),conflict_fraction=sum(x['conflict'] for x in rows)/len(rows),norm_ratio=dict(median=statistics.median(ratios),min=min(ratios),max=max(ratios)) if ratios else None,offline={f'{arm}_rho{rho}':dict(cap_fraction=sum(x['offline'][f'{arm}_rho{rho}']['cap_active'] for x in rows)/len(rows),projected_fraction=sum(x['offline'][f'{arm}_rho{rho}']['projection_active'] for x in rows)/len(rows),simulated_clip_fraction=sum(x['offline'][f'{arm}_rho{rho}']['global_clip'] for x in rows)/len(rows)) for arm in ['CAP','EGP'] for rho in [1,2]})

def gradient_audit():
    diag=read(PR8/'public/GRADIENT_DIAGNOSTICS.json')['by_method_edit_order'];tasks=read(PR8/'private/TASKS_R2_LOCKED.json')['tasks'];native={t['order']:t['native'] for t in tasks};control=Path(read(PR8/'PREDECESSOR.json')['root']);points={};sources={};actual={};limits=[]
    coeff={'P_01':(.1,0.),'P_1':(1.,0.),'SP_01':(.1,.1),'SP_1':(1.,.1),'L':(0.,.01)}
    for name,d in diag.items():
        method,edit=name.split('/');order=int(edit[1:]);beta,lam=coeff[method];out=[]
        for x in d['records']:
            if x['step']==0:continue
            a2,b2,dot=gram(x,beta,lam);a,b=math.sqrt(a2),math.sqrt(b2)
            out.append(dict(step=x['step'],positive_norm=a,protection_norm=b,norm_ratio=b/a if a else None,dot=dot,cosine=dot/(a*b) if a and b else None,conflict=dot<0,coefficients=dict(CE=1.,U_KL=1.,anchor=beta,residual=lam),component_norms=x['gradients']['norms'],offline={f'{arm}_rho{rho}':simulate(a2,b2,dot,rho,arm=='EGP') for arm in ['CAP','EGP'] for rho in [1,2]},batch='fixed native/first S_fit and first U_old/fixed hard; not the sampled training batch'))
        points[name]=out;sources[name]=native[order].get('source_group',native[order].get('image_sha256',str(order)))
    for root,method,label in [(PR8,m,m) for m in coeff]+[(control,'AH','H'),(control,'AHS_01','S')]:
        for p in (root/'private/curves').glob('s*/'+method+'/e*/continuation.json'):
            order=int(p.parent.name[1:])
            if order>24:continue
            d=read(p);cs=d['curve'];name=label+'/e'+str(order)
            actual[name]=dict(steps=len(cs),clip_fraction=sum(x['preclip_norm']>1 for x in cs)/len(cs),preclip_norm_median=statistics.median(x['preclip_norm'] for x in cs),postclip_norm_max=max(x['postclip_norm'] for x in cs))
            if label in ['H','S']:
                out=[]
                for x in cs:
                    if x['step'] not in [20,40,60,80]:continue
                    a,b=x['CE_gradient_norm'],x['protection_gradient_norm'];dot=(x['CE_U_cosine'] or 0)*a*b
                    out.append(dict(step=x['step'],positive_norm=a,protection_norm=b,norm_ratio=b/a if a else None,dot=dot,cosine=x['CE_U_cosine'],conflict=dot<0,offline={f'{arm}_rho{rho}':simulate(a*a,b*b,dot,rho,arm=='EGP') for arm in ['CAP','EGP'] for rho in [1,2]},batch='actual sampled training gradient',components='S combined U-KL + .1 residual available; separate component vectors/norms not recorded' if label=='S' else 'H combined protection is U-KL'))
                points[name]=out;sources[name]=native[order].get('source_group',native[order].get('image_sha256',str(order)))
            if name in points:
                step_curves={x['step']:x for x in cs}
                for point in points[name]:point['actual_global_clipping_indicator']=step_curves[point['step']]['preclip_norm']>1
    summary={}
    for method in sorted({k.split('/')[0] for k in points}):
        selected={k:v for k,v in points.items() if k.startswith(method+'/')};flat=[x for v in selected.values() for x in v];source_groups=collections.defaultdict(list)
        for k,v in selected.items():source_groups[sources[k]].extend(v)
        source_summary=[]
        for source,source_points in source_groups.items():
            keys=[k for k in selected if sources[k]==source]
            source_summary.append(dict(source_index=len(source_summary)+1,edits=len(keys),recorded_gradient_summary=summarize(source_points),actual_training_clip_fraction=sum(actual[k]['clip_fraction'] for k in keys)/len(keys)))
        summary[method]=dict(source_summaries=source_summary,edit_macro=summarize(flat),edits=len(selected),source_clusters=len(source_groups),source_macro_conflict_fraction=sum(sum(x['conflict'] for x in v)/len(v) for v in source_groups.values())/len(source_groups),actual_training_clip_fraction=sum(x['clip_fraction'] for k,x in actual.items() if k.startswith(method+'/'))/len(selected))
    limits=['S separate KL/residual gradient vectors and norms were not retained; its exact combined protection is available. No invented component split.','PR8 anchor arms have fixed-batch Gram reconstruction and actual clip indicators, but lack actual sampled-batch CE-anchor/protection cross dots. Do not conflate the two batches.','All simulations are offline raw-gradient algebra, not new model results or Adam monotonic guarantees.']
    write('HISTORICAL_GRADIENT_CONFLICT.json',dict(status='COMPLETE_RECORDED_FIELDS_WITH_LIMITATIONS',model_calls=0,Judge_calls=0,summary=summary,per_method_edit_step=points,actual_training_clip=actual,limitations=limits,source_not_patient=True))
    s='# PR8 历史梯度审计\n\n零重训、零新Judge。固定诊断batch按真实beta/lambda系数与记录的Gram内积重构g_plus/g_minus；H/S用原训练曲线的实际联合保护梯度。两种batch不混为同一次更新。\n\n|方法|编辑|冲突比例|norm ratio中位数|实际全程clip比例|\n|---|---|---|---|---|\n'
    s+='\n'.join(f'|{m}|{x["edits"]}|{x["edit_macro"]["conflict_fraction"]:.4f}|{x["edit_macro"]["norm_ratio"]["median"]:.4f}|{x["actual_training_clip_fraction"]:.4f}|' for m,x in summary.items())+'\n\n'+'\n'.join('- '+x for x in limits)+'\n'
    text('HISTORICAL_GRADIENT_CONFLICT_ZH.md',s);return summary

if __name__=='__main__':
    m=missing_audit();g=gradient_audit();print(json.dumps(dict(exact={k:v['exact_gate_status'] for k,v in m['candidates'].items()},gradient_methods=list(g),Judge_calls=0,model_calls=0)))
