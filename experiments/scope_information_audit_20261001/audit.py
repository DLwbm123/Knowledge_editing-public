"""Exposed geometry/source diagnosis; no classifier, output generation, or Judge."""
import json,os,sys,time,math,traceback,collections
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);SOURCE=Path(os.environ['SOURCE_ROOT'])
sys.path[:0]=[str(SOURCE),str(SOURCE/'source_patch'),str(SOURCE/'source')]
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def identity(r):return tuple(r[k] for k in ['mode','prefix','edit','task','input_id'])
def label(a,b,scores):
 x=scores.get(a['judge_key']);y=scores.get(b['judge_key'])
 if x is None or y is None:return 'MISSING'
 changed=a['route']['logical_edit_id']!=b['route']['logical_edit_id']
 if changed and x and not y and a['task'] in ['T0','T1G','T2G'] and a['route']['logical_edit_id']==a['edit']:return 'POSITIVE_REJECTION'
 if changed and not x and y and a['task'] in ['T2L_PRESSURE','EXPOSED_REGRESSION']:return 'NEGATIVE_RESCUE'
 return 'OTHER_KNOWN'
def distribution(values):
 finite=sorted(v for v in values if v is not None)
 return dict(n=len(values),finite=len(finite),infinite=len(values)-len(finite),min=min(finite) if finite else None,max=max(finite) if finite else None,median=finite[len(finite)//2] if finite else None)
def summarize(rows):
 out={}
 for name in ['POSITIVE_REJECTION','NEGATIVE_RESCUE','MISSING','OTHER_KNOWN']:
  rs=[r for r in rows if r['label']==name]
  out[name]=dict(occurrences=len(rs),unique_inputs=len({r['input_id'] for r in rs}),unique_input_winner_pairs=len({(r['input_id'],r['winner']) for r in rs}),tasks=dict(collections.Counter(r['task'] for r in rs)),same_image_as_winner=sum(r.get('same_image') is True for r in rs),cross_image_as_winner=sum(r.get('same_image') is False for r in rs),geometry={k:distribution([r[k] for r in rs if k in r]) for k in ['guard_ratio','quotient','nearest_background_over_positive','query_to_key_over_radius']})
 return out
def selfcheck():
 r=dict(judge_key='yes',task='T1G',edit='e',route=dict(logical_edit_id='e'))
 b=dict(r,judge_key='no',route=dict(logical_edit_id=None))
 assert label(r,b,{'yes':True,'no':False})=='POSITIVE_REJECTION'
 assert label(r,b,{'yes':True})=='MISSING'
 assert distribution([None,2.,1.])==dict(n=3,finite=2,infinite=1,min=1.,max=2.,median=2.)
def main():
 import torch
 from m3bench_repro.editors.routing import distances
 from judge_protocol import read_scores,digest
 selfcheck();config=read(ROOT/'public/ROUTER_CONFIG.json');manifest=read(ROOT/'RUN_MANIFEST.json');started=time.time()
 physical=int(os.environ['PHYSICAL_GPU']);assert physical in [6,7] and physical==config['physical_GPU']
 assert read(SOURCE/'RUN_STATUS.json')['status']=='CLOSED_NOT_ADMITTED'
 assert read(SOURCE/'public/FINAL_EXECUTION_AUDIT.json')['public_delivery']=='VERIFIED_GITHUB'
 assert read(SOURCE/'public/DATA_ROLE_AUDIT.json')['status']=='PASS'
 torch.set_grad_enabled(False)
 def forbidden(*a,**k):raise RuntimeError('Diagnostic forbids backward')
 torch.Tensor.backward=torch.autograd.backward=forbidden
 features=torch.load(SOURCE/'private/ROUTER_FEATURES.pt',map_location='cpu',weights_only=False)
 bank={e['edit']:e for e in features['entries']};background=features['background'].cuda();tasks=read(SOURCE/'private/TASKS_R2_LOCKED.json')['tasks'];owners={t['canonical_edit_id']:t for t in tasks}
 raw={digest([r['image_sha256'],r['question']]):r for t in tasks for r in t['evaluation']}
 raw.update({digest([r['image_sha256'],r['question']]):r for r in read(SOURCE/'private/LOCALITY_STRESS_HOLDOUT.json')['rows']})
 scores=read_scores(SOURCE);rows=[]
 for job in read(SOURCE/'QUEUE.json'):
  assert job['status']=='COMPLETE'
  for p in (SOURCE/'jobs'/job['id']).rglob('CONSUMERS.json'):rows.extend(read(p))
 ref={identity(r):r for r in rows if r['arm']=='R0'};neg={identity(r):r for r in rows if r['arm']=='NEG0'};guard={identity(r):r for r in rows if r['arm']=='PLOO'}
 assert ref.keys()==neg.keys()==guard.keys() and len(ref)==989
 assert all(neg[k]['route']==guard[k]['route'] and neg[k]['judge_key']==guard[k]['judge_key'] for k in ref)
 lease=dict(pid=os.getpid(),physical_GPU=physical,GPU_UUID=os.environ['CUDA_VISIBLE_DEVICES'],started_epoch=started,purpose='CACHED_BASE_LATENT_GEOMETRY_ONLY')
 write(ROOT/'ACTIVE_GPU.json',lease);write(ROOT/'RUN_STATUS.json',dict(status='RUNNING',phase='GEOMETRY',epoch=started));metrics=[];cache={}
 try:
  for k,a in ref.items():
   assert time.time()<manifest['deadline_epoch'] and time.time()-started<config['GPU_hours_limit']*3600,'Frozen budget exceeded'
   assert not (ROOT/'STOP').exists(),'User stop'
   b=neg[k];c=guard[k];winner=a['route']['logical_edit_id'];m=dict(mode=a['mode'],prefix=a['prefix'],task=a['task'],input_id=a['input_id'],winner=winner,label=label(a,b,scores))
   if winner is not None:
    e=bank[winner];q=cache.get(a['input_id'])
    if q is None:q=torch.load(SOURCE/'private/keys'/f'{a["input_id"]}.pt',map_location='cpu',weights_only=True).float().reshape(-1).cuda();cache[a['input_id']]=q
    assert torch.isfinite(q).all()
    pos=e['positive'].cuda();d=distances(pos,q,'euclidean');dp=float(d.min());dn=float(distances(e['negative'].cuda(),q,'euclidean').min());quotient=(dn-dp)/(dn+dp+1e-12)
    assert (quotient>=0)==b['route']['activated'],'Cache metric disagrees with executed NEG0'
    radii=e['guard_radii'].cuda();ratios=torch.where(radii>0,d/radii,torch.where(d==0,torch.zeros_like(d),torch.full_like(d,float('inf'))));ratio=float(ratios.min())
    assert bool((d<=radii).any())==bool(c['route_diagnostics'].get('positive_guard'))
    row=raw[a['input_id']];m.update(guard_ratio=ratio if math.isfinite(ratio) else None,quotient=quotient,nearest_background_over_positive=float(distances(background,q,'euclidean').min())/(dp+1e-12),query_to_key_over_radius=float(distances(e['key'].cuda(),q,'euclidean')[0])/e['radius'],same_image=row['image_sha256']==owners[winner]['native']['image_sha256'])
   metrics.append(m)
  write(ROOT/'private/POINTWISE_GEOMETRY.json',metrics)
  pilot=read(SOURCE/'private/PILOT_SELECTION.json')['orders'];checks=read(SOURCE/'private/CHECK_POS.json');review=[]
  for order in pilot:
   for r in checks[str(order)]:review.append(dict(edit_order=order,existing_CHECK_proposal=r,verification_status='UNVERIFIED',verification_evidence=None,reviewer=None,previously_exposed=True,usable_as_independent_confirmation=False,new_CAL_admitted=False))
  write(ROOT/'private/CAL_REVIEW_QUEUE.json',dict(status='UNVERIFIED_REVIEW_ONLY',records=review,no_new_cross_image_records=True,no_new_annotations_created=True))
  public=dict(status='COMPLETE_EXPOSED_DIAGNOSTIC',source_public_commit=config['parent'],consumers=len(metrics),groups=summarize(metrics),PLOO_NEG0_route_changes=0,formal_inputs_used_for_new_router_fit=False,new_classifier_or_threshold_fitted=False,independent_confirmation=False)
  write(ROOT/'public/GEOMETRY_AGGREGATES.json',public)
  readiness=dict(status='NEEDS_VERIFIED_SCOPE_CALIBRATION',pilot_edits=len(pilot),existing_unverified_CHECK_proposals=len(review),verified_isolated_same_image=0,verified_isolated_cross_image=0,minimum_per_edit=dict(same_image=4,cross_image=2),formal_examples_not_repurposed=True,inventory_scope='Existing frozen CHECK/G_SUPPORT/role-audited assets only; not an exhaustive search of all raw datasets',required='Legally role-isolated scope-positive annotations with real verification evidence; candidate proposals and agent automatic rewrites do not qualify')
  write(ROOT/'public/CAL_SCOPE_READINESS.json',readiness)
  positive=public['groups']['POSITIVE_REJECTION'];rescue=public['groups']['NEGATIVE_RESCUE']
  s=f'# Scope 信息诊断结果\n\n989个暴露消费者的固定latent诊断完成；没有训练、生成、Judge或新分类器。PLOO和NEG0路由/输出键均完全相同。\n\n误拒合法正例：{positive["occurrences"]}次出现、{positive["unique_inputs"]}个不同输入；负例救援：{rescue["occurrences"]}次出现、{rescue["unique_inputs"]}个不同输入。出现次数包含多个prefix，不能写成独立样本量。\n\n误拒正例guard覆盖比（<=1才覆盖）：{positive["geometry"]["guard_ratio"]}；NEG0 quotient（>=0接受）：{positive["geometry"]["quotient"]}。误拒正例中同图{positive["same_image_as_winner"]}次、跨图{positive["cross_image_as_winner"]}次。该诊断只解释已有暴露失败，不能证明latent不可分，也不能据此选择新阈值。\n\nCAL仍缺真实verified且正确隔离的标注。已有{len(review)}个CHECK同图待审提案已写入私有review queue，全部UNVERIFIED，不计入新CAL，不声称独立确认，也未创造跨图数据。下一步是取得带验证证据的合法scope标注，再另立计划；当前不继续盲调半径、换expert或消耗GPU。公开只aggregate与代码；原图像、QA、答案、逐题诊断、待审明细保持私有。\n'
  (ROOT/'public/FINAL_RESULTS_ZH.md').write_text(s)
  write(ROOT/'RUN_STATUS.json',dict(status='COMPLETE',phase='CLOSED',next_status=readiness['status'],epoch=time.time()))
 finally:
  lease['ended_epoch']=time.time();lease['resident_seconds']=lease['ended_epoch']-started;write(ROOT/'GPU_RECEIPT.json',lease)
  write(ROOT/'RESOURCE_LEDGER.json',dict(inherited_Judge_attempts=read(SOURCE/'RESOURCE_LEDGER.json')['judge_submission_attempt_items'],new_Judge_attempts=0,training_steps=0,backward_calls=0,optimizer_steps=0,model_backbone_calls=0,new_generation=0,GPU_hours=lease['resident_seconds']/3600))
 audit=dict(status=read(ROOT/'RUN_STATUS.json')['status'],GPU_only=[6,7],GPU_used=[physical],all_GPU_leases_ended=True,training_steps=0,new_Judge_attempts=0,new_generation=0,model_backbone_calls=0,private_pointwise_not_published=True,private_review_not_verified=True,next_status='NEEDS_VERIFIED_SCOPE_CALIBRATION',public_delivery='PENDING_GITHUB',epoch=time.time())
 write(ROOT/'public/FINAL_EXECUTION_AUDIT.json',audit)
 print(json.dumps(audit))
if __name__=='__main__':
 try:main()
 except Exception as e:
  write(ROOT/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),epoch=time.time()));write(ROOT/'RUN_STATUS.json',dict(status='BLOCKED',phase='FAILED_DIAGNOSTIC'));raise
