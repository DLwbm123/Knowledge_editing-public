import os,sys,json,sqlite3,statistics
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
import scoped as e
import torch
rows=[]
for t in e.selected():
 init=e.p.load_state(e.prepared(t))['expert'];a=torch.einsum('aib,bjc->aijc',init['G3'].double(),init['G4'].double()).reshape(4,14336);b0=torch.einsum('aib,bjc->aijc',init['G1'].double(),init['G2'].double()).reshape(4096,4)
 assert torch.allclose(a@a.T,torch.eye(4,dtype=torch.float64),atol=1e-5,rtol=1e-5)
 for arm in e.ARMS:
  final=e.p.load_state(e.node(t,arm,192))['expert'];assert all(torch.equal(init[k],final[k]) for k in ['G1','G3','G4'])
  delta=final['G2'].double()-init['G2'].double();db=torch.einsum('aib,bjc->aijc',init['G1'].double(),delta).reshape(4096,4)
  gram=a@a.T;norm=torch.einsum('ij,jk,ik->',db,gram,db).clamp_min(0).sqrt();base=torch.einsum('ij,jk,ik->',b0,gram,b0).sqrt()
  assert norm>0 and torch.isfinite(norm)
  rows.append(dict(expert_order=t['order'],arm=arm,G2_relative_change=float(delta.norm()/init['G2'].double().norm()),effective_write_relative_change=float(norm/base),effective_write_frobenius=float(norm)))
db=sqlite3.connect('file:'+str(e.RUN/'private/judge_scoped_astra_medium/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
records={(x['method'],x['mode'],x['query_id']):e.c.read(x['path']) for x in db.execute('select * from consumer')}
routes=e.c.read(e.RUN/'private/ROUTES.json');chosen={t['edit_id'] for t in e.selected()};main={q for q in e.p.queries(146) if routes[q]['effective_expert'] in chosen};checks=e.c.read(e.RUN/'private/SOURCE_CHECK_ROWS.json');forced={q['query_id'] for q in checks if e.p.tasks()[q['forced_expert_index']]['edit_id'] in chosen};natural={q['query_id'] for q in checks if routes[q['query_id']]['effective_expert'] in chosen};comparisons=[]
for arm in e.ARMS:
 for mode,qs in [('bank_R0',main),('forced_source_CHECK',forced),('natural_source_CHECK',natural)]:
  label=e.label(arm,192)+('_NAT' if mode=='natural_source_CHECK' else '');w0='W0_NAT' if mode=='natural_source_CHECK' else 'W0';pairs=[(records[w0,mode,q]['R0']['raw_token_ids'],records[label,mode,q]['R0']['raw_token_ids']) for q in qs]
  comparisons.append(dict(arm=arm,mode=mode,n=len(pairs),equal_W0_tokens=sum(a==b for a,b in pairs),W0_at1024=sum(len(a)==1024 for a,b in pairs),new_at1024=sum(len(b)==1024 for a,b in pairs)))
summary={arm:{k:statistics.median(x[k] for x in rows if x['arm']==arm) for k in ['G2_relative_change','effective_write_relative_change','effective_write_frobenius']} for arm in e.ARMS}
out=dict(status='PASS',all40_G2_changed=True,all_fixed_cores_exact=True,weight_changes=rows,medians=summary,matched_outputs=comparisons,new_GPU_work=0,new_Judge=0,interpretation='Descriptive audit after results, not a preregistered success test')
e.c.write(e.RUN/'public/POSTHOC_MECHANISM_AUDIT.json',out);print(json.dumps(dict(medians=summary,matched_outputs=comparisons),indent=2))
