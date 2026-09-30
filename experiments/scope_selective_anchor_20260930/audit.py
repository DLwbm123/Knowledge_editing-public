"""Read-only predecessor audit; write only the explicitly new run namespace."""
import os, json, hashlib, math, random, shutil, time
from pathlib import Path
import torch
OLD = Path(os.environ['PREDECESSOR_ROOT']).resolve()
ROOT = Path(os.environ['RUN_ROOT']).resolve()

def read(p): return json.loads(p.read_text())
def write(p, x):
    assert p.resolve().is_relative_to(ROOT) and not p.resolve().is_relative_to(OLD)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ident(r): return (r['image_sha256'], ' '.join(r['question'].casefold().split()).rstrip('?.。？'))

def main():
    assert ROOT != OLD and not ROOT.is_relative_to(OLD)
    ROOT.mkdir(parents=True, exist_ok=True)
    probe = ROOT / '.write_probe'
    probe.write_text('probe'); assert probe.read_text() == 'probe'; probe.unlink()
    assert shutil.disk_usage(ROOT).free > 28 * 1024**3
    tasks = read(OLD/'private/TASKS_R2_LOCKED.json')['tasks']
    assert len(tasks)==48 and all(len(t['semantic_fit_questions'])==4 for t in tasks)
    inits = read(OLD/'private/INITIALIZERS.json')
    ledger = read(OLD/'STORAGE_LEDGER.json')
    origins = read(OLD/'private/INIT_BINDINGS.json')
    support_path=OLD/read(OLD/'SUPPORT_REPAIR.json')['repaired_path']
    gs={x['order']:x for x in read(support_path)}
    audited=[]; controls=[]
    for t, meta in zip(tasks, inits, strict=True):
        o=t['order']; assert meta['order']==o
        support=dict(native=t['native'],P=t['semantic_fit_questions'],Uold=t['U_fit'],Unew=t['U_new'],G=gs[o])
        assert hashlib.sha256(json.dumps(support,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==meta['binding']['input_target_support']
        original=OLD/f'adapters/s20260929/E_orig/e{o:03d}.pt'
        d=torch.load(original,map_location='cpu',weights_only=False)
        b=d['binding']
        assert d['step']==0 and b['origin']=='INIT_POST80' and b['initialization_sha256']==meta['sha256']
        assert b['edit']==t['canonical_edit_id'] and d['kind']=='LR'
        assert sha(original)==ledger['artifacts'][str(original.relative_to(OLD))]['hash']
        assert set(d['expert'])=={'A','B'} and all(torch.isfinite(v).all() for v in d['expert'].values())
        origin=next((x for x in origins if x['order']==o),None)
        source=Path(origin['path']) if origin else OLD.parents[1]/f'appendix-20260929/run/adapters/s20260927/M1/e{o:03d}.pt'
        tensor_match=None
        if source and source.exists():
            assert sha(source)==meta['sha256']
            prior=torch.load(source,map_location='cpu',weights_only=False)
            tensor_match=all(torch.equal(d['expert'][k],prior['expert'][k]) for k in d['expert'])
            assert tensor_match
        else:
            raise RuntimeError(f'Original initializer provenance unavailable for edit {o}; cannot verify tensor equality')
        c=read(OLD/f'private/scales/{o}.json'); assert c['initial_sha256']==meta['sha256'] and math.isfinite(c['c_i']) and c['c_i']>0
        hard=read(OLD/f'private/hard/{o}.json'); assert hard['frozen_before_new_outputs'] and hard['rows']
        for m in ['A0','AH','AHS_01']:
            p=OLD/f'adapters/s20260929/{m}/e{o:03d}.pt'; x=torch.load(p,map_location='cpu',weights_only=False); cb=x['binding']
            assert sha(p)==ledger['artifacts'][str(p.relative_to(OLD))]['hash']
            assert x['step']==80 and cb['actual_step']==80 and cb['initialization_sha256']==meta['sha256'] and cb['fresh_optimizer']
            assert cb['seed']==20260929 and cb['edit']==b['edit']
            assert all(cb['implementation'][n]==sha(OLD/n) for n in cb['implementation'])
            controls.append(dict(order=o,method=m,weight_verified=True,binding_verified=True))
        audited.append(dict(order=o,relative_path=str(original),sha256=sha(original),origin_sha256=meta['sha256'],tensor_equality=tensor_match,origin_binding=meta['binding'],c_minus=c['c_i'],hard_count=len(hard['rows'])))
    write(ROOT/'private/INITIALIZERS.json',audited)
    write(ROOT/'public/INIT_AND_REUSE_AUDIT.json',dict(status='CPU_BINDINGS_PASS_GPU_PARITY_PENDING',common_initializers=len(audited),original_tensor_equality=sum(x['tensor_equality'] for x in audited),controls=controls,reference_origin='INIT_POST80 via immutable E_orig carrier',low_lambda_reusable=0,seed=20260929,CE_sampling='native .5 plus one of four S_fit .5, shuffled once with derive_seed(edit,20260929)+1',U_sampling='old exact negative_slot, inherited hard list order',c_minus='exact inherited per-edit scalar',GPU_parity_required=True))
    cp=read(OLD/'private/CHECK_POS.json'); cn=read(OLD/'private/CHECK_NEG.json')['rows']; bg=read(OLD/'private/U_bg.json')['rows']
    formal={ident(r) for t in tasks for r in t['official_evaluation_full']+t['evaluation']}
    fit={ident(t['native']) for t in tasks}|{ident(dict(t['native'],question=q)) for t in tasks for q in t['semantic_fit_questions']}
    fit |= {ident(r) for t in tasks for key in ['U_fit','U_new'] for r in t[key]}
    checkrows=[r for rows in cp.values() for r in rows]+cn
    assert not ({ident(r) for r in checkrows}&(fit|formal))
    assert not ({ident(r) for r in bg}&formal)
    assert all(r['purpose']=='U_bg' and r['scope']=='negative' and r.get('scope_basis') for r in bg)
    write(ROOT/'public/DATA_ROLE_AUDIT.json',dict(status='PASS',positive_anchor='current edit native + exactly four S_fit only',CHECK='exposed selection-only; no gradients/mining',CHECK_positive=sum(len(v) for v in cp.values()),CHECK_negative=len(cn),U_bg=len(bg),U_bg_sources=len({r['source_group'] for r in bg}),fit_CHECK_overlap=0,formal_CHECK_overlap=0,formal_U_bg_overlap=0,DEV='EXPOSED_DIAGNOSTIC',REG='EXPOSED_REGRESSION',old47='EXPOSED_REGRESSION',CAL_positive=0,gate_fitting='UNSUPPORTED_CAL_SCOPE',cross_image_anchor_guarantee=False))
    prior=read(OLD/'RESOURCE_LEDGER.json')
    assert not any(not s.get('ended_epoch') for s in prior['gpu_sessions'])
    assert all(a['status'] in ['FORMAT_VALID','FAILED_NO_RETRY'] for a in prior['judge_attempts'])
    sel=read(OLD/'private/PILOT_SELECTION.json')['orders'];assert sel==[4,5,6,11,12,15,16,21]
    n=sum(len(t['evaluation']) for t in tasks if t['order'] in sel);n4=sum(len(t['evaluation']) for t in tasks if t['order'] in sel[:4]);checks=sum(len(cp[str(o)])+len(cn) for o in sel)
    write(ROOT/'public/JUDGE_RESERVATION.json',dict(status='WAITING_FOR_JUDGE_RESERVATION',inherited_actual_attempts=prior['judge_submission_attempt_items'],historical_waiver_scope=read(OLD/'NEW_JUDGE_AUTHORIZATION.json')['scope'],pilot_upper_bound=5*(2*n+n4+checks)+3*checks,smoke_separate=True,unknown_dedup_discount=0,new_authorization=None))
    write(ROOT/'public/RUN_PROTOCOL.json',dict(status='PLANNED_CPU_AUDITED',seed=20260929,steps=80,layer='model.layers.30.mlp.down_proj',rank=4,router='R0 immutable nearest-first',beta=[.1,1.],gamma=[.25,.5,.75,1.],pilot=sel,DEV_new_continuations=88,REG_new_continuations_if_admitted=72,GPU=[5,6,7],soft_bytes=10*1024**3,hard_bytes=20*1024**3,old_artifacts='read-only',hourly_monitor=False,REG_gate='at least one frozen preregistered candidate clear joint PASS',independent_CONFIRM=False))
    print(json.dumps(dict(initializers=len(audited),controls=len(controls),data='PASS',judge_attempts=prior['judge_submission_attempt_items'],pilot_reservation=5*(2*n+n4+checks)+3*checks)))
if __name__=='__main__': main()
