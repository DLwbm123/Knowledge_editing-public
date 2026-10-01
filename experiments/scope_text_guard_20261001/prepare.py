"""Independent router-only state; all historical expert/model assets are read-only."""
import os,json,shutil,time,subprocess
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT']);OLD=Path(os.environ['PR9_ROOT'])
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
    if not (ROOT/'RESOURCE_LEDGER.json').exists():
        for name in ['source','source_patch','models']:(ROOT/name).symlink_to((OLD/name).resolve(),target_is_directory=True)
        for name in ['worker_v3.py','router_r3.py','structures.py','hard_pool.py','CONFIG_LOCK.json']:
            shutil.copy2(OLD/name,ROOT/name)
        for name in ['TASKS_R2_LOCKED.json','BASE_MASKS.json','U_bg.json','LOCALITY_STRESS_HOLDOUT.json','INITIALIZERS.json','CHECK_POS.json','G_SUPPORTS.json','PILOT_SELECTION.json']:
            shutil.copy2(OLD/'private'/name,ROOT/'private'/name)
        for name in ['private/base','private/keys','private/black','private/hard','private/judge_sol']:
            shutil.copytree(OLD/name,ROOT/name)
        for name in ['jobs','logs','private/references']:(ROOT/name).mkdir(parents=True,exist_ok=True)
        prior=read(OLD/'RESOURCE_LEDGER.json');assert all(s.get('ended_epoch') for s in prior['gpu_sessions']) and all(a['status']!='RESERVED' for a in prior['judge_attempts'])
        write(ROOT/'RESOURCE_LEDGER.json',dict(historical_judge_attempts=prior['judge_submission_attempt_items'],current_judge_attempts=0,historical_judge_attempt_count=len(prior['judge_attempts']),judge_submission_attempt_items=prior['judge_submission_attempt_items'],judge_submission_attempt_items_limit=prior['judge_submission_attempt_items']+500,judge_attempts=prior['judge_attempts'],physical_requests=prior['physical_requests'],physical_requests_limit=None,historical_gpu_seconds=0,current_gpu_seconds=0,lifetime_gpu_seconds=0,gpu_sessions=[]))
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True);devices=[]
        for row in text.strip().splitlines():
            i,u,f=[x.strip() for x in row.split(',')]
            if int(i) in [6,7]:devices.append(dict(index=int(i),uuid=u,free_MiB=int(f)))
        assert len(devices)==2
        write(ROOT/'GPU_BINDINGS.json',dict(devices=devices,authorization='GPU6/7 only'))
        policy=dict(quotas_enabled=True,soft_bytes=10*1024**3,hard_bytes=20*1024**3,min_free_bytes=8*1024**3,checkpoint_bytes=0,teacher_bytes=0,final_total_bytes=16*1024**3,final_model_bytes=0,control_reserved_bytes=2*1024**3,environment_reserved_bytes=0,emergency_bytes=256*1024**2,local_scorer_reserved_bytes=512*1024**2,external_owned_roots=[])
        write(ROOT/'STORAGE_POLICY.json',policy)
    else:
        state=read(ROOT/'RESOURCE_LEDGER.json')
        assert not (ROOT/'RUN_MANIFEST.json').exists() and not state['gpu_sessions'] and state['current_judge_attempts']==0,'Only incomplete zero-compute initialization may resume'
    write(ROOT/'PREDECESSOR.json',dict(read(OLD/'PREDECESSOR.json'),prior_stage_root=str(OLD)))
    from judge_protocol import digest,read_scores,NAMESPACE
    config=read(ROOT/'public/ROUTER_CONFIG.json');config['id']=digest(config);write(ROOT/'SCIENCE_LOCK.json',config)
    assert len(read_scores(ROOT))==1336
    assert len(read(ROOT/NAMESPACE/'JUDGE_MISSING_LOCK.json')['keys'])==41
    tasks=read(ROOT/'private/TASKS_R2_LOCKED.json')['tasks'];bg=read(ROOT/'private/U_bg.json')['rows'];old47=read(ROOT/'private/LOCALITY_STRESS_HOLDOUT.json')['rows']
    inp=lambda r:(r['image_sha256'],r['question'])
    panels=dict(DEV=[r for t in tasks[:24] for r in t['evaluation']],REG=[r for t in tasks[24:] for r in t['evaluation']],old47=old47)
    overlaps={k:dict(input=len(set(map(inp,bg))&set(map(inp,rs))),source=len({r['source_group'] for r in bg}&{r['source_group'] for r in rs})) for k,rs in panels.items()}
    assert all(not d['input'] and not d['source'] for d in overlaps.values()),'U_bg formal/source overlap: no router construction allowed'
    formal={inp(r) for rs in panels.values() for r in rs}
    assert len(tasks)==48 and all(len(t['semantic_fit_questions'])==4 for t in tasks)
    assert all((t['native']['image_sha256'],q) not in formal for t in tasks for q in t['semantic_fit_questions'])
    assert all(r['purpose']=='U_bg' and r['scope']=='negative' for r in bg)
    audit=dict(status='PASS',tasks=48,U_bg=len(bg),U_bg_sources=len({r['source_group'] for r in bg}),overlaps=overlaps,positive='native + exactly four S_fit',S_fit_formal_input_overlap=0,native_T0_role_overlap='authorized same native input; not added formal support',formal_outcomes_used_for_router=False,model_layer_precision_backend_weights_bank_order_frozen=True,DEV='EXPOSED_DIAGNOSTIC',REG='EXPOSED_REGRESSION',old47='EXPOSED_REGRESSION',epoch=time.time())
    write(ROOT/'public/DATA_ROLE_AUDIT.json',audit)
    write(ROOT/'RUN_MANIFEST.json',dict(started_epoch=(ROOT/'RESOURCE_LEDGER.json').stat().st_mtime,target_hours=[0,6],deadline_epoch=(ROOT/'RESOURCE_LEDGER.json').stat().st_mtime+6*3600,GPU_hours_limit=4,new_Judge_items_limit=500,finite_plan=True,training_steps=0))
    write(ROOT/'RUN_STATUS.json',dict(status='READY_FOR_LATENT_AUDIT',phase='P1'))
    write(ROOT/'QUEUE.json',[])
    print(json.dumps(dict(status='DATA_ROLE_PASS',overlaps=overlaps,devices=read(ROOT/'GPU_BINDINGS.json')['devices'],inherited_attempts=read(ROOT/'RESOURCE_LEDGER.json')['historical_judge_attempts'])))
if __name__=='__main__':main()
