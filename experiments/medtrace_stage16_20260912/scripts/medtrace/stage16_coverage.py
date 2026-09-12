#!/usr/bin/env python3
"""Frozen-checkpoint coverage extension only; this module has no training path."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(os.environ.get('MEDTRACE_BASELINE_CODE', Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(ROOT))
from scripts.medtrace import stage15 as prior
from scripts.medtrace.stage15_sources import read, write, digest, norm, canonical, KAPPA, MODES
from scripts.medtrace.stage16 import summarize, exact_p, BASELINE


def prepare(old, run, recovered):
    if (run/'private/QUEUE.json').exists(): raise FileExistsError('Coverage queue already frozen')
    source=read(recovered/'RECOVERED.json'); original=read(old/'private/QUEUE.json'); tasks=[]
    for t in original:
        probes=[]
        for p in t['probes']:
            if p['status'] != 'UNSUPPORTED_IMAGE_MISSING' or p['source_image'] not in source['bindings']: continue
            a=source['bindings'][p['source_image']]; path=recovered/'images'/Path(p['source_image']).name
            assert path.is_file() and path.stat().st_size==a['bytes']
            assert a['source_ref']==p['source_image']
            probes.append(dict(p,image_path=str(path),image_sha256=a['sha256'],status='AVAILABLE',source_binding=a))
        if probes: tasks.append(dict(t,probes=probes,U=[],H=[],G=[]))
    assert tasks, 'No new exact supported probes; GPU execution unnecessary'
    cfg=read(old/'private/CAMPAIGN_CONFIG.json')
    cfg.update(stage15_root=str(old),allowed_physical_gpus=[2],worker_gpus=[2],judge_gpu=2,
               train_seconds=24*3600,campaign_epoch=None,kind='STAGE16_COVERAGE_ONLY')
    cfg['gpu_uuids']={'2':subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()}
    write(run/'private/QUEUE.json',tasks); write(run/'private/CAMPAIGN_CONFIG.json',cfg)
    write(run/'private/JUDGE_LOCK.json',read(old/'private/JUDGE_LOCK.json'))
    lock=dict(status='FROZEN_SYSTEM_COVERAGE_EXTENSION',baseline_public_commit=BASELINE,
        queue_sha256=digest(tasks),historical_protocol_sha256=digest(read(old/'public/PROTOCOL_AND_METHOD_LOCK.json')),
        method='Stage14 C_FACT V1 unchanged; this coverage extension loads Stage15 C_NO_H and BE only',
        historical_primary='RC_FIXED_OLD16',secondary='BE_ROUTE_R0',fixed_kappa=KAPPA,
        training_steps=0,checkpoint_reuse='own-edit Stage15 step320 C_NO_H or final BE; stored Base router',
        official_probe_changes='Only missing image_path/binding becomes available. Original question, answer and source-ref unchanged.',
        generation_and_judge='Identical Stage15 actual-input/generation and full-context Judge protocol; no historical rescoring',
        new_probes=len([p for t in tasks for p in t['probes']]),new_edits=len(tasks),
        GPU_authorization=[2],C='UNSUPPORTED_NEW_H_SCOPE_AND_INDEPENDENT_ROLES',D='NOT_RUN_CONDITIONAL_SUPPORT_MISSING',
        RC_EXTCAL_V1='CALIBRATION_UNSUPPORTED; never selected from Stage15 diagnostic curves')
    write(run/'public/PROTOCOL_AND_METHOD_LOCK.json',lock)
    # Audit a new availability/exposure question, not another scan of the exhausted 4490 training rows.
    assets=read(old/'private/ASSET_BINDINGS.json'); selected={t['canonical_edit_id'] for t in original}
    main=read(old.parents[1]/'datasets/MedMKEB/eval_data_threehop_final.json')
    remaining=[r for r in main if canonical(r) not in selected and r['image'] in assets]
    exposed={p['image_sha256'] for t in original for p in t['probes'] if p.get('image_sha256')}
    audit=dict(status='H_SOURCE_LABEL_AND_EDIT_SCOPE_UNSUPPORTED',historical_train_audit_reused=True,
        old_train_rows_rescanned=0,old_SLAKE_pool_rescanned=False,new_H_fit=0,new_H_evaluation=0,new_EXTRA_G=0,
        remaining_native_resolvable_records=len(remaining),
        remaining_native_records_without_known_Stage15_image_overlap=sum(assets[r['image']]['sha256'] not in exposed for r in remaining),
        candidate_status='Availability only; NOT a clean frozen development or confirmation cohort',
        additional_source='PMC-VQA original locality images recovered for evaluation only, never H training',
        source_access='GMAI gated at revision a8c7450a66400adda3f8e93d9cdc7b2fd541a295; no automatic acceptance or bypass',
        gap='No newly verified unused training image + original QA + explicit edit-scope H/G+ + independent fit/calibration/evaluation package',
        GMAI_test_answers='Author card says TEST answers withheld; model pred is not a source gold label',
        outcome_based_selection=False,terms_accepted=False,clinical_review_claimed=False,
        action='Preserve branch-level unsupported state. A/B proceed; do not weaken H loss or reuse evaluation probes for training.')
    write(run/'public/H_SUPPORT_AUDIT_SUMMARY.json',audit)
    write(run/'public/DATA_ROLE_AND_EXPOSURE_SUMMARY.json',dict(status='C_NOT_FROZEN',new_images_role='official_evaluation_only',
        dev_edits_frozen=0,confirmation_edits_frozen=0,heldout_unsealed=False,prior_native_probe_exposure_checked=True,
        fit_calibration_evaluation_isolation='No new training roles assigned; additional H/G/U require source-connected separation',
        candidate_audit=audit,patient_or_study_identity='UNKNOWN'))
    write(run/'public/ROUTER_CALIBRATION_SUMMARY.json',dict(status='CALIBRATION_UNSUPPORTED',threshold_selected=None,
        dev30_frozen=False,missing=['reviewed G_POS','reviewed H','independent U calibration','independent confirmation cohort'],
        old_RC_unchanged=KAPPA,posthoc_A_used_for_calibration=False))
    write(run/'public/COVERAGE_AND_COST.json',dict(status='PREPARED',A='COMPLETE',B='PREPARED',C=audit['status'],D='NOT_RUN',
        restored_images=len(source['bindings']),download_bytes=source['download_bytes'],new_training_steps=0,
        old_image_locality=67,old_image_generality=164,requested_new_image_locality=133,requested_new_image_generality=36,
        still_missing_image_generality=36,public_baseline=BASELINE,publication='PENDING'))
    print('B_FROZEN',len(tasks),lock['new_probes'],flush=True)


def worker(run):
    import torch
    from methods.medtrace import AsymmetricCPExpert, MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter
    from scripts.medtrace.run_realmodel_core import load_real_runtime,LAYER
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');old=Path(cfg['stage15_root']);tasks=read(run/'private/QUEUE.json')
    assert digest(tasks)==read(run/'public/PROTOCOL_AND_METHOD_LOCK.json')['queue_sha256']
    runtime=load_real_runtime(argparse.Namespace(cpu_gate=Path(cfg['runtime']['cpu_gate'])))
    assert not runtime.model.training and not any(p.requires_grad for p in runtime.model.parameters())
    write(run/'private/WORKER_READY.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],model_loaded=True))
    replay=set();began=time.time()
    for t in tasks:
        olddir=old/'private/edits'/f"e{t['order']:03d}"; record=prior.record_for(t)
        runtime.run_root=run/'private/edits'/f"e{t['order']:03d}"
        for method in ('C_NO_H','BE'):
            result_path=runtime.run_root/method/'RESULT.json'
            if result_path.exists(): continue
            hook=editor=expert=cp=None; entries=[]; replays=[]
            checkpoint=olddir/method/('latest.pt' if method=='C_NO_H' else 'editor_state.pt')
            try:
                if method=='C_NO_H':
                    state=torch.load(checkpoint,map_location=runtime.device,weights_only=True)
                    assert state['step']==320 and state['condition']==method and state['seed']==t['seed'] and state['canonical_edit_id']==record.record_id
                    cp=AsymmetricCPExpert(14336,4096,4).to(runtime.device)
                    expert=LowRankExpert(cp,t['seed'],rank=4).to(runtime.device);expert.load_state_dict(state['expert']);expert.requires_grad_(False)
                    del state
                    assert sum(p.numel() for p in expert.parameters())==73728
                    router=MemoryRouter.from_state(torch.load(olddir/'ROUTER.pt',map_location='cpu',weights_only=True),device=runtime.device)
                    hook=MedTraceLayerHook(runtime.get_module(LAYER),expert);hook.attach()
                else:
                    editor=BalanceEditPaperSpecEditor(runtime);base_module,target=editor.wrapper.base,editor.target
                    editor.load_editor_state(checkpoint);router=editor.router
                assert router.logical_ids==[record.record_id]
                for probe in t['probes']:
                    prior.budget(run,cfg)
                    if hook:hook.clear_request_routing()
                    base,raw,batch,eqkey=prior.base_output(runtime,run,probe,record)
                    key=runtime.extract_layer_input_key(batch,module_path=runtime.target_lock['balancedit']['targets'][0],pooling='mean')
                    route=asdict(router.route(key))
                    from scripts.medtrace.stage4_scope import accepted
                    rc=accepted(route,KAPPA)
                    with editor._activated(record.record_id) if editor else nullcontext():
                        forced=prior.generate(runtime,raw,base['binding'],hook)
                    entry=dict(probe=probe,status='COMPLETE',eqkey=eqkey,base=base,forced=forced,route=route,rc_on=rc,
                        checkpoint_binding=dict(canonical_edit_id=record.record_id,method=method,path=str(checkpoint),
                            stage15_public_commit=BASELINE,stage15_protocol=read(run/'public/PROTOCOL_AND_METHOD_LOCK.json')['historical_protocol_sha256']))
                    entries.append(entry)
                    for mode,on in (('BE_ROUTE_R0',route['activated']),('RC_FIXED_OLD16',rc)):
                        signature=(method,probe['image_path'] is None,bool(on))
                        if signature in replay:continue
                        with editor._activated(record.record_id if on else None) if editor else nullcontext():
                            actual=prior.generate(runtime,raw,base['binding'],hook if on else None)
                        expected=forced if on else base
                        assert actual['raw_token_ids']==expected['raw_token_ids'] and actual['raw_answer']==expected['raw_answer']
                        replay.add(signature);replays.append(dict(mode=mode,signature=signature,passed=True))
                write(result_path,dict(status='COMPLETE',canonical_edit_id=record.record_id,method=method,entries=entries,replay=replays))
            finally:
                if hook:hook.detach()
                if editor:editor.reset_editor_state();runtime.replace_module(target,base_module)
                del hook,editor,expert,cp
                gc.collect();torch.cuda.empty_cache()
            assert runtime.base_guard.verify()['unchanged']
            print('COVERAGE_DONE',t['order'],method,len(entries),flush=True)
    write(run/'private/WORKER_COMPLETE.json',dict(status='COMPLETE',seconds=time.time()-began,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),training_steps=0))


def prepare_judge(run):
    old=Path(read(run/'private/CAMPAIGN_CONFIG.json')['stage15_root'])
    protocol=read(run/'private/JUDGE_LOCK.json')['config_sha256'];historical=read(old/'private/judge/SIDECAR.json')['bindings']
    old_verdict={j['opaque_query_id']:j for j in map(json.loads,(old/'private/judge/OUTPUT.jsonl').open())}
    packets={};side={};reuse={}
    for _,method,e in prior.all_results(run):
        for field in ('base','forced'):
            key,kind,full=prior.judge_identity(e,e[field],protocol)
            side[key]=full
            if key in historical:
                assert historical[key]==full;reuse[key]=old_verdict[key];continue
            packets[key]=dict(opaque_query_id=key,adjudication_pass=kind,question=e['probe']['question'],gold_answer=e['probe']['reference'],raw_base_answer=e[field]['raw_answer'])
    path=run/'private/judge/PACKET.jsonl';path.parent.mkdir(exist_ok=True)
    with path.open('x') as f:
        for k in sorted(packets):f.write(json.dumps(packets[k],ensure_ascii=False)+'\n')
    write(path.parent/'SIDECAR.json',dict(bindings=side,protocol_sha256=protocol,new=len(packets),reused=len(reuse)))
    write(path.parent/'REUSED_VERDICTS.json',reuse)
    print('B_JUDGE_NEW',len(packets),'REUSED',len(reuse),flush=True)


def report(run):
    old=Path(read(run/'private/CAMPAIGN_CONFIG.json')['stage15_root'])
    path=run/'private/judge';side=read(path/'SIDECAR.json');verdict=read(path/'REUSED_VERDICTS.json')
    if (path/'OUTPUT.jsonl').exists():
        for j in map(json.loads,(path/'OUTPUT.jsonl').open()):
            assert j['parse_valid'] and type(j['is_correct']) is bool and j['opaque_query_id'] not in verdict
            verdict[j['opaque_query_id']]=j
    assert set(verdict)==set(side['bindings']), 'Incomplete new Judge coverage'
    rows=[];replays=[];paired_inputs={}
    for t,m,e in prior.all_results(run):
        p=e['probe'];b=e['base'];f=e['forced']
        keys=[prior.judge_identity(e,o,side['protocol_sha256'])[0] for o in (b,f)]
        assert b['binding']==f['binding'] and b['binding']['image_source_sha256']==p['image_sha256']
        bound=(p,b['binding'],b['raw_answer'],b['raw_token_ids'],e['route'])
        pair_id=(t['order'],p['probe_id'])
        if pair_id in paired_inputs:assert paired_inputs[pair_id]==bound
        else:paired_inputs[pair_id]=bound
        rows.append(dict(edit=t['order'],probe=p['probe_id'],method=m,metric=p['metric'],subtype='ALL',
            r0=e['route']['activated'],rc=e['rc_on'],base_exact=norm(b['raw_answer'])==norm(p['reference']),
            forced_exact=norm(f['raw_answer'])==norm(p['reference']),base_semantic=verdict[keys[0]]['is_correct'],
            forced_semantic=verdict[keys[1]]['is_correct'],forced_preserve=norm(f['raw_answer'])==norm(b['raw_answer']),
            base_target_copy=norm(b['raw_answer'])==norm(t['raw_record']['alt']),forced_target_copy=norm(f['raw_answer'])==norm(t['raw_record']['alt']),
            base_cap=b['cap_hit'],forced_cap=f['cap_hit']))
    tasks=read(run/'private/QUEUE.json');assert len(rows)==sum(len(t['probes']) for t in tasks)*2
    # Old and new stay separately identifiable; only tables are combined, never relabelled as independent confirmation.
    oldrows=read(run/'private/A_BOUND_ROWS.json'); tables=[];effects=[]
    for cohort,rs in [('OLD_SUPPORTED',oldrows),('NEWLY_RESTORED',rows),('COMBINED_COVERAGE',oldrows+rows)]:
        for metric in ('I_Locality','I_Generality'):
            for method in ('C_NO_H','BE'):
                subset=[r for r in rs if r['metric']==metric and r['method']==method]
                if not subset:continue
                for mode in ('BASE',*MODES):
                    on=[mode=='FORCED_ON' or mode=='BE_ROUTE_R0' and r['r0'] or mode=='RC_FIXED_OLD16' and r['rc'] for r in subset]
                    for score in ('exact','semantic'):
                        tables.append(dict(cohort=cohort,metric=metric,method=method,mode=mode,score=score,
                            **summarize(subset,on,score),status='FROZEN_SYSTEM_COVERAGE_EXTENSION'))
            for mode in MODES:
                for score in ('exact','semantic'):
                    pairs=defaultdict(dict)
                    for r in rs:
                        if r['metric']!=metric:continue
                        on=mode=='FORCED_ON' or mode=='BE_ROUTE_R0' and r['r0'] or mode=='RC_FIXED_OLD16' and r['rc']
                        value=(r['forced_preserve'] if on else True) if metric=='I_Locality' and score=='exact' else r[('forced_' if on else 'base_')+score]
                        pairs[r['edit'],r['probe']][r['method']]=value
                    assert all(set(p)=={'C_NO_H','BE'} for p in pairs.values())
                    delta=[int(p['C_NO_H'])-int(p['BE']) for p in pairs.values()]
                    if not delta:continue
                    lo,hi=prior.bootstrap(delta);wins=delta.count(1);losses=delta.count(-1)
                    effects.append(dict(cohort=cohort,metric=metric,mode=mode,score=score,n=len(delta),delta=mean(delta),
                        ci_low=lo,ci_high=hi,NOH_only=wins,BE_only=losses,exact_two_sided_p=exact_p(wins,losses),
                        status='COVERAGE_EXTENSION_NOT_INDEPENDENT_CONFIRMATION'))
    prior.csv_write(run/'public/COVERAGE_EXTENSION_RESULTS.csv',tables)
    prior.csv_write(run/'public/PAIRED_EFFECTS.csv',effects)
    for t in tasks:
        for m in ('C_NO_H','BE'):replays+=read(run/'private/edits'/f"e{t['order']:03d}"/m/'RESULT.json')['replay']
    cost=read(run/'public/COVERAGE_AND_COST.json');cost.update(status='A_B_COMPUTE_COMPLETE_C_D_UNSUPPORTED',B='COMPLETE',
        new_completed_writer_probes=len(rows),new_judgments=side['new'],reused_judgments=side['reused'],actual_routing_replays=replays,
        text_only_replay='Not naturally present in newly recovered image-only probes; Stage15 text-only binding checked by A',
        worker=read(run/'private/WORKER_COMPLETE.json'),process_intervals=read(run/'private/PROCESS_INTERVALS.json'),publication='PENDING')
    write(run/'public/COVERAGE_AND_COST.json',cost)
    write(run/'RUN_COMPLETION.json',dict(status=cost['status'],publication='PENDING',new_training_steps=0,
        C='UNSUPPORTED_NEW_H_SCOPE_AND_INDEPENDENT_ROLES',D='NOT_RUN_CONDITIONAL_SUPPORT_MISSING'))
    lines=['# MedTRACE Stage16 — frozen coverage extension and bounded closeout','',
        'A and B compute completed on the supported inputs. Stage15 results/checkpoints/thresholds/Judge verdicts were not modified.', '',
        f'B restored {cost["restored_images"]} exact author PMC-VQA locality images, evaluated {len(rows)} writer-probes from saved C_NO_H/BalancEdit checkpoints, and obtained {side["new"]} new full-context judgments. No training was performed.', '',
        'COVERAGE_EXTENSION_RESULTS.csv and PAIRED_EFFECTS.csv separate OLD_SUPPORTED, NEWLY_RESTORED and COMBINED_COVERAGE. This is the same official cohort with expanded coverage, not new independent method confirmation.', '',
        'The 36 missing cross-image-general probes remain unsupported because their GMAI source is gated. No access terms were accepted. GMAI TEST labels are withheld by the author; pred was not promoted to clinical source gold.', '',
        'C lacks newly verified H/G+ scope and independent fit/calibration/evaluation source packages. No 30/100 cohort or RC_EXTCAL_V1 was frozen, no four-arm pilot was run, and no confirmation result is claimed. New PMC locality probes remain evaluation-only.', '',
        'See STAGE16_A_READONLY_REPORT.md for the unchanged historical routing trade-off and supplemental exact/source-cluster diagnostics. Post-hoc thresholds never replace the historical primary.', '',
        'Source and aggregate release only. Images, QA, raw outputs/tokens, per-item Judge mappings, environment paths and checkpoints remain private. Patient independence and near-duplicate closure are not established.', '',
        'Sources: [MedMKEB release](https://github.com/pkusixspace/MedMKEB/tree/d9f38639ec2285a0e9f541e22156ec14f87271d8), [PMC-VQA author card and license](https://huggingface.co/datasets/RadGenome/PMC-VQA/blob/b56ae594f794867893143b337b4118a835794647/README.md), [GMAI access and withheld TEST answers](https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench).','']
    (run/'public/GPT_PRO_REVIEW.md').write_text('\n'.join(lines))
    print('B_REPORT_COMPLETE',len(rows),flush=True)


def launch(run):
    from scripts.medtrace.neutral_entrypoint import neutral_command
    from scripts.medtrace import coordinate_selective_write as common
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');assert cfg['allowed_physical_gpus']==[2]
    if (run/'PIPELINE_PIDS.json').exists():raise FileExistsError('Already launched; inspect before resume')
    common.GPUS.clear();common.GPUS.update(cfg['gpu_uuids'])
    write(run/'private/GPU_START_CHECK.json',common.gpu_check('2'))
    cfg['campaign_epoch']=time.time();write(run/'private/CAMPAIGN_CONFIG.json',cfg)
    temporary=Path(tempfile.mkdtemp(prefix='job.'));shutil.copyfile(ROOT/'scripts/medtrace/neutral_entrypoint.py',temporary/'main.py')
    cmd,env=neutral_command([sys.executable,str(Path(__file__).resolve()),'coordinate','--run-root',str(run)],
        dict(os.environ,JOB_ENTRYPOINT=str(temporary/'main.py'),CUDA_VISIBLE_DEVICES=''),'main')
    with (run/'pipeline.log').open('x') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(run/'PIPELINE_PIDS.json',dict(coordinator=p.pid,argv=cmd));print('DETACHED',p.pid,flush=True)


def coordinate(run):
    from scripts.medtrace.neutral_entrypoint import neutral_command
    from scripts.medtrace import coordinate_selective_write as common
    cfg=read(run/'private/CAMPAIGN_CONFIG.json');common.GPUS.clear();common.GPUS.update(cfg['gpu_uuids'])
    intervals={}
    try:
        for action in ('worker','judge'):
            if action=='judge':
                prepare_judge(run)
                if not read(run/'private/judge/SIDECAR.json')['new']:continue
            env=common.environment('2',judge=action=='judge');env['JOB_ENTRYPOINT']=os.environ['JOB_ENTRYPOINT']
            cmd,env=neutral_command([common.JUDGE_PYTHON if action=='judge' else sys.executable,str(Path(__file__).resolve()),action,'--run-root',str(run)],env,'job' if action=='judge' else 'run')
            began=time.time()
            with (run/(action+'.log')).open('x') as f:
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                write(run/'PIPELINE_PIDS.json',dict(coordinator=os.getpid(),active_action=action,child=p.pid,argv=cmd))
                rc=p.wait()
            intervals[action]=dict(start=began,end=time.time(),exit=rc,gpu=2)
            write(run/'private/PROCESS_INTERVALS.json',intervals)
            if rc:raise RuntimeError(action+' failed; preserve completed results and do not start another branch')
        report(run)
    except Exception as error:
        write(run/'PIPELINE_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','launch','coordinate','worker','judge','report'))
    p.add_argument('--stage15-root',type=Path);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--recovered',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.stage15_root,a.run_root,a.recovered)
    elif a.action=='judge':prior.judge(a.run_root)
    else:globals()[a.action](a.run_root)
