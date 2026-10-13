"""Keep the declared next consumers available; never infer qualification from coverage."""
import json
import os
from pathlib import Path
import time
import traceback

UP=Path(os.environ['RUN_ROOT']).resolve()
NEXT=Path(os.environ['FOLLOWUP_ROOT']).resolve()


def read(path):return json.loads(path.read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def gate(result):
    expected={'M1_source_and_generalization','M3_damage','natural_M1_preserved','natural_M3_damage'}
    return (result.get('status')=='COMPLETE' and result.get('missing_consumers')==0
            and result.get('gates',{}).get('complete') is True
            and set(result.get('compatibility',{}))==expected
            and all(result['compatibility'][key] is True for key in expected))


def selfcheck():
    keys=('M1_source_and_generalization','M3_damage','natural_M1_preserved','natural_M3_damage')
    good=dict(status='COMPLETE',missing_consumers=0,gates=dict(complete=True),compatibility={k:True for k in keys})
    assert gate(good)
    for key in keys:
        bad=json.loads(json.dumps(good));bad['compatibility'][key]=False;assert not gate(bad)
    for field,value in [('missing_consumers',1),('status','PARTIAL')]:
        assert not gate(dict(good,**{field:value}))
    assert not gate(dict(good,compatibility={}))
    return dict(status='PASS',all_frozen_compatibility_gates_required=True,missing_and_partial_rejected=True)


def retain(source,destination):
    assert source.is_file() and not source.is_symlink() and UP in source.resolve().parents
    destination.parent.mkdir(parents=True,exist_ok=True)
    assert NEXT in destination.parent.resolve().parents
    if not destination.exists():os.link(source,destination)
    a,b=source.stat(),destination.stat()
    assert (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)
    return dict(source=str(source),destination=str(destination),device=b.st_dev,inode=b.st_ino,bytes=b.st_size)


def preflight():
    env=read(UP/'private/LAUNCH_ENV.json')
    original=read(Path(env['PROTECTION_PARENT'])/'private/BENCHMARK146_QUEUE.json')['tasks']
    current=read(Path(env['SUPPLEMENT_PARENT'])/'private/FROZEN_TASKS.json')
    anchors=read(Path(env['SUPPLEMENT_PARENT'])/'private/FROZEN_ANCHORS.json')
    assert len(original)==146 and len({t['edit_id'] for t in original})==146
    assert all(len(t['fit_questions'])==4 for t in original)
    matched=[t['order'] for t,s in zip(original,current) if all(t[k]==s[k] for k in ('native','fit_questions','seed'))]
    available={a['owner'] for a in anchors}
    reusable=set(matched)&available
    assert len(matched)==7 and len(reusable)==7
    write(NEXT/'private/ORIGINAL146_TASKS.json',original)
    write(NEXT/'private/EXISTING_SOURCE_BINDINGS.json',dict(tasks=current,anchors=anchors,
        exact_original_task_matches=matched,possible_reuse_only_after_complete_source_binding=True))
    audit=dict(status='SOURCE_ADMISSION_PENDING',original_edits=146,same_image_FIT_questions=584,
        existing_GFIT_coverage_owners=len(available),original_scope_compatible_owners=len(reusable),
        owners_requiring_source_qualification=146-len(reusable),current_owner8_version_mismatch=True,
        prior_T2L_diagnostic_probe_count=54,prior_T2L_diagnostic_edit_count=31,
        new_model_calls=0,new_Judge=0,training_ready=False,independent_confirmation=False)
    write(NEXT/'public/SOURCE_PREFLIGHT.json',audit)
    return audit


def main():
    assert NEXT!=UP and NEXT.parent.parent==UP.parent.parent
    with (NEXT/'private/WATCH_ONCE').open('x'):pass
    selfcheck()
    receipt=dict(status='NEXT_CONSUMERS_PENDING',paths=[],additional_physical_weight_bytes=0,
        reason='Explicit conditional original146 verification',epoch=time.time())
    receipt['paths'].append(retain(UP/'private/OPERATOR_FISHER.pt',NEXT/'private/inherited/OPERATOR_FISHER.pt'))
    write(NEXT/'private/RETENTION.json',receipt)
    while not (UP/'private/CANDIDATES_FROZEN.json').exists():
        if list((UP/'private').glob('FAILURE_full*')) or (UP/'private/FAILURE_full_handoff1.json').exists():
            write(NEXT/'public/PROGRESS.json',dict(status='UPSTREAM_FAILED_RETAINED_NO146_TRAINING',new_model_calls=0));return
        time.sleep(20)
    for owner in range(1,9):
        result=read(UP/'private/results/REFINE'/f'{owner}.json')
        assert result['save_load_exact'] and len(result['curve'])==160
        receipt['paths'].append(retain(UP/'private/candidates/REFINE'/f'{owner}.pt',NEXT/'private/inherited/REFINE'/f'{owner}.pt'))
    assert len(receipt['paths'])==9
    write(NEXT/'private/RETENTION.json',receipt)
    while not (UP/'private/FULL_COMPLETE.json').exists():
        if list((UP/'private').glob('FAILURE_full*')):
            write(NEXT/'public/PROGRESS.json',dict(status='UPSTREAM_FAILED_RETAINED_NO146_TRAINING',new_model_calls=0));return
        time.sleep(20)
    result=read(UP/'public/FULL_RESULTS.json')
    if not gate(result):
        for item in receipt['paths']:
            path=Path(item['destination']);stat=path.stat()
            assert not path.is_symlink() and NEXT in path.resolve().parents
            assert (stat.st_dev,stat.st_ino)==(item['device'],item['inode'])
            path.unlink()
        write(NEXT/'private/CONDITIONAL_RELEASE.json',dict(files=9,additional_physical_weight_bytes=0,
            reason='Complete upstream compatibility gate did not pass',epoch=time.time()))
        write(NEXT/'public/PROGRESS.json',dict(status='NOT_STARTED_UPSTREAM_COMPATIBILITY_FAILED',new_model_calls=0));return
    write(NEXT/'private/UPSTREAM_GATE_PASS.json',dict(status='PASS',complete_scoring=True,epoch=time.time()))
    preflight()
    write(NEXT/'public/PROGRESS.json',dict(status='AUTHORIZED146_SOURCE_AND_IMPLEMENTATION_ADMISSION_PENDING',
        training_started=False,new_model_calls=0,upstream_compatibility_pass=True,epoch=time.time()))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(NEXT/'private/FAILURE_followup146.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
