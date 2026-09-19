"""One isolated reference/relation review of frozen candidates, after the current Judge owner exits."""
import json,sys,subprocess,base64
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_score import query_id
from scripts.medtrace.stage18_support import conflict
from scripts.medtrace.astra_judge_bundle import read,write_new
from scripts.medtrace.stage20_reference_review import run_one


def prepare(directory):
    d=Path(directory);p=d/'private';frozen=read(p/'CONFIRM_CANDIDATE_FREEZE.json');stream=read(p/'run/private/STREAM.json')
    assert digest({k:v for k,v in frozen.items() if k!='binding'})==frozen['binding']
    rows=frozen['rows'];groups=sorted({r['source_group'] for r in rows});assert len(rows)==5
    dest=p/'confirm_reference_images';dest.mkdir(exist_ok=True)
    local={g:str(dest/f'image{i:02d}.jpg') for i,g in enumerate(groups)}
    remote={g:next(r['image_path'] for r in rows if r['source_group']==g) for g in groups}
    missing={g:remote[g] for g in groups if not Path(local[g]).exists()}
    if missing:
        # Existing authorized images only; credentials never enter this payload or command line.
        script='import json,base64\nfrom pathlib import Path\nfiles='+repr(missing)+'\nprint(json.dumps({k:base64.b64encode(Path(v).read_bytes()).decode() for k,v in files.items()}))\n'
        proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-S','/tmp/job-519ft/socket','-p','30177','root@hb01-ssh.gpuhome.cc','python -'],input=script,text=True,capture_output=True,timeout=30)
        assert proc.returncode==0,proc.stderr
        data=json.loads(proc.stdout);assert set(data)==set(missing)
        for g,b in data.items():Path(local[g]).write_bytes(base64.b64decode(b,validate=True))
    assert all(Path(path).stat().st_size>0 for path in local.values())
    batches=[]
    for offset in range(0,len(groups),4):
        gs=groups[offset:offset+4]
        records=[dict(id=query_id(r),image_index=gs.index(r['source_group']),question=r['question'],reference=r['reference']) for r in rows if r['source_group'] in gs]
        batches.append(dict(name=f'confirm_ref_{offset//4:03d}',records=records,images=[local[g] for g in gs]))
    visible=lambda r:{k:r[k] for k in ('source_group','image_sha256','question','reference')}
    relations=[]
    for r in rows:
        if r['role']!='H_eval':continue
        matches=[t['native'] for t in stream['tasks'] if conflict(t['native'],r)]
        assert matches
        # Exact duplicate native source/question pairs have one relation decision, with all consumers retained.
        for native in {query_id(n):n for n in matches}.values():
            relations.append(dict(id=digest([query_id(native),query_id(r)]),native=visible(native),probe=visible(r)))
    if relations:batches.append(dict(name='confirm_rel_000',records=relations,images=[]))
    queue=dict(candidate_binding=frozen['binding'],batches=batches,items=sum(len(b['records']) for b in batches),category='22A',protocol='Unchanged isolated Stage20 source-reference/relation; no Base/student outputs',configuration_selection_access=False)
    queue['binding']=digest(queue);write_new(p/'CONFIRM_REVIEW_QUEUE.json',queue)
    print('Frozen reference/relation review queue:',queue['items'],'items; copied existing images:',len(missing))


def execute(directory):
    d=Path(directory);p=d/'private';root=p/'run';queue=read(p/'CONFIRM_REVIEW_QUEUE.json')
    assert digest({k:v for k,v in queue.items() if k!='binding'})==queue['binding']
    assert read(root/'public/REPORT_STATUS.json')['all_six_scored'] and read(root/'public/EXIT_22B.json')['exit_code']==0
    assert read(root/'public/CONTROLLER_STATUS.json')['phase']=='22B_COMPLETE_SCORED'
    assert not (p/'FINAL_STAGE23_LOCK.json').exists(),'Reference/member qualification must precede final lock'
    # This gate prevents concurrent CPU writers to the shared Judge ledger.
    for batch in queue['batches']:
        path=root/'private/reference_review'/batch['name']/'operator/RESULT.json'
        if not path.exists():
            attempt=path.parent/'ATTEMPT.json';assert not attempt.exists(),'No automatic semantic retry'
            run_one(root,batch['name'],batch['records'],batch['images'],category='22A')
        verdict=read(path);assert verdict['source_binding']==digest(batch['records'])
    refs={};rels={}
    for batch in queue['batches']:
        result=read(root/'private/reference_review'/batch['name']/'operator/RESULT.json')
        (refs if batch['images'] else rels).update({r['id']:r for r in result['records']})
    frozen=read(p/'CONFIRM_CANDIDATE_FREEZE.json');stream=read(p/'run/private/STREAM.json');kept=[];excluded=[]
    for row in frozen['rows']:
        q=query_id(row);reason=[]
        if refs[q]['verdict']!='SUPPORTED':reason.append('REFERENCE_'+refs[q]['verdict'])
        if row['role']=='H_eval':
            matches=[t['native'] for t in stream['tasks'] if conflict(t['native'],row)]
            if not all(rels[digest([query_id(n),q])]['verdict']=='SUPPORTED' for n in matches):reason.append('RELATION_NOT_SUPPORTED')
        if reason:excluded.append(dict(query_id=q,reasons=reason))
        else:kept.append({k:v for k,v in row.items() if k!='reference_review'})
    panel=dict(rows=kept,excluded=excluded,parent_candidate_binding=frozen['binding'],review_binding=queue['binding'],sealed_for_once_only_final_stage23=True,Base_generated=False,clinical_signoff=False,patient_study='UNKNOWN')
    panel['binding']=digest(panel);write_new(p/'CONFIRM_QUALIFIED_FREEZE.json',panel)
    summary=dict(qualified={role:dict(QA=sum(r['role']==role for r in kept),sources=len({r['source_group'] for r in kept if r['role']==role})) for role in ('H_eval','U_eval','positive_image')},excluded=len(excluded),items=queue['items'],scope='At most five source-held-out probes; insufficient for target panel or independent broad confirmation',parent_candidate_binding=frozen['binding'],qualified_binding=panel['binding'],Base_student_outputs_used=False,clinical_signoff=False,patient_study='UNKNOWN')
    write_new(d/'CONFIRM_QUALIFICATION_SUMMARY.json',summary)


if __name__=='__main__':
    (execute if sys.argv[1]=='execute' else prepare)(sys.argv[2])
