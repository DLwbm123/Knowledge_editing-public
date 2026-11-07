"""Construct source-bound T5 presence tasks without any model or Judge calls."""
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
import shutil
import struct
import time
import zipfile

from split_zip import SplitFile, volumes

PRESENCE = {'stable':'11','improving':'11','worsening':'11','new':'01','resolved':'10'}


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def jsonl(path, rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')


def main():
    cfg = json.loads(Path(os.environ['T5_CONFIG']).read_text())
    source, output = Path(cfg['source']), Path(cfg['package'])
    assert not output.exists() and time.time() < cfg['clock']['hard_end']-120
    assert shutil.disk_usage(output.parent).free > 8*1024**3
    output.mkdir(mode=0o700)
    probe=output/'.probe';probe.write_text('ok');assert probe.read_text()=='ok';probe.unlink()
    reports = json.loads((source/'metadata/grounded_reports_20240819.json').read_text())
    report_index = {r['ImageID']:r for r in reports}
    assert len(report_index)==len(reports)
    with zipfile.ZipFile(source/'metadata/master_table.csv.zip') as z:
        with z.open('master_table.csv') as f:
            rows=list(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig')))
    images, groups = defaultdict(list), defaultdict(list)
    for line, row in enumerate(rows, 1):
        images[row['ImageID']].append(row)
        if row['progression_status'] in PRESENCE and row['prior_imageID']:
            groups[(row['ImageID'],row['prior_imageID'],row['label'])].append((line,row))
    patients = sorted({a[0][1]['PatientID'] for a in groups.values()})
    pseudonyms = {p:f'patient_{i:04d}' for i,p in enumerate(patients,1)}
    jsonl(output/'PATIENT_MAPPING.private.jsonl',[dict(patient_id=v,source_PatientID=k) for k,v in pseudonyms.items()])
    cases, exclusions, source_rows = [], [], []
    pair_audit = {}
    for key, members in sorted(groups.items()):
        current, prior, label = key
        row = members[0][1]
        assert all(x['PatientID']==row['PatientID'] and x['StudyDate_DICOM']==row['StudyDate_DICOM'] and x['StudyID']==row['StudyID'] for x in images[current])
        report=report_index[current]
        assert report['PreviousImageID']==prior
        assert str(report['StudyID'])==str(row['StudyID'])
        assert all(any(x['label'] in (f.get('labels') or []) and x['progression_status']==f.get('progression') for f in report['findings']) for _,x in members)
        codes = {PRESENCE[x['progression_status']] for _,x in members}
        if len(codes)!=1:
            exclusions.append(dict(image=current,prior=prior,label=label,reason='CONFLICTING_BINARY_PRESENCE',source_rows=[n for n,_ in members]))
            continue
        code=next(iter(codes));prior_date=None;independent=False
        current_date=datetime.strptime(row['StudyDate_DICOM'],'%Y%m%d').date()
        if prior in images:
            p=images[prior][0]
            assert all(x['PatientID']==p['PatientID'] and x['StudyDate_DICOM']==p['StudyDate_DICOM'] for x in images[prior])
            prior_date=datetime.strptime(p['StudyDate_DICOM'],'%Y%m%d').date()
            if p['PatientID']!=row['PatientID'] or prior_date>=current_date or str(p['StudyID'])!=str(report['PreviousStudyID']):
                exclusions.append(dict(image=current,prior=prior,label=label,reason='PATIENT_OR_CHRONOLOGY_CONFLICT',source_rows=[n for n,_ in members]))
                continue
            independent=True
        pair_audit[(prior,current)]=dict(independent_prior_metadata_crosscheck=independent,
            chronology_basis='two_source_study_dates' if independent else 'explicit_previous_study_link_in_both_metadata_files')
        question=f'Is there evidence of {label} in this chest radiograph?'
        case_id=f't5_{len(cases)+1:04d}'
        def view(image, study, date, answer, point):
            return dict(image_id=image,study_id=str(study),study_date=str(date) if date else None,
                time_point=point,relative_image_path='images/'+image,question=question,gold_answer=answer,
                model_answer_before=None,is_correct_before=None)
        task=dict(task='T5',dataset='padchest_gr',case_id=case_id,patient_id=pseudonyms[row['PatientID']],
            source_split=row['split'],label=label,progression_statuses=sorted({x['progression_status'] for _,x in members}),
            presence_code=code,source_master_rows=[n for n,_ in members],
            edit_case=view(prior,report['PreviousStudyID'],prior_date,'Yes' if code[0]=='1' else 'No','T1'),
            eval_case=view(current,report['StudyID'],current_date,'Yes' if code[1]=='1' else 'No','T2'),
            linkage_audit=pair_audit[(prior,current)],Base_eligibility='PENDING_CURRENT_CORE_BASE_WRONG_CHECK')
        task['eval_case'].update(model_answer_after=None,is_correct_after=None)
        cases.append(task)
        source_rows.extend(dict(master_row=n,canonical_case_id=case_id,progression=x['progression_status']) for n,x in members)
    assert cases
    needed={t[k]['image_id'] for t in cases for k in ('edit_case','eval_case')}
    archive_audit, locations, modes = [], {}, Counter()
    (output/'images').mkdir()
    with ExitStack() as stack:
        archives=[]
        for directory,prefix,count in [(source/'Padchest_GR_files','PadChest_GR.zip',37),
            (source/'PadChest_GR_progression_prior_studies','PadChest_GR_progression_prior_studies.zip',9)]:
            parts=volumes(directory,prefix,count);stream=stack.enter_context(SplitFile(parts));z=stack.enter_context(zipfile.ZipFile(stream))
            archives.append(z)
            archive_audit.append(dict(prefix=prefix,volumes=len(parts),compressed_bytes=sum(p.stat().st_size for p in parts),
                member_count=len(z.infolist()),uncompressed_bytes=sum(x.file_size for x in z.infolist())))
            for info in z.infolist():
                name=Path(info.filename).name
                if name in needed and not info.is_dir():
                    locations.setdefault(name,(z,info))
        missing=sorted(needed-locations.keys())
        assert not missing, f'Missing {len(missing)} required images'
        expected_bytes=sum(locations[n][1].file_size for n in needed)
        main_bytes=archive_audit[0]['compressed_bytes']
        assert main_bytes+expected_bytes+2*1024**3 < cfg['new_remote_bytes_limit']
        assert expected_bytes+1024**3 < cfg['new_local_bytes_limit']
        for index,name in enumerate(sorted(needed),1):
            assert Path(name).name==name and time.time()<cfg['clock']['hard_end']-120
            z,info=locations[name];dest=output/'images'/name
            with z.open(info) as src,dest.open('xb') as dst:
                shutil.copyfileobj(src,dst,1024*1024)
            assert dest.stat().st_size==info.file_size
            with dest.open('rb') as f:
                header=f.read(33)
                assert header[:8]==b'\x89PNG\r\n\x1a\n' and header[12:16]==b'IHDR'
                width,height,depth,color,compression,filtering,interlace=struct.unpack('>IIBBBBB',header[16:29])
                assert width>0 and height>0 and compression==0 and filtering==0 and interlace in (0,1)
                f.seek(-12,2);assert f.read(12)[4:8]==b'IEND'
            mode=f'PNG_color{color}_bit{depth}'
            modes[mode]+=1
            if index%25==0:
                p=Path(cfg['progress']);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(extracted=index,total=len(needed),time=time.time())));tmp.replace(p)
    # Keep the full single-edit cohort. A conservative separate sequential list
    # prevents any registered later image being used as another edit's input.
    followup_ids={t['eval_case']['image_id'] for t in cases}
    sequential=[t for t in cases if t['edit_case']['image_id'] not in followup_ids]
    jsonl(output/'T5_CANONICAL.private.jsonl',cases)
    jsonl(output/'T5_SEQUENTIAL_ROLE_SAFE.private.jsonl',sequential)
    jsonl(output/'SOURCE_ROW_BINDINGS.private.jsonl',source_rows)
    jsonl(output/'EXCLUSIONS.private.jsonl',exclusions)
    fits=[]
    for t in cases:
        label=t['label'];source=t['edit_case']
        for i,q in enumerate([f'Does this chest radiograph show {label}?',f'Is {label} present in this chest X-ray?',
            f'Can {label} be seen in this chest radiograph?',f'Is this chest X-ray positive for {label}?'],1):
            fits.append(dict(case_id=t['case_id'],view=i,image_id=source['image_id'],relative_image_path=source['relative_image_path'],
                question=q,gold_answer=source['gold_answer'],role='PRIOR_ONLY_FIT'))
    jsonl(output/'T5_PRIOR_FIT.private.jsonl',fits)
    patient_splits=defaultdict(set)
    for t in cases:patient_splits[t['patient_id']].add(t['source_split'])
    summary=dict(status='DATA_BUILT_BASE_ELIGIBILITY_PENDING',source_rows=len(source_rows),canonical_cases=len(cases),
        merged_redundant_rows=len(source_rows)-len(cases),excluded_groups=len(exclusions),
        patient_groups=len(patient_splits),cross_split_patient_groups=sum(len(v)>1 for v in patient_splits.values()),
        image_pairs=len(pair_audit),unique_required_images=len(needed),image_bytes=expected_bytes,
        raw_source_image_modes=dict(modes),lossless_source_files_preserved=True,
        image_validation='Built-in ZIP CRC on extraction; PNG signature, IHDR dimensions/bit depth and IEND. No full pixel decoding.',
        model_input_preprocessing_admission='PENDING; preserve raw PNG bit depth, no automatic clipping or window selection',
        independent_date_patient_crosschecked_pairs=sum(x['independent_prior_metadata_crosscheck'] for x in pair_audit.values()),
        source_previous_link_only_pairs=sum(not x['independent_prior_metadata_crosscheck'] for x in pair_audit.values()),
        labels=len({t['label'] for t in cases}),progression_case_counts=dict(Counter(t['presence_code'] for t in cases)),
        original_split_cases=dict(Counter(t['source_split'] for t in cases)),
        sequential_global_image_role_safe_cases=len(sequential),FIT_views=len(fits),archives=archive_audit,
        author_reported_reference=dict(images=440,edits=324,diseases=43),
        exact_author_cohort_reproduction=False,selection_uses_method_results=False,model_calls=0,Judge_calls=0,
        new_remote_bytes_upper_bound=main_bytes+expected_bytes+2*1024**3)
    write(output/'SUMMARY.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':main()
