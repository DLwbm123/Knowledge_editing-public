"""Extract only the fixed train-candidate records with a tested streaming index."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.private/med150_source_review_v1_20261028'


def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Source review 30 minute limit')));signal.alarm(1800)
    assert not OUT.exists();OUT.mkdir(parents=True,mode=0o700)
    candidates=json.loads((ROOT/'.private/med150_source_metadata_v1_20261028/CANDIDATES_FROZEN.json').read_text())
    assert candidates['status']=='METADATA_FROZEN_NOT_SOURCE_ADMITTED' and len(candidates['selections'])==8
    chosen=candidates['selections'];flat=[q for row in chosen for q in row['qid_pool']]
    assert len(flat)==len(set(flat))==48
    parser=(ROOT/'experiments/med150_source_review_v1/stream_extract.py').read_text()
    remote='parser='+repr(parser)+'\nselection='+repr(chosen)+'\n'+'''from pathlib import Path
import json
scope={'__name__':'stream_extract'};exec(compile(parser,'<stream_extract>','exec'),scope)
assert scope['selfcheck']()['status']=='PASS'
wanted=[q for row in selection for q in row['qid_pool']]
p=Path('/data/bmw/DataP/knowledge_editing/data/m3bench/SLAKE/train.json')
with p.open('rb') as f:records,receipt=scope['extract'](f,wanted)
for group in selection:
 for q in group['qid_pool']:
  row=records[q]
  assert Path(row['img_name']).parts[0].lower()==group['source_group']
  assert row['q_lang']=='en' and row['answer_type']=='OPEN'
print(json.dumps(dict(records=records,receipt=receipt),ensure_ascii=False))
'''
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','pro5000','/data/bmw/envs/v0/bin/python','-'],input=remote,text=True,capture_output=True,timeout=300)
    if p.returncode:raise RuntimeError('Remote stream extraction failed: '+p.stderr[-2000:])
    assert len(p.stdout.encode())<20_000_000
    packet=json.loads(p.stdout);assert packet['receipt']['selected_records']==48 and packet['receipt']['unselected_records_decoded']==0
    assert set(map(int,packet['records']))==set(flat)
    target=OUT/'SOURCE_PACKET_PRIVATE.json';fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(packet,f,ensure_ascii=False,indent=2)
    receipt=dict(status='FIXED_SOURCE_RECORDS_EXTRACTED_NOT_REVIEWED',time=time.time(),candidate_groups=8,selected_records=48,
        scanned_records=packet['receipt']['scanned_records'],unselected_records_decoded=0,raw_byte_stream_scanned=True,
        validation_test_files_opened=False,pixel_reads=0,model_calls=0,patient_independence='UNKNOWN')
    (OUT/'EXTRACTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
