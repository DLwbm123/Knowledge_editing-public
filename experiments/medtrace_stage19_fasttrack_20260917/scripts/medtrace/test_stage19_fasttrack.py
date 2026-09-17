"""Small CPU regression for cumulative budget, strict old schema and new branch scope."""
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage19_fasttrack_budget import remaining
from scripts.medtrace.stage18_support import validate_task
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_cfact import artifact_binding, extra_roles
from scripts.medtrace.stage19_fasttrack import audited_resume


def main():
    ledger=dict(limit_seconds=28800,sessions=[dict(started_epoch=1,seconds=120),dict(started_epoch=300)])
    assert remaining(ledger,now=350)==28630
    ledger['sessions'][1]['seconds']=80
    assert remaining(ledger,now=1000)==28600
    row=dict(dataset='SLAKE',image_path='/data/xmlab1/source.jpg',image_sha256='a',source_group='g1',source_qid=1,question='Does the picture contain spleen?',reference='No',role='native')
    q=row['question'];edit='test:1'
    t=dict(canonical_edit_id=edit,order=1,seed=int(digest([20260912,edit])[:8],16),native=row,
           fit_questions=[f'Please answer the following question: {q}',f'Question: {q}',f'{q} Please provide an answer.',f'Please respond to this question: {q}'],
           U_fit=[dict(row,role='U_fit',question='What modality is shown?',reference='CT',source_qid=2,image_path='/data/xmlab2/source.jpg',image_sha256='b',source_group='g2')],H_fit=[],G_fit=[])
    validate_task(t,fasttrack_branch='C_NO_H')
    for kwargs in ({},{'fasttrack_branch':'C_FACT'}):
        try:validate_task(t,**kwargs)
        except ValueError:pass
        else:raise AssertionError('Missing H accepted')
    t['H_fit']=[dict(row,role='H_fit',reference='Yes',source_qid=3,image_path='/data/xmlab3/source.jpg',image_sha256='c',source_group='g3')]
    validate_task(t,fasttrack_branch='C_FACT')
    try:validate_task(t)
    except ValueError:pass
    else:raise AssertionError('Old H/G contract relaxed')
    assert extra_roles(True)==('H_fit',) and extra_roles(False)==('H_fit','G_fit')
    cfg=dict(mode='STAGE19_FASTTRACK_TWO_ARMS',approved_predecessor_commits=['old'],stream_binding='frozen')
    saved=dict(code='old',task='unchanged')
    assert artifact_binding(saved,dict(saved,code='new'),cfg)==saved
    for expected in (dict(code='new',task='changed'),dict(code='new',task='unchanged',extra=1)):
        try:artifact_binding(saved,expected,cfg)
        except ValueError:pass
        else:raise AssertionError('Scientific binding change accepted')
    with TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'private').mkdir();(root/'public').mkdir()
        assert audited_resume(root,{},['one'])['inserted']==[]
        rows=[dict(arm=arm,mode=mode,inserted=['one'],code='old',output=dict(raw_token_ids=[1,2]))
              for mode in ('insertion','integration_replay') for arm in ('A','B')]
        (root/'private/OUTPUTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (root/'public/INTEGRATION_001.json').write_text('{"status":"PASS"}')
        bank=dict(inserted=['one'],stream='frozen',routes={'one':{}},banks={a:{'one':{'code':'old'}} for a in ('A','B')})
        torch.save(bank,root/'private/BANKS.pt')
        cfg['resume_audit']=dict(completed_pairs=1,outputs_binding=digest(rows))
        assert audited_resume(root,cfg,['one','two'])['inserted']==['one']
        bank['inserted']=['one','two'];torch.save(bank,root/'private/BANKS.pt')
        try:audited_resume(root,cfg,['one','two'])
        except ValueError:pass
        else:raise AssertionError('Unaudited prefix accepted')
    print('PASS: cumulative budget; legacy H/G; FASTTRACK H only; strict code lineage; audited paired resume')


if __name__=='__main__':main()
