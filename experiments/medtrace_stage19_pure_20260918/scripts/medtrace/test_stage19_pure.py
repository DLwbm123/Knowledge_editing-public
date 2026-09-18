"""CPU regression: pure19/pure18 freeze, no background fallback, 300-item guard."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage19_pure_audit import finalize
from scripts.medtrace.stage19_fasttrack_closeout import judge
from scripts.medtrace.stage18_score import query_id,score_key
from scripts.medtrace.stage17_prepare import digest


def main():
    with TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'private').mkdir();(root/'public').mkdir()
        def write(name,value):(root/name).write_text(json.dumps(value))
        tasks=[dict(canonical_edit_id=str(i),order=i+1,native=dict(image_sha256=str(i),source_group=str(i),question='Q',reference='R'),H_fit=[{}],U_fit=[{}]) for i in range(19)]
        stream=dict(tasks=tasks,anchors=[str(i) for i in range(11)],track='P',role_transitions=[])
        records=[dict(query_id=query_id(t['native']),output=dict(raw_answer='W')) for t in tasks]
        scores={score_key(t['native'],r['output']):False for t,r in zip(tasks,records)}
        for expected in (19,18):
            write('private/STREAM_CANDIDATES.json',stream)
            write('public/RUN_AUTHORIZATION.json',dict(approved=True,candidate_stream_binding=digest(stream)))
            write('private/FRESH_BASE_OUTPUTS.json',dict(records=records))
            if expected==18:scores[score_key(tasks[-1]['native'],records[-1]['output'])]=True
            write('private/QUALIFIED_SCORE_CACHE.json',dict(scores=scores))
            with patch('scripts.medtrace.stage19_pure_audit.validate_task'):
                result=finalize(root)
            assert result['N']==result['K']==expected and result['track']=='P' and result['background_newly_trained']==0
            for name in ['private/STREAM.json','private/TRAINING_ELIGIBILITY.json','public/STREAM_MANIFEST.json','public/TRAINING_ELIGIBILITY.json']:(root/name).unlink()
        write('public/BUDGET_LEDGER.json',dict(new_judgment_limit=300,new_judgment_items_dispatched=300,qualification_limit=45,qualification_items_dispatched=0))
        novel=[dict(source=dict(image_sha256='new',question='Q',reference='R'),output=dict(raw_answer='new'))]
        try:judge(root,novel,'budget_test')
        except ValueError as error:assert 'budget exceeded' in str(error)
        else:raise AssertionError('300-item budget failed')
    print('PASS: pure19 and pure18 preserve all-FACT identity; independent 300-item budget blocks dispatch')


if __name__=='__main__':main()
