"""Small CPU check for the external benchmark boundaries, not a performance gate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.medtrace.stage15_sources import select, canonical, identity, paraphrases, norm
from scripts.medtrace.stage15 import TARGET_PROMPT, SOURCE_PROMPT, render_judge


def test_contract():
    a=dict(image='i1',src='What is shown?',pred='A',alt='B',clinical_VQA_task='task1',modality='CT')
    b=dict(a,image='i2',clinical_VQA_task='task2')
    c=dict(a,image='i3')
    assets={r['image']:dict(sha256=r['image']) for r in (a,b,c)}
    picked, counts=select([a,b,c,a],assets,{'i3'})
    assert {canonical(r) for r in picked}=={canonical(a),canonical(b)}
    assert counts['duplicate_edit']==1 and counts['historical_identity_excluded']==1
    assert select([dict(r,Base_correct=True,H_available=False) for r in (a,b,c)],assets,{'i3'})[0]==[dict(r,Base_correct=True,H_available=False) for r in picked]
    assert identity(a)!=identity(dict(a,alt='C'))
    assert len(set(map(norm,paraphrases(a['src']))))==4
    assert all(a['src'] in q for q in paraphrases(a['src']))
    assert norm('  Ａ\nB ')==norm('a b') and norm('yes.')!=norm('yes')
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):return messages
    row=dict(opaque_query_id='x',question='q',gold_answer='counterfactual',raw_base_answer='a',adjudication_pass='TARGET_ADHERENCE_JUDGE')
    assert render_judge(Tokenizer(),row)[0]['content']==TARGET_PROMPT
    row['adjudication_pass']='SOURCE_ANSWER_JUDGE'
    assert render_judge(Tokenizer(),row)[0]['content']==SOURCE_PROMPT
    row['adjudication_pass']='method-name'
    try:render_judge(Tokenizer(),row)
    except KeyError:pass
    else:raise AssertionError('Unknown Judge protocol accepted')


if __name__=='__main__':
    test_contract();print('Stage15 CPU contract passed')
