"""Apply fixed capability gates to explicit complete-answer semantic reviews."""
import json
import os
from pathlib import Path


def gate(rows, reviews, outputs):
    assert len(rows) == 32 and set(reviews) == {r['qid'] for r in rows}
    correct, unknown = {}, 0
    for row in rows:
        qid = row['qid']
        output = json.loads((outputs/(qid+'.json')).read_text())
        assert output['qid'] == qid and output['condition'] == 'image'
        answer = reviews[qid]['answer']
        assert answer in ['Yes', 'No', 'UNKNOWN']
        unknown += answer == 'UNKNOWN'
        correct[qid] = answer == row['reference']
    forms = {}
    for form in ['direct', 'reverse']:
        subset = [r for r in rows if r['form'] == form]
        totals = {p:sum(correct[r['qid']] for r in subset if r['polarity'] == p) for p in ['positive','negative']}
        cells = {label+'_'+p:sum(correct[r['qid']] for r in subset if r['label']==label and r['polarity']==p) for label in ['nodule','infiltrates'] for p in ['positive','negative']}
        pair_rows = {}
        for r in subset:
            pair_rows.setdefault(r['pair'], []).append(r)
        assert len(pair_rows)==8 and all(len(v)==2 and {r['polarity'] for r in v}=={'positive','negative'} for v in pair_rows.values())
        both = sum(all(correct[r['qid']] for r in pair) for pair in pair_rows.values())
        forms[form] = dict(correct_by_polarity=totals, correct_by_class_polarity=cells,
            pairs_both_correct=both, gate_pass=min(totals.values())>=6 and min(cells.values())>=3 and both>=5)
    return dict(gate_pass=all(f['gate_pass'] for f in forms.values()), forms=forms,
        correct=sum(correct.values()), denominator=32, unknown=unknown)


def main():
    root = Path(os.environ['REVIEW_ROOT'])
    panel = os.environ['REVIEW_PANEL']
    rows = [r for r in json.loads((root/'SOURCE_ROWS.json').read_text()) if r['panel']==panel]
    reviews = json.loads((root/(panel+'_REVIEW.json')).read_text())
    assert set(reviews)=={'O','M'} if panel=='D' else len(reviews)==1
    result = {key:gate(rows, value, root/'remote_results'/(key+'_'+panel)/'outputs') for key,value in reviews.items()}
    if panel == 'D':
        result.update(selected='O' if result['O']['gate_pass'] else 'M' if result['M']['gate_pass'] else None,
                      all_64_image_outputs_reviewed=True)
        (root/'DECISION_D.json').write_text(json.dumps(result,indent=2))
    else:
        result['status']='C_CAPABILITY_PASS' if next(iter(result.values()))['gate_pass'] else 'C_CAPABILITY_FAIL'
        (root/'DECISION_C.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
