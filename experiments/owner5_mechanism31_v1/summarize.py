"""Aggregate the fixed mechanism audit and same-modality admission without model calls."""
import json,sys
from pathlib import Path

def summarize(root):
    def read(path):return json.loads((root/path).read_text())
    admission=read('audit/ADMISSION.json');assert admission['status']=='PASS' and admission['diagnostic_cells']==8
    diagnostic=[dict(json.loads(p.read_text()),arm=p.stem.rsplit('_',1)[0]) for p in sorted((root/'audit/diagnostics').glob('*.json'))];assert len(diagnostic)==8
    assert all(v['zero_retention_parity'] and v['gradient_addition_checked'] for v in diagnostic)
    result=[]
    for v in diagnostic:
        a=v['variants'];e=a['EDIT_ONLY'];l=a['LEGACY_MIXED'];s=a['SEPARATED']
        result.append(dict(arm=v['arm'],step=v['step'],gradient_stats=v['gradient_stats'],semantic_update_distance=v['proposal_distance'],legacy_score_minus_edit=l['score']['score']-e['score']['score'],separated_score_minus_edit=s['score']['score']-e['score']['score'],separated_score_minus_legacy=s['score']['score']-l['score']['score'],legacy_edit_loss_minus_edit=l['weighted_edit_loss']-e['weighted_edit_loss'],separated_edit_loss_minus_edit=s['weighted_edit_loss']-e['weighted_edit_loss']))
    data=read('DATA_ADMISSION.json');verdicts=read('BASE_VERDICTS.json')['verdicts'];frozen=read('CANDIDATES_FROZEN.json');assert len(verdicts)==len(frozen['rows'])==6
    by={v['query_id']:v for v in verdicts}
    for row in frozen['rows']:
        output=read('qualification/outputs/BASE/'+row['query_id']+'.json')
        assert output['binding']['question']==row['question'] and output['binding']['reference']==row['reference']
        assert output['EOS'] and not output['at_cap'] and by[row['query_id']]['verdict']=='CORRECT'
    assert data['status']=='DATA_GATE_NOT_MET' and not data['six_arm_admitted']
    counts={k:sum(read(p+'/COMPLETE.json')['counts'][k] for p in ('audit','qualification')) for k in read('audit/COMPLETE.json')['counts']}
    assert counts==dict(updates=62,backwards=258,operator_backwards=94,generations=6,forwards=975)
    return dict(status='MECHANISM_PASS_DATA_GATE_NOT_MET',mechanism_admission=admission,diagnostics=result,source_article_candidates=12,source_licensed_figures_available=6,source_licensed_figures_unavailable=6,data_admission=data,base_semantic_review='Assistant fixed-reference review; candidate IDs visible; not strict independent role-blinding or clinical adjudication',counts=counts,six_arm_started=False,six_arm_updates=0,external_judge_calls=0,clinical_claim=False)

if __name__=='__main__':
    value=summarize(Path(sys.argv[1]));Path(sys.argv[2]).write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
