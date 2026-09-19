"""Freeze DEV selection and held-source candidates without reading student scores."""
from pathlib import Path
import json,sys
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_support import conflict,train_row
from scripts.medtrace.prepare_stage2_sources import normalized,reviewed_attribute
from scripts.medtrace.stage18_score import query_id
from scripts.medtrace.astra_judge_bundle import write_new


def read(p):return json.loads(p.read_text())

def freeze(out):
    out=Path(out);p=out/'private';s=read(ROOT/'reports/medtrace_stage21b_20260919/private/run/private/STREAM.json');inventory=read(p/'SOURCE_INVENTORY.json');by=defaultdict(list)
    for row in s['new_rows']:
        if row['role']=='H_eval':by[row['source_group']].append(row)
    h=[r for source in sorted(by) for r in sorted(by[source],key=lambda r:digest([22001,query_id(r)]))[:2]];assert len(h)==16 and len(by)==8
    def unrelated(r,n):return all(normalized(r['question'])!=normalized(t['native']['question']) and (reviewed_attribute(r['question']) is None or reviewed_attribute(r['question'])!=reviewed_attribute(t['native']['question'])) for t in s['tasks'][:n])
    u=[r for r in s['new_rows'] if r['role']=='U_eval' and unrelated(r,19)]
    extra=sorted([r for r in s['core_rows'] if r['role']=='U_eval' and unrelated(r,19)],key=lambda r:digest([22001,query_id(r)]))
    u+=extra[:max(0,16-len(u))];assert len(u)==16
    rewrite=[r for r in s['core_rows'] if r['role']=='native_text_extension'];positive=s['positive_rows'];rows=h+u+rewrite+positive
    assert len(rewrite)==11 and len(positive)==4 and len(rows)==47 and len({query_id(r) for r in rows})==len(rows)
    train=[r for t in s['tasks'] for r in [t['native']]+t['H_fit']+t['U_fit']];ts={r['source_group'] for r in train};ti={r['image_sha256'] for r in train}
    assert not ({r['source_group'] for r in h+u+positive}&ts) and not ({r['image_sha256'] for r in h+u+positive}&ti)
    dev=dict(N=19,rows=rows,natives=[t['native'] for t in s['tasks'][:19]],selection_seed=22001,selection_rule='H two per existing new-panel source via hash; all 14 strict-prefix19 new U plus two hash-ordered old U; original11 rewrites and4 positive',exposure='ALL_VIEWED_DEV; source-isolated from full45 training; not unseen confirmation',reference_status='Exact unchanged source rows inherited from qualified frozen Stage20 panels',per_arm_queries=66,student_answers_read=False)
    dev['binding']=digest(dev);write_new(p/'DEV_PANEL.json',dev)
    candidates=inventory['CONFIRM_candidates'];hs=[r for r in candidates if any(conflict(t['native'],r) for t in s['tasks'])]
    us=[r for r in candidates if reviewed_attribute(r['question']) in ('modality','plane') and unrelated(r,45)]
    gs=[r for r in candidates if any(r['image_sha256']!=t['native']['image_sha256'] and normalized(r['question'])==normalized(t['native']['question']) and normalized(r['reference'])==normalized(t['native']['reference']) for t in s['tasks'])]
    # Deterministic candidates stay frozen even if later reference review rejects them.
    roles={'H_eval':hs,'U_eval':us,'positive_image':gs};caps={'H_eval':32,'U_eval':32,'positive_image':8};reserved=[]
    for role,cand in roles.items():
        counts=Counter()
        for r in sorted(cand,key=lambda r:digest([23001,query_id(r)])):
            if counts[r['source_group']]>=2:continue
            reserved.append(dict(train_row(r,role),reference_review=r['reference_review']));counts[r['source_group']]+=1
            if sum(counts.values())>=caps[role]:break
    # No image-level model/student outputs have been generated on these candidates.
    candidate=dict(status='SEALED_CANDIDATES_REFERENCE_RELATION_REVIEW_PENDING',rows=reserved,reserved_sources=sorted({r['source_group'] for r in candidates}),exposure='No training or Base-generation exposure found in enumerated prior ledgers; not proof of pretraining-unseen data',selection_seed=23001,patient_study='UNKNOWN',selection_no_student_outputs=True,qualification_failures='Drop with reason, no outcome-based replacement, preserve this original freeze')
    candidate['binding']=digest(candidate);write_new(p/'CONFIRM_CANDIDATE_FREEZE.json',candidate)
    summary=dict(DEV=dict(H_QA=16,H_sources=8,U_QA=16,U_sources=len({r['source_group'] for r in u}),U_strict_full45_QA=sum(unrelated(r,45) for r in u),rewrites=11,positive_image=4,queries_per_arm=66,exposure='VIEWED_DEV',binding=dev['binding']),CONFIRM_candidates={role:dict(QA=sum(r['role']==role for r in reserved),sources=len({r['source_group'] for r in reserved if r['role']==role})) for role in roles},CONFIRM_qualified=False,CONFIRM_binding=candidate['binding'],CONFIRM_limit='Finite authorized pool and current conservative relation rules; not a claim that every conceivable semantic relation was exhausted',budgets=dict(Stage22B_worst_new_student_judgments=396,Stage22A_worst_DEV_Base_judgments=66,new_GPU_seconds=0,new_judgments=0),source_pool_binding=inventory['pool_binding'])
    write_new(out/'PANEL_FREEZE_SUMMARY.json',summary);return summary

if __name__=='__main__':print(json.dumps(freeze(Path(__file__).parent),indent=2))
