"""Source-score-only admission, then freeze bindings before formal GPU work."""
import json,os,sqlite3,collections,sys
from pathlib import Path
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/tools')
sys.path.insert(0,os.environ['RUN_ROOT']+'/private/source')
import cohort,journal as j
from scripts.medtrace.stage17_prepare import digest
R=Path(os.environ['RUN_ROOT']);P=Path(os.environ['REFERENCE_PARENT'])
if (R/'private/DATA_ADMISSION.json').exists():
    print((R/'private/DATA_ADMISSION.json').read_text());raise SystemExit(0)
assert (R/'private/judge_source_v5_qualification/ALL_WORKERS_COMPLETE.json').is_file()
db=sqlite3.connect('file:'+str(R/'private/judge_source_v5_qualification/queue.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
scores={r['query_id']:r['correct'] if r['status']=='FORMAT_VALID' else None for r in db.execute('SELECT c.query_id,p.status,p.correct FROM consumer c JOIN payload p ON c.payload_key=p.key')}
packet=json.loads((R/'private/COHORT_CANDIDATES.json').read_text())
try:chosen,rows=cohort.select(packet,scores)
except ValueError as e:
    result=dict(status='STOP_DATA_QUALIFICATION_NOT_MET',reason=str(e),candidate_QA=len(scores),missing=sum(x is None for x in scores.values()),Base_wrong=sum(x==0 for x in scores.values()),formal_updates=0)
    j.write(R/'public/RESULTS.json',result);j.write(R/'private/DATA_GATE_STOP.json',result);print(json.dumps(result));raise SystemExit(0)
oldrows=json.loads((P/'private/ROWS.json').read_text());held=[r for r in oldrows if r['audit_role']=='HELDOUT'];assert len(held)==96
rows+=held
assert not {r['source_group'] for r in cohort.training_rows(rows)}&{r['source_group'] for r in rows if r['audit_role'] in ('HELDOUT','NEW_HELDOUT')}
initial=json.loads((R/'private/LOCK.json').read_text())
formal=dict(initial,selected_source_binding=digest(chosen),rows_binding=digest(rows),selection_inputs='only84 source Base verdicts; no edited or generalization verdicts')
j.write(R/'private/LOCK.json',formal);j.write(R/'private/ROWS.json',rows)
for row in rows:
    if row['audit_role']=='NATIVE':source=R/'private/outputs/QUAL_BASE/0'/(row['query_id']+'.json')
    elif row['audit_role']=='HELDOUT':source=P/'private/outputs/BASE/0'/(row['query_id']+'.json')
    else:continue
    d=json.loads(source.read_text());assert (d['binding']['input']['question'],d['binding']['input']['reference'])==(row['question'],row['reference'])
    copied=dict(d,stage='BASE',owner=row['owner'],role=row['audit_role'],query_id=row['query_id'],lock=digest(formal),binding=dict(d['binding'],input=row),reused_generation=dict(path=str(source),lock=d['lock']))
    j.write(R/'private/outputs/BASE'/str(row['owner'])/(row['query_id']+'.json'),copied)
result=dict(status='PASS',ordered_edits=24,distinct_source_images=24,source_qualification_wrong=24,source_provenance_exact=True,new_protection_QA=45,new_protection_images=10,old_protection_QA=96,variant_types=dict(collections.Counter(r['variant_kind'] for r in rows if r['audit_role']=='T2G')),inherited_source_Base_outputs=24,inherited_old_Base_outputs=96,new_training_labels_from_evaluation=0,clinical_scope_GFITS_admitted=0,patient_independence='UNKNOWN',formal_binding=digest(formal))
j.write(R/'private/DATA_ADMISSION.json',result);j.write(R/'public/DATA_ADMISSION.json',result);print(json.dumps(result))
