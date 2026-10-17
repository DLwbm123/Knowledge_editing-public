"""Choose one of three completed DEV candidates under predeclared protection gates."""
import sqlite3
from pathlib import Path
import curriculum_report as report
from curriculum_protocol import read
CANDIDATES=('C','M1','M13P')
TIE_ORDER={'M13P':0,'M1':1,'C':2}

def choose(candidates):
    assert set(candidates)==set(CANDIDATES)
    eligible=[m for m,d in candidates.items() if all(v is True for v in d['protection_gate'].values()) and d['T2G'] is not None and d['T2G']>80.]
    return min(eligible,key=lambda m:(-candidates[m]['T2G'],candidates[m]['new_damage'],TIE_ORDER[m])) if eligible else None

def evaluate(single):
    import astra_queue as a
    records={};scores={};q=a.q
    with sqlite3.connect('file:'+str(single/'private/judge_curriculum146_astra_medium/queue.sqlite')+'?mode=ro',uri=True) as db:
        assert not db.execute("select 1 from payload where status in ('PENDING','RESERVED')").fetchone()
        for path,binding,status,score in db.execute('select c.path,c.output_binding,p.status,p.correct from consumer c join payload p on p.key=c.payload_key'):
            d=read(Path(path));assert q.digest(d)==binding
            if d['method'] not in (*CANDIDATES,'P160','BASE'):continue
            key=tuple(d[x] for x in ('method','mode','prefix','owner'))+(d['binding']['input']['query_id'],)
            assert key not in records;records[key]=d;scores[key]=score if status=='FORMAT_VALID' else None
    out={};details={}
    for m in CANDIDATES:
        subset={k:v for k,v in records.items() if k[0] in (m,'P160','BASE')}
        result,_=report.calculate(read(single/'private/ORIGINAL146_TASKS.json'),read(single/'private/ORIGINAL_BASE_MASK146.json'),subset,{k:scores[k] for k in subset},methods=('P160',m),single_only=True)
        terms=result['paired_native_FIT160_contrasts']['single_1'];damage=result['protection'][m+'_single_1']
        gate=dict(complete_no_missing=result['complete_no_missing'])
        for role in ('FIT','T1G','T1L','T2L'):gate[role]=terms[role][m+'_preserves_each_P160_owner']
        gate['damage']=terms[m+'_damage_no_worse_than_P160_per_owner'] and damage['new_damage']<=2 and damage['fresh_Base_correct']==85
        out[m]=dict(T2G=result['panels'][m]['single_1']['T2G']['macro'],new_damage=damage['new_damage'],protection_gate=gate);details[m]=result
    return dict(candidates=out,selected_method=choose(out),all_scoring_complete=all(d['protection_gate']['complete_no_missing'] for d in out.values()),selection_scope='Fixed three-arm exposed DEV selection; not independent confirmation'),details
