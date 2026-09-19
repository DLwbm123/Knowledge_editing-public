"""One bounded CPU inventory from frozen authorized sources; no student-based selection."""
import sys,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage19_fasttrack_budget import read,write
from scripts.medtrace.astra_judge_bundle import write_new
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_score import query_id,score_key
from scripts.medtrace.stage18_support import conflict
from scripts.medtrace.prepare_stage2_sources import normalized,reviewed_attribute
from scripts.medtrace.stage20_closeout import outputs


def run():
    d=Path(__file__).parent;p=d/'private';old=ROOT/'reports/medtrace_stage22_20260919';run=old/'private/run';s=read(run/'private/STREAM.json');inv=read(old/'private/SOURCE_INVENTORY.json');dev=read(old/'private/DEV_PANEL.json');confirm=read(old/'private/CONFIRM_QUALIFIED_FREEZE.json')
    # Preserve memberships and panel-specific roles even where a query appears in both panels.
    union={query_id(r):r for r in dev['natives']+dev['rows']}
    for r in s['core_rows']:union.setdefault(query_id(r),r)
    assert len(union)<=92
    panel=dict(rows=list(union.values()),DEV66=[query_id(r) for r in dev['natives']+dev['rows']],core=[query_id(r) for r in s['core_rows']],exposure='VIEWED_DEV',student_selection=False);panel['binding']=digest(panel);write_new(p/'DEV_UNION.json',panel)
    used=confirm['rows'];groups={r['source_group'] for r in used};images={r['image_sha256'] for r in used}
    candidates=[r for r in inv['CONFIRM_candidates'] if r['source_group'] not in groups and r['image_sha256'] not in images]
    def unrelated(r):return all(normalized(r['question'])!=normalized(t['native']['question']) and reviewed_attribute(r['question'])!=reviewed_attribute(t['native']['question']) for t in s['tasks'])
    roles={'H_eval':[r for r in candidates if any(conflict(t['native'],r) for t in s['tasks'])], 'U_eval':[r for r in candidates if reviewed_attribute(r['question']) in ('modality','plane') and unrelated(r)], 'positive_image':[r for r in candidates if any(r['image_sha256']!=t['native']['image_sha256'] and normalized(r['question'])==normalized(t['native']['question']) and normalized(r['reference'])==normalized(t['native']['reference']) for t in s['tasks'])]}
    freeze=dict(candidate_rows=roles,remaining_unexposed_metadata=candidates,prior_pool_binding=inv['pool_binding'],used_confirm_excluded_sources=len(groups),rule='Same conservative exact-proposition and modality/plane qualification as Stage22; no claim to exhaustive medical semantic eligibility',patient_study='UNKNOWN',new_gold_created=False);freeze['binding']=digest(freeze);write_new(p/'CONFIRM_V2_INVENTORY.json',freeze)
    summary=dict(prior_bounded_scan_QA=inv['public']['scanned_QA'],prior_bounded_scan_sources=inv['public']['scanned_sources'],remaining_candidate_QA=len(candidates),remaining_candidate_sources=len({r['source_group'] for r in candidates}),roles={k:dict(QA=len(v),sources=len({r['source_group'] for r in v})) for k,v in roles.items()},status='INSUFFICIENT_NEW_CONFIRM_SUPPORT',new_GPU_seconds=0,new_judgments=0,rule=freeze['rule'],binding=freeze['binding'],DEV_union=len(union),DEV_binding=panel['binding'],source_counts_target={'H':[32,16],'U':[32,16],'positive':[8,8]});write_new(d/'SOURCE_AND_PANEL_AUDIT.json',summary)
    scores=read(run/'private/QUALIFIED_SCORE_CACHE.json')['scores'];rows=[r for r in outputs(run) if r['arm'] in ('R_E0','R_E1','R_E2') and r['prefix']==45 and r['mode']=='endpoint'];paired=[]
    for r in rows:
        paired.append(dict(arm=r['arm'],query_id=r['query_id'],source=r['source'],Base_correct=scores[score_key(r['source'],r['Base'])],correct=scores[score_key(r['source'],r['output'])],output_binding=digest(r['output']),Base_binding=digest(r['Base']),route=r['route'],panel='old_DEV' if r['query_id'] in {query_id(x) for x in s['core_rows']} else 'native' if r['source']['role']=='native' else 'viewed_new_DEV'))
    write_new(p/'PAIRED_AUDIT.json',dict(rows=paired,source_judge='Exact frozen existing labels; no new judging'))
    selected=sorted(s['tasks'][:19],key=lambda t:t['canonical_edit_id'])[:2]
    gpu=read(run/'public/GPU_BUDGET_LEDGER.json');judge=read(run/'public/BUDGET_LEDGER.json');used_gpu=sum(x['seconds'] for x in gpu['sessions']);assert used_gpu==14729.182081222534 and judge['new_judgment_items_dispatched']==51
    registration=dict(stage='24',parent='c85df124e99ba60bab1924f8a5ea9891b229134b',fixed=dict(writer='model.layers.30.mlp.down_proj',router='model.layers.31.mlp.up_proj',H=.25,U=.01,support='S0',rank=4,steps=320,lambda_x=.001,lambda_y=.001,sigma=1,epsilon=1e-5,observations=5,pooling='all valid expanded tokens; unchanged assistant predictor writer mask',regularizer_frequency='every update',X='frozen layer0 input',Z_patch='post-hook L30 down output',Z_final='same-forward layer31 block output'),mechanical_positions=[t['order'] for t in selected],mechanical_rule='first two canonical IDs within original19, before new semantic scores',mechanical_updates_per_case=3,CPU_only_counterfactual_selected=False,cumulative_GPU_limit=28800,cumulative_Judge_limit=2000,prior_GPU_seconds=used_gpu,prior_Judge_items=51,phase_GPU_caps={'24A':600,'24B':1200,'24C':4800,'25':5700,'recovery24':28800-used_gpu-12300},phase_Judge_caps={'24A':300,'24B':0,'24C':200,'25':1200,'recovery24':249},stage25_requires='DEV promotion plus qualified new panel plus conservative resource reserve',no_semantic_retry=True,old_CONFIRM='Viewed regression only for new methods',optional_counterfactual='Not scheduled; avoid optional GPU cost',new_confirm_status=summary['status']);registration['binding']=digest(registration);write_new(d/'PREREGISTRATION.json',registration)
    print(json.dumps(summary));print('mechanical_positions',registration['mechanical_positions'])

if __name__=='__main__':run()
