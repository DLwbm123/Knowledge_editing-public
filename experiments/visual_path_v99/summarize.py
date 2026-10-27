"""CPU-only aggregation of the frozen visual-path diagnostic."""
import json
from pathlib import Path
import sys
import torch


def read(p):return json.loads(Path(p).read_text())


def distance(a,b):
    a,b=a.double(),b.double();p,q=a.softmax(-1),b.softmax(-1);m=(p+q)/2
    return dict(max_logit_abs=float((a-b).abs().max()),total_variation=float((p-q).abs().sum()/2),
        JS_nats=float(((p*(p.log()-m.log())).sum()+(q*(q.log()-m.log())).sum())/2),
        first_token_changed=int(a.argmax())!=int(b.argmax()))


def extent(values):return dict(min=min(values),max=max(values))


def main(root,destination):
    root,destination=Path(root),Path(destination);cases=read(root/'CASES_FROZEN.json')
    outputs={route:{p.stem:read(p) for p in (root/'outputs'/route).glob('*.json')} for route in ['C','N','R']}
    assert all(len(rows)==12 for rows in outputs.values())
    assert all(x['EOS'] and not x['at_cap'] for rows in outputs.values() for x in rows.values())
    logits={route:torch.load(root/(route+'_LOGITS.pt'),map_location='cpu',weights_only=True) for route in outputs}
    detailed={};summaries={}
    for route,rows in outputs.items():
        image_rows=[rows[str(r['qid'])] for r in cases]
        assert all(x['trace']['calls']==dict(vision=1,projector=1) and x['trace']['image_markers']==1 and x['trace']['injected_visual_block_max_abs_difference']==0 for x in image_rows)
        assert all(x['trace']['calls']==dict(vision=0,projector=0) and x['trace']['image_markers']==0 for x in rows.values() if x['condition']=='noimage')
        assert all(x['raw_answer'].startswith(('Yes,','No,')) for x in rows.values())
        correct=sum(rows[str(r['qid'])]['raw_answer'].split(',')[0]==r['reference'] for r in cases)
        positive_correct=sum(rows[str(r['qid'])]['raw_answer'].startswith('Yes,') for r in cases if r['role']=='positive_direct')
        swapped=[];noimage=[]
        equal_text=equal_fact=0
        for i in range(0,8,2):
            negative,positive=cases[i:i+2];a,b=str(negative['qid']),str(positive['qid']);group=negative['source_group']
            swapped.append(distance(logits[route][a],logits[route][b]))
            equal_text+=rows[a]['raw_answer']==rows[b]['raw_answer']
            equal_fact+=rows[a]['raw_answer'].split(',')[0]==rows[b]['raw_answer'].split(',')[0]
            for key in [a,b]:noimage.append(distance(logits[route][key],logits[route][group+'_noimage']))
        detailed[route]=dict(image_swap=swapped,image_vs_noimage=noimage)
        summaries[route]=dict(image_correct=correct,image_total=8,positive_correct=positive_correct,positive_total=4,
            negative_correct=correct-positive_correct,negative_total=4,
            swapped_pair_same_full_text=equal_text,swapped_pair_same_fact_direction=equal_fact,pairs=4,
            image_swap_distribution={k:extent([x[k] for x in swapped]) for k in ['max_logit_abs','total_variation','JS_nats']},
            image_swap_first_token_changes=sum(x['first_token_changed'] for x in swapped),
            image_vs_noimage_distribution={k:extent([x[k] for x in noimage]) for k in ['max_logit_abs','total_variation','JS_nats']},
            image_vs_noimage_first_token_changes=sum(x['first_token_changed'] for x in noimage),
            noimage_yes_answers=sum(x['raw_answer'].startswith('Yes,') for x in rows.values() if x['condition']=='noimage'),
            all_visual_injection_checks_pass=True,all_noimage_bypass_checks_pass=True)
    parity={}
    for a,b in [('C','N'),('N','R'),('C','R')]:
        differences=[distance(logits[a][key],logits[b][key]) for key in outputs[a]]
        parity[a+'_'+b]=dict(full_text_equal=sum(outputs[a][k]['raw_answer']==outputs[b][k]['raw_answer'] for k in outputs[a]),
            fact_direction_equal=sum(outputs[a][k]['raw_answer'].split(',')[0]==outputs[b][k]['raw_answer'].split(',')[0] for k in outputs[a]),total=12,
            first_token_changes=sum(x['first_token_changed'] for x in differences),
            logit_max_abs_range=extent([x['max_logit_abs'] for x in differences]),
            caveat='Unaligned official one-pixel padding draws and precision/version differences are not causally separated')
    feature={**read(root/'PAIR_FEATURE_DIFFERENCES_current.json'),**read(root/'PAIR_FEATURE_DIFFERENCES_reference.json')}
    totals={k:0 for k in read(root/'RESOURCE_LIMITS.json')['counts']}
    for name in ['EXIT_current.json','EXIT_reference.json']:
        receipt=read(root/name);assert receipt['exit_code']==0
        for k,n in receipt['counts'].items():totals[k]+=n
    assert totals['evaluation_generations']==36 and totals['updates']==totals['backwards']==0
    leases=read(root/'RESOURCE_TARGETS.json')['targets'];assert all('ended_epoch' in t for t in leases)
    environment={mode:read(root/f'ENVIRONMENT_{mode}.json') for mode in ['current','reference']}
    result=dict(status='VISUAL_CHAIN_ACTIVE_DISCRIMINATION_NOT_RECOVERED',patients=8,source_pairs=4,
        independent_generations=36,condition_consumers=72,historical_current_token_matches=sum(outputs['C'][str(r['qid'])]['V98_exact_tokens'] for r in cases),
        routes=summaries,cross_route_parity=parity,
        native_preprocess_pixel_matches={route:sum(x['pixels_equal'] for x in read(root/(route+'_INPUT_PARITY.json')).values()) for route in ['N','R']},
        same_image_seed_official_pinned_CPU_exact_matches=sum(x['same_image_seed_max_abs_difference']==0 for x in read(root/'CPU_REFERENCE_PREPROCESS_ALIGNED.json')['comparisons']),
        pair_feature_ranges={key:extent([x[key] for x in feature.values()]) for key in next(iter(feature.values()))},
        counts=totals,gpu_process_seconds=sum(t['ended_epoch']-t['started_epoch'] for t in leases if t.get('gpu') is not None),
        environments={k:{key:v[key] for key in ['torch','transformers','tokenizers','forward_compat_installed']} for k,v in environment.items()},
        engineering_incidents=['Dependency receipt checked before package installation finished; completed current outputs retained',
            'Reference loader device cuda:0 requested an unsupported device_map; zero model calls; corrected to official default cuda'],
        all_EOS=True,no_method_training=True,
        inference='Visual features enter and alter calculations; three paths retain an absent-finding bias on these fixed cases. No general clinical capability or whole-model visual-blindness claim.')
    (root/'LOGIT_DISTANCES_PRIVATE.json').write_text(json.dumps(detailed,indent=2))
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main(*sys.argv[1:])
