"""Exact finite output coverage, missing/duplicate rejection and bounded lifecycle."""
import json
from pathlib import Path
ARMS=('SOURCE','CE','KL','CE_PROJ','RESERVE','BALANCEDIT','BALANCEDIT_CE')
def read(p):return json.loads(Path(p).read_text())
def coverage(actual,expected):
    assert len(actual)==len(set(actual)) and set(actual)==set(expected),'Missing, extra or duplicate output consumer'
    return len(actual)
def validate(root):
    config=read(root/'DATA_ADMISSION.json');queries=read(root/'QUERY_ROWS.json');expected=[];actual=[]
    for task in config['tasks']:
        part=(task['order']-1)%5;folder=root/str(part)/task['key']
        for arm in ARMS:
            expected.extend((task['key'],arm,r['query_id']) for r in task['eval_rows'])
            for p in (folder/'outputs'/arm).glob('*.json'):actual.append((task['key'],arm,p.stem))
    return dict(method_outputs=coverage(actual,expected),base_outputs=coverage([p.stem for p in (root/'base').glob('*/*.json')],queries),scientific_scoring='REQUIRES_COMPLETE_HIDDEN_VERDICTS')
def retire_edit(folder,write):
    import torch
    assert read(folder/'MODEL_CONSUMERS_COMPLETE.json')['status']=='PASS'
    # Preserve SOURCE active/final for the registered source-bank and recovery consumers.
    paths=[folder/a/n for a in ARMS[1:5] for n in ('active.pt','final.pt')]+[folder/(a+'.pt') for a in ARMS[5:]]
    anchors={a:torch.load(folder/(a+'.pt'),map_location='cpu',weights_only=True)['router'] for a in ARMS[5:]}
    torch.save(anchors,folder/'R0_ANCHORS.pt')
    assert all(p.is_file() for p in paths)
    deleted=[dict(file=str(p.relative_to(folder)),bytes=p.stat().st_size) for p in paths]
    for p in paths:p.unlink()
    write(folder/'LIFECYCLE.json',dict(status='RETIRED_NON_SOURCE',deleted=deleted,SOURCE='Retained only for registered bank consumer',last_consumers='all seven-arm answers, bindings, routes, save-load'))
def selfcheck():
    assert coverage([('e','a','q')],[('e','a','q')])==1
    for bad in [[],[('e','a','q')]*2,[('e','a','z')]]:
        try:coverage(bad,[('e','a','q')])
        except AssertionError:pass
        else:raise AssertionError('coverage accepted invalid')
    assert len(ARMS)==7
    return dict(status='PASS',missing_duplicate_extra_rejected=True,all_seven_required=True)

def blind(root,partial=False):
    """Private hidden-method packet. Equal input/reference/answer shares one verdict."""
    if not partial:validate(root)
    queries=read(root/'QUERY_ROWS.json');paths=list(sorted((root/'base').glob('*/*.json')))
    paths+=list(sorted(root.glob('[0-4]/E*/outputs/*/*.json')))
    assert paths
    historical=read(root/'EVAL_BINDINGS.json');old={}
    for x in historical.values():
        key=(x['image_sha256'],x['question'],x['reference'],x['output']['model_answer_raw'])
        old[key]=x['query_id']
    mask=read(root/'ORIGINAL_BASE_MASK146.json');unique={};items=[];mapping={};reuse={}
    for path in paths:
        x=read(path);b=x['binding'];key=(b['image_sha256'],b['question'],b['reference'],x['raw_answer'])
        if key not in unique:
            code='B%06d'%(len(unique)+1);unique[key]=code
            items.append(dict(code=code,question=b['question'],reference=b['reference'],answer=x['raw_answer']))
            # Frozen prior correct answers only; old binary false is not four-class evidence.
            if key in old and mask.get(old[key]) is True:reuse[code]=dict(label='CORRECT',reason='Exact prior input/reference/answer with frozen correct verdict')
        mapping[str(path.relative_to(root))]=unique[key]
    packet=dict(status='PARTIAL_DEVELOPMENT_PACKET' if partial else 'ALL_OUTPUTS_HIDDEN',classes=['CORRECT','INCORRECT','UNCERTAIN','ABSTAIN'],instruction='Judge source-answer agreement only against the fixed reference; equivalent wording allowed, contradiction/wrong entity/polarity incorrect; ambiguous unsupported specificity uncertain; refusal abstain. Do not infer method or repair reference.',items=items)
    (root/'BLIND_PACKET.json').write_text(json.dumps(packet,indent=2));(root/'BLIND_MAPPING.json').write_text(json.dumps(mapping,indent=2));(root/'BLIND_REUSE.json').write_text(json.dumps(reuse,indent=2))
    return dict(outputs=len(paths),unique_answers=len(items),exact_correct_reuse=len(reuse),partial=partial)

def summarize(root):
    validate(root)
    mapping=read(root/'BLIND_MAPPING.json');verdicts=read(root/'BLIND_VERDICTS.json');reuse=read(root/'BLIND_REUSE.json');merged=dict(reuse);merged.update(verdicts)
    assert set(merged)==set(mapping.values()),'All unique hidden judgments required; no missing/extra verdicts'
    assert all(x['label'] in ('CORRECT','INCORRECT','UNCERTAIN','ABSTAIN') and x.get('reason') for x in merged.values())
    def label(path):return merged[mapping[str(path.relative_to(root))]]['label']
    config=read(root/'DATA_ADMISSION.json');queries=read(root/'QUERY_ROWS.json');base={q:root/'base'/str(i%5)/(q+'.json') for i,q in enumerate(sorted(queries))};table={}
    for arm in ARMS:
        from collections import Counter,defaultdict
        roles=defaultdict(Counter);damage=Counter();repair=Counter();steps=Counter();caps=0
        for task in config['tasks']:
            folder=root/str((task['order']-1)%5)/task['key']
            if arm in ARMS[:5]:
                tr=read(folder/arm/'TRAIN_COMPLETE.json');steps[tr['status']]+=1;steps['accepted_updates']+=tr['accepted_steps']
            else:steps['COMPLETE']+=1;steps['accepted_updates']+=read(folder/(arm+'_TRAIN.json'))['steps']
            for row in task['eval_rows']:
                q=row['query_id'];p=folder/'outputs'/arm/(q+'.json');v=label(p);b=label(base[q]);caps+=bool(read(p)['at_cap'])
                for role in row['roles']:
                    roles[role][v]+=1
                    if role in ('T1L','T2L') and b=='CORRECT':
                        damage['Base_correct_denominator']+=1;damage['lost_correct']+=v!='CORRECT';damage['explicit_incorrect']+=v=='INCORRECT'
                    if role in ('T0','T1G','T2G') and b=='INCORRECT':repair['Base_explicit_incorrect_denominator']+=1;repair['repaired']+=v=='CORRECT'
        table[arm]=dict(roles={k:dict(v) for k,v in roles.items()},damage=dict(damage),repair=dict(repair),training=dict(steps),at_cap=caps)
    result=dict(status='COMPLETE_SCORED_DEVELOPMENT',table=table,independent_confirmation=False,source_bank_routing='SEPARATE_PENDING_CONSUMER')
    (root/'RESULTS.json').write_text(json.dumps(result,indent=2));return result
