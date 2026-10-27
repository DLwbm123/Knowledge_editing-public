"""Complete SOURCE input-by-expert matrix; append blind codes without renumbering."""
import json
from pathlib import Path
def read(p):return json.loads(Path(p).read_text())
def coverage(actual,expected):
    assert len(actual)==len(set(actual)) and set(actual)==set(expected),'Missing/extra/duplicate bank cell'
def collect(root):
    plan=read(root/'ROUTE_PLAN_FROZEN.json');cells=dict(plan['existing_source_outputs'])
    actual=[p.parent.name+'|'+p.stem for p in (root/'bank/outputs').glob('*/*.json')]
    coverage(actual,plan['missing_pairs'])
    for pair in actual:
        e,q=pair.split('|');cells[pair]=str(Path('bank/outputs')/e/(q+'.json'))
    expected={t['key']+'|'+q for t in plan['tasks'] for q in plan['queries']};coverage(list(cells),expected)
    counts=dict(forwards=0,backwards=0,updates=0,generations=0,vision_forwards=0,query_keys=0);seconds=0
    for part in range(5):
        x=read(root/'bank'/str(part)/'EXIT.json');assert x['code']==0
        assert read(root/'bank'/str(part)/'MODEL_CONSUMERS_COMPLETE.json')['status']=='PASS'
        for k in counts:counts[k]+=x['counts'][k]
        seconds+=x['seconds']
    assert counts['generations']==1584 and counts['backwards']==counts['updates']==counts['query_keys']==0
    assert counts['forwards']<=1750000 and seconds<=10*3600
    for pair,path in cells.items():
        q=pair.split('|')[1];row=plan['queries'][q];x=read(root/path);b=x['binding']
        assert all(b[k]==row[k] for k in ('image_sha256','question','reference'))
        assert x['raw_token_ids'] and isinstance(x['raw_answer'],str)
    result=dict(status='MODEL_CONSUMERS_COMPLETE_NOT_SCORED',cells=cells,counts=counts,GPU_process_seconds=seconds,reused=216,new=1584)
    (root/'BANK_CELLS.json').write_text(json.dumps(result,indent=2)+'\n');return result
def blind(root):
    # Main packet and judgments must be complete first; existing codes never change.
    from consumers_v22 import summarize
    summarize(root);matrix=collect(root)
    packet=read(root/'BLIND_PACKET.json');mapping=read(root/'BLIND_MAPPING.json');old={}
    for path,code in mapping.items():
        x=read(root/path);b=x['binding'];old[(b['image_sha256'],b['question'],b['reference'],x['raw_answer'])]=code
    items=list(packet['items']);known=set(mapping.values())
    for pair,path in sorted(matrix['cells'].items()):
        x=read(root/path);b=x['binding'];key=(b['image_sha256'],b['question'],b['reference'],x['raw_answer'])
        if key not in old:
            code='B%06d'%(len(items)+1);assert code not in known;old[key]=code;known.add(code)
            items.append(dict(code=code,question=b['question'],reference=b['reference'],answer=x['raw_answer']))
        mapping[path]=old[key]
    packet=dict(packet,status='MAIN_AND_BANK_HIDDEN',items=items)
    (root/'BANK_BLIND_PACKET.json').write_text(json.dumps(packet,indent=2)+'\n')
    (root/'BANK_BLIND_MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
    return dict(main_codes_preserved=True,new_unique_judgments=len(items)-len(read(root/'BLIND_PACKET.json')['items']),cells=len(matrix['cells']))
def selfcheck():
    coverage(['a','b'],{'a','b'})
    for x in (['a'],['a','a','b'],['a','c']):
        try:coverage(x,{'a','b'})
        except AssertionError:pass
        else:raise AssertionError('Bad bank coverage accepted')
    return dict(status='PASS',missing_extra_duplicate_rejected=True)
if __name__=='__main__':print(selfcheck())
