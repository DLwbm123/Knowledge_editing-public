"""Rebuild the frozen result and classify unknown semantic payloads."""
import json, os
from pathlib import Path
w=Path(os.environ['WORKTREE'])
s=(w/'experiments/med_medinit_101_v1/score_driver.py').read_text()
a=s.index('    packets={}\n'); b=s.index('    legacy={k for p in packets.values()',a)
s=s[:a]+'''    packets={arm:json.loads((Path(os.environ['OLD_SCORE_ROOT'])/('PACKET_'+arm+'.private.json')).read_text()) for arm in ('MCE','MDU')}
'''+s[b:]
a=s.index("    write('RESULTS.json',result)")
s=s[:a]+'''    expected=json.loads((worktree / 'reports/med_medinit_101_v1_20261104/RESULTS.json').read_text())
    if result != expected: raise ValueError('frozen baseline does not reproduce')
    targets={r['key'] for r in consumers if r['key'] not in decisions}
    payloads={}; categories={}; conflicts=set()
    for arm,p in packets.items():
        part=score_reuse.resolve(p,history)
        for category in ('history_unverified','conflict_unknown','known_unknown'):
            for key in set(part[category]) & targets:
                row=p['payloads'][key]['record']
                if key in payloads and payloads[key]!=row: raise ValueError('payload mismatch')
                payloads[key]=row
                if key in categories and categories[key]!=category: conflicts.add(key)
                else: categories[key]=category
    for key in conflicts: categories[key]='classification_conflict'
    unbound=targets-set(payloads)
    write('BASELINE.private.json',{'consumers':consumers,'decisions':decisions,'routing':routing})
    write('SUPPLEMENT_INPUT.private.json',{'status':'SUPPLEMENT_IDENTITIES_FROZEN','payloads':payloads,'categories':categories,'unbound':sorted(unbound)})
    from collections import Counter
    write('PREPARE_COMPLETE.json',{'status':'BASELINE_REPRODUCED','targets':len(targets),'bound':len(payloads),'unbound':len(unbound),'categories':dict(Counter(categories.values()))})
    return
'''+s[s.index("if __name__=='__main__':",a):]
s=s.replace('worktree /', 'w /')
exec(compile(s,'<reconstructed-frozen-score>','exec'),{'__name__':'__main__'})
