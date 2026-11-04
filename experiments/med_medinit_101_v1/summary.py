"""Same-support macro and paired counts, grouped by frozen route or owner."""
import random
import statistics

ROLES=('T0','FIT','T1G','T2G','T1L','T2L')
ARMS=('MED','MCE','MDU')
FIELDS=('core_correct','full_supported')

def summarize(rows):
    grouped={}
    for owner,value in rows:
        if value is not None and type(value) is not bool:raise ValueError('invalid score')
        grouped.setdefault(owner,[]).append(value)
    if not grouped:return {'events':0,'edits':0,'unknown':0,'bounds':None}
    return {'events':len(rows),'edits':len(grouped),'unknown':sum(v is None for _,v in rows),
            'bounds':[100*statistics.mean(sum(v is True for v in vs)/len(vs) for vs in grouped.values()),
                      100*statistics.mean(sum(v is not False for v in vs)/len(vs) for vs in grouped.values())]}

def compare(consumers,decisions,roles,routing):
    if roles.get('status')!='EVALUATION_ROLES_FROZEN' or roles.get('owners')!=150 or routing.get('status')!='ROUTING_FROZEN':
        raise ValueError('role or route lock missing')
    pairs={};routes={r['query_id']:r for r in routing['records']}
    if len(routes)!=2148:raise ValueError('routing identity differs')
    admitted=set(routing['admitted_owners'])
    if len(admitted)!=101:raise ValueError('admitted owner coverage differs')
    for row in consumers:
        key=(row['arm'],row['query_id'])
        if key in pairs:raise ValueError('duplicate arm/query')
        pairs[key]=row['key']
    sets=[{q for a,q in pairs if a==arm} for arm in ARMS]
    if any(x!=set(routes) for x in sets):raise ValueError('arm identities differ')
    selectors={'all':lambda o,q:True,'original25_route':lambda o,q:routes[q]['group']=='original25',
               'new76_route':lambda o,q:routes[q]['group']=='new76','med49_route':lambda o,q:routes[q]['group']=='med49',
               'owner_admitted101':lambda o,q:o in admitted,'owner_unresolved49':lambda o,q:o not in admitted,
               'owner_original146':lambda o,q:o<=146,'owner_added4':lambda o,q:o>146}
    out={}
    for group,select in selectors.items():
        out[group]={}
        for role in ROLES:
            entries=[(o,q) for o in range(1,151) for q in roles['roles'][str(o)][role] if q in routes and select(o,q)]
            out[group][role]={}
            for field in FIELDS:
                values={a:[(o,decisions.get(pairs[a,q],{}).get(field)) for o,q in entries] for a in ARMS}
                comparison={}
                for left,right in (('MED','MCE'),('MED','MDU'),('MCE','MDU')):
                    gain=loss=unknown=0;by_owner={}
                    for (owner,old),(_,new) in zip(values[left],values[right]):
                        if old is None or new is None:unknown+=1
                        else:
                            gain+=old is False and new is True;loss+=old is True and new is False
                            by_owner.setdefault(owner,[]).append(int(new)-int(old))
                    interval=None
                    if entries and unknown==0:
                        deltas=[statistics.mean(x) for x in by_owner.values()];rng=random.Random(20261104)
                        samples=sorted(100*statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(10000))
                        interval=[samples[249],samples[9749]]
                    comparison[left+'_to_'+right]={'gain':gain,'loss':loss,'unknown':unknown,'edit_bootstrap_95':interval}
                out[group][role][field]={'arms':{a:summarize(rows) for a,rows in values.items()},'pairs':comparison}
    return {'status':'PAIRED_DEV_SUMMARY','events':2148,'groups':out,'bootstrap_seed':20261104,
            'bootstrap_note':'Edit-level resampling, not accounting for common routed expert or source dependence.'}
