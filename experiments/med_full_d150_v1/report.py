"""Aggregate fixed role memberships and preserve score uncertainty."""
import json
import statistics
from pathlib import Path

ROLES=('T0','FIT','T1G','T2G','T1L','T2L')
FIELDS=('core_correct','full_supported')


def summarize(rows):
    by={}
    for owner,value in rows:
        if value not in (True,False,None):raise ValueError('invalid verdict')
        by.setdefault(owner,[]).append(value)
    if not by:return {'events':0,'support_edits':0,'correct':0,'unknown':0,'macro_percent':None,'bounds':None}
    unknown=sum(value is None for _,value in rows)
    lower=100*statistics.mean(sum(v is True for v in values)/len(values) for values in by.values())
    upper=100*statistics.mean(sum(v is not False for v in values)/len(values) for values in by.values())
    return {'events':len(rows),'support_edits':len(by),'correct':sum(v is True for _,v in rows),
            'unknown':unknown,'macro_percent':lower if not unknown else None,'bounds':[lower,upper]}


def run(packet,decisions,roles,scoring,cohorts=None):
    if packet.get('status')!='SCORE_PACKET_FROZEN' or len(packet['consumers'])!=4296:
        raise ValueError('score packet not complete')
    if roles.get('status')!='EVALUATION_ROLES_FROZEN' or roles.get('owners')!=150:
        raise ValueError('role identity lock missing')
    pairs={}
    for row in packet['consumers']:
        key=(row['arm'],row['query_id'])
        if key in pairs:raise ValueError('duplicate bank score consumer')
        pairs[key]=row['key']
    if len(pairs)!=4296:raise ValueError('bank score pair coverage mismatch')
    result={}
    cohorts = cohorts or {}
    selections = [('all',range(1,151)),('original',range(1,147)),('additional',range(147,151))]
    selections.extend(cohorts.items())
    for cohort,selected in selections:
        result[cohort]={}
        for role in ROLES:
            entries=[(o,q) for o in selected for q in roles['roles'][str(o)][role]]
            result[cohort][role]={}
            for field in FIELDS:
                a=[];b=[];gain=loss=unknown_pairs=0
                for owner,q in entries:
                    if ('MED',q) not in pairs or ('MED_D',q) not in pairs:raise ValueError('role consumer missing')
                    old=decisions.get(pairs['MED',q],{}).get(field)
                    new=decisions.get(pairs['MED_D',q],{}).get(field)
                    a.append((owner,old));b.append((owner,new))
                    if old is None or new is None:unknown_pairs+=1
                    else:gain+=old is False and new is True;loss+=old is True and new is False
                result[cohort][role][field]={'MED':summarize(a),'MED_D':summarize(b),
                    'gained':gain,'lost':loss,'unknown_pairs':unknown_pairs}
    def gate(role,threshold,strict=False):
        p=result['all'][role]['core_correct'];a=p['MED']['bounds'];b=p['MED_D']['bounds']
        if a is None or b is None:return None
        if strict:
            if b[0]>a[1]+1e-10:return True
            if b[1]<=a[0]+1e-10:return False
        else:
            if b[0]>=a[1]-threshold-1e-10:return True
            if b[1]<a[0]-threshold-1e-10:return False
        return None
    gates={r:gate(r,5 if r=='T2L' else 0) for r in ('T0','FIT','T1G','T1L','T2L')}
    gates['T2G_improved']=gate('T2G',0,strict=True)
    status='DEV_FAIL' if any(v is False for v in gates.values()) else ('INCONCLUSIVE' if any(v is None for v in gates.values()) else 'DEV_SIGNAL')
    return {'status':status,'events':2148,'panels':result,'gates':gates,'scoring':scoring,
            'candidate_experts':25,'fallback_experts':125,'independent_confirmation':False}
