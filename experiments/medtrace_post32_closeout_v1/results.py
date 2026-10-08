"""Anonymous paired summaries. Seeds are repeated optimization, not new sources."""
import os,sys
from pathlib import Path
RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import core
common=core.common;report=core.tool("report")
import csv

def main():
    records,scores=report.load();ts=[t for t in common.read(RUN/'private/QUEUES.json')['tasks'] if t['cohort']=='P2'];panels=[];coeff={};groups={}
    for prefix in [8,24]:
        for arm in ['FROZEN_W0','CE_U_MULTI','U_ONLY','HOLD_U']:
            for task in ['T0','T1G','T2G','T1L','T2L']:
                m,c,g=report.panel(records,scores,'P2',arm,0 if arm=='FROZEN_W0' else 160,'bank_R0',task,ts[:prefix],prefix);m.update(seed_repeats=2,interpretation='MECHANISM_DEV_ONLY' if prefix==8 else 'BACKGROUND_PRESSURE_PARTIAL_FIRST8_UPDATE');panels.append(m);coeff[(prefix,arm,task)]=c;groups.update(g)
    fixed_on_records=[]
    for d,k in records:
        b=d['binding'];phase=b['phase'];q=b['input']
        if b['mode']=='bank_R0' and q.get('role') in ('CAL','CHECK'):
            fixed=common.read(RUN/'private'/('FIXED_ON_'+str(phase['prefix'])+'_'+str(phase['slot'])+'.json'))['query_ids']
            if q['query_id'] in fixed:fixed_on_records.append((d,k))
    contrasts=[]
    for prefix in [8,24]:
        for arm in ['CE_U_MULTI','U_ONLY','HOLD_U']:
            for reference in ['FROZEN_W0','CE_U_MULTI']:
                if arm==reference:continue
                for task in ['T0','T1G','T2G']:
                    a=coeff[(prefix,arm,task)];b=coeff[(prefix,reference,task)];paired={e:report.combine([a[e],{k:-v for k,v in b[e].items()}]) for e in a.keys()&b.keys()};bounds=report.score_bounds(report.combine([{k:v/len(paired) for k,v in c.items()} for c in paired.values()]),scores);contrasts.append(dict(prefix=prefix,comparison=arm+'-'+reference,task=task,delta_bounds_pp=[v*100 for v in bounds],paired_edits=len(paired),**report.bootstrap(paired,scores,groups)))
    common.write(RUN/'public/CORE_RESULTS.json',dict(panels=panels,contrasts=contrasts,U=report.summarize_u(records,'P2'),U_fixed_W0_ON=report.summarize_u(fixed_on_records,'P2'),medical_accuracy=None,Base_correct_retention=None,scope_qualified_sources=0,protection_promotion=False))
    with (RUN/'public/CORE_RESULTS.csv').open('w',newline='') as f:
        keys=['arm','prefix','task','macro','known_correct','observations','missing','route_ON'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(panels)
    common.write(RUN/'public/VERSION_LOCK.json',dict(selected='FROZEN_W0_R0',reason='No qualified normal-bank protection improvement evidence; old CHECK is DEV only',training_completed=True,core_evaluation_completed=True,independent_confirmation_completed=False,benchmark146_completed=False,final_comparison='PENDING'))
if __name__=='__main__':main()
