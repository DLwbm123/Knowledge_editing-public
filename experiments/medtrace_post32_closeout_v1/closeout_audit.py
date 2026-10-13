"""Public aggregates only; retain per-edit evidence in the private run."""
import sqlite3
import core

common, RUN, BASE = core.common, core.RUN, core.BASE


def main():
    report = core.tool('report')
    records, scores = report.load()
    tasks = [t for t in common.read(RUN/'private/QUEUES.json')['tasks'] if t['cohort']=='P2']
    panels, training = [], []
    for slot in (0, 1):
        rs = [(d,k) for d,k in records if d['binding']['phase']['slot']==slot]
        for prefix in (8, 24):
            for arm in ('FROZEN_W0','CE_U_MULTI','U_ONLY','HOLD_U'):
                for task in ('T0','T1G','T2G'):
                    m,_,_ = report.panel(rs,scores,'P2',arm,0 if arm=='FROZEN_W0' else 160,'bank_R0',task,tasks[:prefix],prefix)
                    m['continuation_seed_slot']=slot;panels.append(m)
        for arm in ('CE_U_MULTI','U_ONLY','HOLD_U'):
            root = BASE if arm=='CE_U_MULTI' and slot==0 else RUN
            rows = [common.read(root/'private/edits'/t['anonymous_edit']/('s'+str(slot))/arm/'TRAINING.json') for t in tasks[:8]]
            assert all(x['status']=='COMPLETE' and x['actual_updates']==160 for x in rows)
            curve = [s for r in rows for s in r['curve']]
            avg = lambda xs: sum(xs)/len(xs) if xs else None
            def term(which, endpoint):
                return avg([r['curve'][endpoint]['terms'][which]['unweighted'] for r in rows if which in r['curve'][endpoint]['terms']])
            training.append(dict(arm=arm,continuation_seed_slot=slot,trajectories=8,
                reused=arm=='CE_U_MULTI' and slot==0,updates=len(curve),
                update_forwards=sum(s['forwards'] for s in curve),update_backwards=sum(s['backwards'] for s in curve),
                actual_U_tokens=sum(s['terms']['U']['tokens'] for s in curve),
                update_tokens=sum(s['train_tokens'] for s in curve),
                first_native_CE=term('native',0),last_native_CE=term('native',-1),
                first_fit_CE=term('fit',0),last_fit_CE=term('fit',-1),
                first_U_KL=term('U',0),last_U_KL=term('U',-1),
                hinge_active_fraction=sum(s.get('hinge_active',0) for s in curve)/(2*len(curve)) if arm=='HOLD_U' else None,
                nonzero_parameter_update_steps=sum(any(v>0 for v in s['core_updates'].values()) for s in curve),
                mean_final_function_change=avg([r['curve'][-1]['net_function_change'] for r in rows]),
                mean_function_path=avg([r['curve'][-1]['cumulative_function_path'] for r in rows])))
    db=sqlite3.connect('file:'+str(RUN/'private/judge_common/queue.sqlite')+'?mode=ro',uri=True)
    counts=dict(db.execute('SELECT status,count(*) FROM payload GROUP BY status'))
    consumers=db.execute('SELECT count(*) FROM consumer').fetchone()[0];db.close()
    common.write(RUN/'public/CLOSEOUT_AUDIT.json',dict(seed_panels=panels,training=training,
        scoring=dict(status_counts=counts,consumers=consumers),new_trajectories=40,reused_trajectories=8,
        shared_W0=True,independent_warm_starts=False,parameters=7168,
        counters='Optimization forwards/backwards/tokens; teacher, baseline and native-test diagnostics are additional GPU cost',
        U_ONLY_edit_CE='Not evaluated inside the U-only update; NA rather than fabricated',
        U_gradient_counter='Legacy U_nonzero_updates is inapplicable to new arms; use native gradient tests and recorded nonzero parameter/function changes',
        provenance='Initial embedded execution manifest predates import/receipt repairs; later deployment provenance retained separately, completed bindings unchanged'))


if __name__=='__main__':
    main()
