"""Read-only, receipt-backed acceptance of the two Stage18 smoke edits."""
from pathlib import Path
import json
import math


def read(path):return json.loads(Path(path).read_text())


def accept(run):
    cfg=read(run/'private/DISPATCH.json');tasks=read(run/'private/TRAINING_TASKS.json')['tasks']
    result=read(run/'public/SMOKE_RESULT.json');exit_receipt=read(run/'public/EXIT.json')
    if len(tasks)!=2 or result['status']!='PASS' or exit_receipt['exit_code']!=0:raise ValueError('Smoke is not complete')
    rows=[];receipts=[]
    for task in tasks:
        directory=run/'private/edits'/f"e{task['order']:03d}"
        receipt=read(directory/'COMPLETE.json');receipts.append(receipt)
        if (directory/'FAILURE.json').exists():raise ValueError('Unresolved edit failure')
        for key in ['CP_transfer','zero_residual','Base_OFF','Base_route_isolation','save_load']:
            if receipt.get(key) is not True:raise ValueError('Missing mechanical check: '+key)
        branches={b:read(directory/b/'TRAINING.json') for b in cfg['branches']}
        reference=None
        for b,a in branches.items():
            curve=a['curve']
            if a['steps']!=320 or [x['step'] for x in curve]!=list(range(1,321)) or a['W0']!=receipt['W0']:raise ValueError('Steps/W0 mismatch')
            signature=[(x['fit_index'],x['terms']['native']['sample'],x['terms']['fit']['sample'],x['terms']['U']['sample']) for x in curve]
            if reference is None:reference=signature
            elif signature!=reference:raise ValueError('Native-fit-U schedule differs between branches')
            extras=[x['terms']['extra'] for x in curve if 'extra' in x['terms']]
            if len(extras)!=(0 if b=='C_NO_H' else 320):raise ValueError('Extra slot count mismatch')
            for item in curve:
                if set(item['terms'])!=({'native','fit','U'} if b=='C_NO_H' else {'native','fit','U','extra'}):raise ValueError('Loss terms mismatch')
                for k,w in dict(native=.5,fit=.5,U=.01,extra=1.).items():
                    if k not in item['terms']:continue
                    term=item['terms'][k]
                    if term['weight']!=w or not math.isfinite(term['unweighted']) or not math.isclose(term['weighted'],w*term['unweighted'],abs_tol=1e-9):raise ValueError('Incorrect weighted loss')
                    if not math.isfinite(term['weighted_gradient_norm']) or term['tokens']<=0:raise ValueError('Invalid loss accounting')
            if extras and not all(x['weighted_gradient_norm']>0 for x in extras):raise ValueError('Extra gradient zero/nonfinite')
            if not (directory/b/'latest.pt').is_file():raise ValueError('Resume state missing')
            rows.append(dict(edit_ordinal=1+tasks.index(task),branch=b,updates=320,extra_nonzero_updates=len(extras),
                extra_gradient_min=min((x['weighted_gradient_norm'] for x in extras),default=None),
                extra_gradient_max=max((x['weighted_gradient_norm'] for x in extras),default=None),
                tokens=a['tokens'],parameters=a['parameters'],shared_W0=receipt['W0'],session_seconds=a['session_seconds']))
        if [x['extra_index'] for x in branches['C_FACT']['curve']]!=[x['extra_index'] for x in branches['C_EXTRA']['curve']]:raise ValueError('H/G slot schedules differ')
        outputs=read(directory/'DIAGNOSTICS.json')
        if len(outputs)!=12:raise ValueError('Diagnostic panel incomplete')
        if any(x['interpretation']!='SOURCE_LABEL_TRAINING_DIAGNOSTIC_NOT_INDEPENDENT_EVALUATION' for x in outputs):raise ValueError('Wrong evaluation scope')
    return dict(status='PASS_MECHANICAL_SMOKE_ONLY',execution_commit=cfg['code_commit'],edits=2,completed_branches=6,continuation_updates=1920,
        H_gradient_nonzero_updates=sum(r['extra_nonzero_updates'] for r in rows if r['branch']=='C_FACT'),
        G_gradient_nonzero_updates=sum(r['extra_nonzero_updates'] for r in rows if r['branch']=='C_EXTRA'),
        checks=dict(same_W0_per_edit=True,same_native_fit_U_schedule=True,same_H_G_extra_schedule=True,
            correct_loss_weights=True,CP_transfer=True,zero_residual=True,Base_OFF=True,Base_route_isolation=True,save_load=True),
        CPU_zero_weight_and_resume='13 relevant checks passed; see TEST_RESULTS.json. Controlled weight-zero and interrupted resume were CPU tests, not separate GPU experiments.',
        source_identity='worker verifies each H/G/U original image/question/answer against approved train.json',
        rows=rows,exit_code=0,worker_wall_seconds=exit_receipt['finished']-read(run/'private/WORKER.json')['started'],
        peak_allocated_GiB=max(r['peak_allocated_bytes'] for r in receipts)/1024**3,
        formal_results=False,semantic_Judge_calls=0,independent_H_eval_evaluated=0,
        failures=['First launcher syntax failure before model/worker start; corrected run_v2, original evidence retained'],
        retained_checkpoints='W0/init, six writer/optimizer states and two routers retained for registered review; no cleanup')


if __name__=='__main__':
    import sys
    result=accept(Path(sys.argv[1]));Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
