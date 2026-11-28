"""Preflight, calibration, and matched writer training for the dense branch."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import statistics
import sys

import torch

from .dense_writer import dense_initialization, input_scale, principal_angles
from .experiment import bound_solution, json_read, json_write, load_state, write_state
from .features import load_manifest
from .runtime import FrozenRuntime
from .train_value import gradient_ratio, mounted, positive, train, validate_kl


KU_ARMS = ('V-U','V-U-ft')
CONTROLS = ('V-SGD-dense','V-SGD-dense-vis','V-SGD-TT')
MULTIPLIERS = (.3,1.,3.)


def digest(path):
    with Path(path).open('rb') as stream:
        return sha256(stream.read()).hexdigest()


def p4_schedules(plan, pilot):
    rows = defaultdict(list)
    for row in plan['rows']:
        if row.get('owner') is not None:
            rows[row['owner']].append(row)
    result = {}
    for edit in pilot['entries']:
        owner = edit['owner']
        entries = rows[owner]
        by_text = {r['question']:r['id'] for r in entries if r['role'] in {'fit','train_eq'}
                   and r.get('transform','identity')=='identity'}
        negatives = {r['id'] for r in entries if r['role']=='train_ind'}
        neg_ids = [f"{owner}-{r['query_id']}" for r in edit['negative']['train']]
        if len(neg_ids)!=len(negatives) or set(neg_ids)!=negatives:
            raise ValueError('P4 independent-negative binding differs')
        native, = [r['id'] for r in entries if r['role']=='native']
        schedule = []
        for n,step in enumerate(edit['steps'],1):
            if step['step']!=n or step['native_source_index']!=0:
                raise ValueError('P4 schedule ordinal differs')
            views = [by_text[edit['views'][i]['question']] for i in step['expanded_view_indices']]
            if views != plan['schedules'][str(owner)][n-1]['views']:
                raise ValueError('P4 expanded-view ordering differs')
            local = [neg_ids[i] for i in step['independent_train_negative_indices']]
            schedule.append(dict(native=native, views=views, negatives=local))
        if len(schedule)!=160 or owner in result:
            raise ValueError('Expected 160 unique P4 steps')
        result[owner] = schedule
    if set(result)!=set(plan['stage1_owners']):
        raise ValueError('P4 owner set differs')
    return result


def references(layout):
    refs = {}
    for path in layout['references']:
        shard = load_state(path)
        if refs.keys() & shard.keys():
            raise ValueError('Overlapping reference state shards')
        refs.update(shard)
    return refs


def initializations(solutions, refs):
    from .scope_solver import input_matrix
    result = {}
    for owner, info in solutions.items():
        p4 = refs[owner]['P4']
        torch.testing.assert_close(input_matrix(p4['G3'].double(),p4['G4'][:,:,0].double()),
                                   info['K-P4']['A'])
        result[owner] = dense_initialization(info,p4)
    return result


def calibration_selection(candidates, cap):
    # Joint admission takes precedence even when an earlier candidate admits one arm.
    passing = [(row['beta'],[arm for arm in KU_ARMS if row['medians'][arm]<=cap]) for row in candidates]
    for expected in (2,1):
        for beta, arms in passing:
            if len(arms)==expected:
                return dict(beta=beta,ku_arms=arms,training_arms=arms+list(CONTROLS))
    return dict(beta=None,ku_arms=[],training_arms=[])


def calibrate(runtime, mode, owners, inputs, events, schedule, refs, lock, lane):
    by_owner = defaultdict(list)
    for row in events.values():
        if row.get('owner') in owners and row['role'] in {'val_pos','val_neg'}:
            by_owner[row['owner']].append(row)
    if any({r['role'] for r in by_owner[o]}!={'val_pos','val_neg'} for o in owners):
        raise ValueError('Missing development validation support')
    if mode=='warm':
        output = {}
        for o in owners:
            runtime.seed(runtime.by_order[o]['seed'])
            pre,_ = train(runtime,runtime.expert(o,dense=True),dict(A=inputs[o]['ku']),
                          events,schedule[o][:8],0.,False)
            p,n = gradient_ratio(runtime,runtime.expert(o,pre),by_owner[o])
            control = runtime.expert(o,refs[o]['P4'])
            cap = [validate_kl(runtime,r,control) for r in by_owner[o] if r['role']=='val_neg']
            output[o] = dict(positive=p,negative=n,p4=cap)
            print('WARM',o,flush=True)
    else:
        warm = json_read('scale.json')
        if warm['lock']!=lock:
            raise ValueError('Warmup binding differs')
        output = {}
        for o in owners:
            output[o] = {}
            for multiplier in MULTIPLIERS:
                beta = warm['scale'] * multiplier
                output[o][str(multiplier)] = {}
                for arm in KU_ARMS:
                    runtime.seed(runtime.by_order[o]['seed'])
                    trained,_ = train(runtime,runtime.expert(o,dense=True),dict(A=inputs[o]['ku']),
                                      events,schedule[o],beta,arm=='V-U-ft',read_lr=inputs[o]['lr'])
                    editor = runtime.expert(o,trained)
                    validation = [validate_kl(runtime,r,editor) for r in by_owner[o] if r['role']=='val_neg']
                    output[o][str(multiplier)][arm] = validation
                    print('GRID',o,multiplier,arm,flush=True)
            json_write(f'grid-edit-{o}.json',dict(lock=lock,values=output[o]))
    json_write(f'{mode}{lane}.json',dict(lock=lock,owners=output))


def merged(mode, lock, expected):
    output = {}
    for i in (0,1):
        receipt = json_read(f'{mode}{i}.json')
        if receipt['lock']!=lock or output.keys() & receipt['owners'].keys():
            raise ValueError('Calibration binding or disjointness failed')
        output.update(receipt['owners'])
    if set(map(int,output))!=set(expected):
        raise ValueError('Calibration support differs')
    return output


def run():
    stage = sys.argv[1]
    layout = json_read('layout.json')
    plan = load_manifest(layout['manifest'])
    solutions = bound_solution(layout['solution'],layout['manifest'])
    refs = references(layout)
    pilot = json_read(layout['pilot'])
    schedules = p4_schedules(plan,pilot)
    inputs = initializations(solutions,refs)
    scale_keys = ('lr','ku_scale','med_scale','fit_target')
    registration = dict(manifest=digest(layout['manifest']), solution=digest(layout['solution']),
                        pilot=digest(layout['pilot']), p4_reference=[digest(p) for p in layout['references']],
                        per_editor={str(o):{k:v[k] for k in scale_keys} for o,v in inputs.items()})
    if stage=='lock':
        json_write('lock.json',registration)
        return
    if registration!=json_read('lock.json'):
        raise ValueError('Frozen dense schedule or initialization changed')
    lock = digest('lock.json')
    val_owners = sorted(plan['beta_validation_owners'])
    if len(val_owners)!=8 or len(set(val_owners))!=8:
        raise ValueError('Expected eight locked calibration edits')
    if stage=='scale':
        warm = merged('warm',lock,val_owners)
        p,n,cap = [statistics.median(x for v in warm.values() for x in v[key])
                    for key in ('positive','negative','p4')]
        if min(p,n)<=0 or not 0<p/n<float('inf'):
            raise ValueError('Invalid gradient scale')
        json_write('scale.json',dict(lock=lock,scale=p/n,reference=cap))
        return
    if stage=='select':
        grid = merged('grid',lock,val_owners)
        warm = json_read('scale.json')
        assert warm['lock']==lock
        candidates = [
            dict(beta=warm['scale']*m, multiplier=m,
                 medians={a:statistics.median(x for v in grid.values() for x in v[str(m)][a]) for a in KU_ARMS})
            for m in MULTIPLIERS]
        selection = calibration_selection(candidates,warm['reference'])
        json_write('selection.json',dict(selection,lock=lock,reference=warm['reference'],
                                         candidates=candidates,candidate_steps=160))
        return

    lane = int(sys.argv[2])
    if lane not in (0,1):
        raise ValueError('Invalid device shard')
    runtime = FrozenRuntime(Path.cwd(),json_read(f'config{lane}.json'))
    runtime.load_prefixes(layout['prefixes'],plan['rows'])
    events = {r['id']:r for r in plan['rows']}
    if stage in ('warm','grid'):
        calibrate(runtime,stage,val_owners[lane::2],inputs,events,schedules,refs,lock,lane)
    elif stage=='smoke':
        o = val_owners[lane]
        params = inputs[o]
        runtime.seed(runtime.by_order[o]['seed'])
        editor = runtime.expert(o,dense=True)
        with torch.no_grad():
            editor.A.copy_(params['ku'])
            editor.G1.zero_()
        batch = runtime.teacher_batch(events[schedules[o][0]['native']],'reference')
        with torch.no_grad():
            logits = runtime.forward(batch)
            with mounted(runtime,editor) as hook:
                hook.set_teacher_routing(batch['labels'])
                assert torch.equal(runtime.forward(batch),logits)
        norm = float(input_scale(editor.A.detach(),solutions[o]['fits'].to(runtime.device)))
        assert abs(norm-params['fit_target'])<=2e-6*params['fit_target']
        assert sum(p.numel() for p in editor.parameters())==59904
        with torch.no_grad():
            editor.G1.fill_(1e-3)
        with mounted(runtime,editor) as hook:
            hook.set_teacher_routing(batch['labels'])
            gradients = torch.autograd.grad(positive(runtime,batch,True), (editor.A,editor.G1))
        if any(not torch.isfinite(g).all() or not g.norm()>0 for g in gradients):
            raise ValueError('Disconnected dense or TT writer gradient')
        json_write(f'smoke{lane}.json',dict(lock=lock,forward_count=runtime.forwards,
                                         fit_amplitude=norm,parameters=59904,
                                         gradient_norms=[float(g.norm()) for g in gradients]))
    elif stage=='train':
        visual_schedules = {o:plan['schedules'][str(o)] for o in solutions}
        for o,steps in visual_schedules.items():
            if len(steps)!=160:
                raise ValueError('Invalid visual-negative schedule length')
            for mixed,base in zip(steps,schedules[o],strict=True):
                visual_roles = sorted(events[i]['role'] for i in mixed['negatives'])
                if (mixed['native']!=base['native'] or mixed['views']!=base['views']
                        or visual_roles!=['train_ind','train_vis']
                        or mixed['negatives'][0]!=base['negatives'][0]
                        or any(events[i]['owner']!=o for i in mixed['negatives'])):
                    raise ValueError('Visual control schedule changed')
        selection = json_read('selection.json')
        if selection['lock']!=lock or not selection['ku_arms']:
            raise ValueError('No admitted dense route')
        bank = {}
        folder = Path(f'parts{lane}')
        folder.mkdir(exist_ok=True)
        for o in sorted(solutions)[lane::2]:
            params = inputs[o]
            bank[o] = {}
            traces = {}
            for arm in selection['training_arms']:
                dense = arm != 'V-SGD-TT'
                if dense:
                    start = dict(A=params['med' if arm in CONTROLS else 'ku'])
                else:
                    start = dict(G3=refs[o]['MED']['G3']*params['med_scale'],G4=refs[o]['MED']['G4'])
                runtime.seed(runtime.by_order[o]['seed'])
                schedule = visual_schedules[o] if arm=='V-SGD-dense-vis' else schedules[o]
                from .scope_solver import input_matrix
                initial_read = start['A'] if dense else input_matrix(start['G3'],start['G4'][:,:,0])
                bank[o][arm],traces[arm] = train(runtime,runtime.expert(o,dense=dense),
                    start,events,schedule,selection['beta'],arm!='V-U',
                    read_lr=params['lr'] if dense else 1e-3,
                    angle_basis=params['ku_basis'] if arm!='V-U' else None)
                traces[arm][0]['ku_angles_deg_before_training'] = principal_angles(initial_read,params['ku_basis'])
                print('TRAINED',o,arm,flush=True)
            write_state(folder/f'{o}.pt',bank[o])
            json_write(folder/f'{o}.json',traces)
        write_state(f'bank{lane}.pt',bank)
    else:
        raise ValueError('Unknown dense stage')


if __name__=='__main__':
    run()
