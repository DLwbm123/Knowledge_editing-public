"""Dependency-ordered CLI for the scope-subspace experiment."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

import torch

from .evaluate import d0, fit_scale, routing_report, s0_report
from .features import DEV, POSITIVES, TEST, VISUAL_NEGATIVE, load_manifest, acquire, read_features
from .router import ScopeRouter
from .runtime import FrozenRuntime
from .solve_panel import solve
from .train_value import gradient_ratio, mounted, shifted_scores, train, validate_kl


def json_read(path):
    return json.loads(Path(path).read_text())


def json_write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,allow_nan=False,indent=2)


def write_state(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value,stream)


def load_state(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def frozen_core(path):
    if not path:
        return {}
    result = json_read(path)
    if 'decisions' in result:
        result = result['decisions']
    return {k:v['core_correct'] if isinstance(v,dict) else v for k,v in result.items()}


def bound_solution(path, manifest):
    obj = load_state(path)
    if obj['manifest_digest'] != sha256(Path(manifest).read_bytes()).hexdigest():
        raise ValueError('Manifest differs from solve lock')
    return obj['solutions']


def get_schedule(plan, owner):
    schedule = plan['schedules'][str(owner)]
    if len(schedule) != 160:
        raise ValueError('Expected 160 locked writer steps')
    return schedule


def writer_initialization(solutions, owner):
    result = solutions[owner]
    tt = result['K-TT']
    fits = result['fits'].to(tt['A'])
    measured = torch.quantile((fits @ tt['A'].T).norm(dim=-1), .5)
    reference = torch.quantile((fits @ result['K-P4']['A'].to(fits).T).norm(dim=-1), .5)
    if min(float(measured),float(reference))<=0:
        raise ValueError('Missing FIT read-in scale')
    ratio = float(reference / measured)
    scale = ratio if ratio>2 or ratio<.5 else 1.
    return dict(tt,G3=tt['G3']*scale,fit_readin_ratio=ratio,writer_input_scale=scale)


def reference_arm_states(runtime, owners):
    arms = {}
    for owner in owners:
        arms[owner] = {name:runtime.checkpoint(name,owner) for name in ('MED','MDU','P4')}
    return arms


def scoped_payload(plan, features, solutions):
    by_id = {r['id']:r for r in plan['rows']}
    for event,val in features.items():
        expected = by_id[event]
        if val['owner'] != expected.get('owner') or val['role'] != expected['role'] or val['image'] != expected['source_sha256']:
            raise ValueError('Feature manifest binding differs')
    scores = {}
    basis = {o:{k:fit_scale(v['A'],s['fits']) for k,v in s.items() if k.startswith('K-')}
             for o,s in solutions.items()}
    for event, row in features.items():
        if row['role'] not in {'hold_pos','hold_ind','hold_vis','official_ind'}:
            continue
        x = row['feature'][0]
        if row['owner'] not in basis:
            raise ValueError('Held-out example lacks a fitted expert')
        scores[event] = {}
        for k,(a,scale) in basis[row['owner']].items():
            y = x.to(a) @ a.T
            scores[event][k] = dict(role=row['role'],image=row['image'],score=float(y.norm()/scale))
    return scores


def measure(runtime,row,expert):
    result = {}
    target_path = row.get('transform') in VISUAL_NEGATIVE
    if row.get('answer'):
        with torch.no_grad():
            batch = runtime.teacher_batch(row,'reference')
            if expert is None:
                scores,tokens = shifted_scores(runtime,batch)
            else:
                with mounted(runtime,expert) as hook:
                    hook.set_teacher_routing(batch['labels'])
                    scores,tokens = shifted_scores(runtime,batch)
            lp = scores.log_softmax(-1)
            nll = -lp.gather(1,tokens[:,None]).squeeze(1)[tokens!=runtime.eos]
            if len(nll):
                if target_path:
                    result['target_path_nll'] = float(nll.mean())
                else:
                    result.update(first_nll=float(nll[0]),answer_nll=float(nll.mean()))
    if row['role'] in {'hold_ind','hold_vis','t2l','ood'}:
        for prefix in ('reference','base'):
            if prefix=='reference' and not row.get('answer'):
                continue
            metric = 'target_path_kl' if prefix=='reference' and target_path else prefix+'_kl'
            result[metric] = validate_kl(runtime,row,expert,prefix) if expert is not None else 0.
    return result


def eval_workers(args, runtime, plan, solutions):
    rows = [r for r in plan['rows'] if r['role'] in set(args.roles)]
    if not rows:
        raise ValueError('No evaluation rows')
    features = read_features(args.features) if args.features else {}
    expected = {r['id']:r for r in plan['rows']}
    for k,v in features.items():
        row = expected[k]
        if v['role'] != row['role'] or v['image'] != row['source_sha256'] or v['owner'] != row.get('owner'):
            raise ValueError('Routing feature roles changed')
    if args.policy == 'bank' and (not args.bank or not features or not solutions):
        raise ValueError('Bank needs paired features, trained states and solution FIT bindings')
    if args.policy == 'stage1b' and any(r['role']!='official_ind' for r in rows):
        raise ValueError('Stage 1b is the frozen official indirect family only')
    rows = rows[args.shard::args.shards]
    routes = runtime.frozen_routes(rows) if args.policy != 'writer' and not args.owner_only else {}
    bank = {}
    for path in args.bank or []:
        shard = load_state(path)
        if set(bank) & set(shard):
            raise ValueError('Overlapping trained expert shards')
        bank.update(shard)
    if args.policy=='bank':
        bank = {o:state[args.bank_arm] for o,state in bank.items()}
        if len(bank)!=150 and not args.bank_pilot:
            raise ValueError('Full S2 requires all 150 experts. Mark a smaller bank as a pilot.')
    if args.policy == 'bank' and set(bank) != set(solutions):
        raise ValueError('Trained bank and feature owners must match. No partial mixed-bank substitution.')
    ans_by_owner = {r['owner']:r['answer'] for r in plan['rows'] if r['role']=='native'}
    decisions = frozen_core(args.decisions)
    routers = {}
    result = []
    if args.policy == 'bank':
        g3g4 = {o:runtime.expert(o,ck).factors()[1].detach().cpu() for o,ck in bank.items()}
        fit = {o:s['fits'] for o,s in solutions.items()}
        proto = {o:s['prototypes'] for o,s in solutions.items()}
        cal = defaultdict(list)
        for v in features.values():
            if v['role'] in {'cal_ind','cal_vis'}:
                cal[v['image']].append(v['feature'][0])
        for name in ('coupled','prototype'):
            for alpha in (.05,.10):
                router = ScopeRouter(g3g4,fit,prototypes=proto if name=='prototype' else None)
                router.calibrate([torch.stack(v) for v in cal.values()],alpha)
                routers[f'{name}_{alpha:.2f}'] = router
    p_owners = set(plan.get('stage1_owners',[]))
    if args.policy=='stage1b' and not p_owners:
        raise ValueError('Declare the frozen Stage-1 checkpoint owners')
    if args.owner_only and args.policy!='stage1b':
        raise ValueError('Owner-only routing is a Stage-1b diagnostic')
    fallback = 0
    for number,row in enumerate(rows,1):
        rid = row['id']
        if own_seed := row.get('owner'):
            runtime.seed(runtime.by_order[own_seed]['seed'])
        own = row.get('owner')
        if args.policy == 'stage1b':
            selected = own if args.owner_only else routes[rid]
            if not args.owner_only and 'frozen_r0_owner' in row and selected != row['frozen_r0_owner']:
                raise ValueError('R0 differs from the frozen Stage-1b routing ledger')
            fallback += int(selected is not None and selected not in p_owners)
            methods = [(arm,selected,runtime.checkpoint(arm if selected in p_owners else 'MDU',selected)
                        if selected is not None else None) for arm in args.stage1b_arms]
        elif args.policy == 'writer':
            methods = [(arm,own,bank[own][arm]) for arm in args.writer_arms]
            methods.append(('P4',own,runtime.checkpoint('P4',own)))
        else:
            selected = routes[rid]
            methods = [('R0',selected,runtime.checkpoint('MDU',selected) if selected is not None else None),
                       ('decoupled_R0',selected,bank.get(selected))]
            for name,router in routers.items():
                owner,score = router.route(features[rid]['feature'][0])[0]
                methods.append((name,owner,bank.get(owner)))
        # Reuse identical outputs across routing controls for the same query.
        generated = {}
        for method, selected, ck in methods:
            if ck is None:
                selected = None
            key = (selected,runtime.state_digest(ck)) if ck is not None else (None,None)
            if key not in generated:
                expert = runtime.expert(selected,ck) if ck is not None else None
                generated[key] = (runtime.generate(row,expert),measure(runtime,row,expert))
            output, measurement = generated[key]
            from experiments.med_mdu_v2_admission_v1.summarize_reference import norm
            target = ans_by_owner.get(selected,ans_by_owner.get(own))
            normalized = norm(output['answer'])
            from experiments.med_mdu_v2_admission_v1.reference_scores import PROTOCOL, digest
            opaque = digest([PROTOCOL,row['question'],row.get('answer'),output['answer']])
            exact = normalized == norm(target) if target is not None else None
            result.append(dict(id=rid,role=row['role'],image=row['source_sha256'],owner=own,
                              route=method,selected=selected,core=decisions.get(opaque),exact=exact,
                              raw=output,metric=measurement,score_id=opaque,family=row.get('family'),question=row['question'],reference=None if row.get('transform') in VISUAL_NEGATIVE else row.get('answer')))
        if number==1 or number%20==0:
            print('EVAL',number,flush=True)
    thresholds = {k:(r.threshold if math.isfinite(r.threshold) else None) for k,r in routers.items()}
    return dict(rows=result,thresholds=thresholds,calibration_images=len(cal) if routers else 0,
                stage1b_mdu_fallback=fallback,owner_only=args.owner_only,
                forwards=runtime.forwards,generations=runtime.generations)


def run(args):
    if args.command=='d0':
        with Path(args.diagnostics).open() as stream:
            rows = [json.loads(s) for s in stream]
        image_groups = json_read(args.images)
        report = d0(rows,image_groups)
        if args.reference_audit:
            valid = {r['q'] for r in json_read(args.reference_audit)['labels']
                     if r['type']=='negation_uncertainty' and r['judge']=='V'}
            subset = [r for r in rows if r.get('variant')!='negation_uncertainty' or r['query'] in valid]
            report['reference_valid_sensitivity'] = d0(subset,image_groups)
        json_write(args.output,report)
        return
    if args.command=='score-input':
        rows = [row for p in args.inputs for row in json_read(p)['rows'] if row.get('reference')]
        from experiments.med_mdu_v2_admission_v1.reference_scores import PROTOCOL, digest
        records = {}
        consumers = []
        for row in rows:
            key = digest([PROTOCOL,row['question'],row['reference'],row['raw']['answer']])
            if key!=row['score_id']:
                raise ValueError('Score input differs from evaluation receipt')
            records[key] = dict(opaque_query_id=key,question=row['question'],gold_answer=row['reference'],
                                raw_base_answer=row['raw']['answer'])
            consumers.append({k:row[k] for k in ('id','route','score_id')})
        if not records:
            raise ValueError('No scoreable reference rows')
        out = Path(args.output)
        out.mkdir()
        keys = sorted(records)
        for index,start in enumerate(range(0,len(keys),1548)):
            part = keys[start:start+1548]
            folder = out/str(index)
            folder.mkdir()
            json_write(folder/'SUPPLEMENT_INPUT.private.json',dict(status='SUPPLEMENT_IDENTITIES_FROZEN',
                payloads={k:records[k] for k in part},categories=dict(history_unverified=part,
                    conflict_unknown=[],known_unknown=[])))
        json_write(out/'consumers.private.json',consumers)
        return
    if args.command=='summarize':
        shards = [json_read(p) for p in args.inputs]
        rows = [row for shard in shards for row in shard['rows']]
        thresholds = shards[0]['thresholds']
        if any(s['thresholds']!=thresholds or s['calibration_images']!=shards[0]['calibration_images'] for s in shards):
            raise ValueError('Routing threshold differs across workers')
        if len({(r['id'],r['route']) for r in rows}) != len(rows):
            raise ValueError('Overlapping evaluation shards')
        decisions = {}
        for file in args.decisions or []:
            source = frozen_core(file)
            if any(decisions[k]!=source[k] for k in set(decisions)&set(source)):
                raise ValueError('Conflicting scores for the same frozen output')
            decisions.update(source)
        if decisions:
            for row in rows:
                row['core'] = decisions.get(row['score_id'])
        report = routing_report(rows)
        report.update(thresholds=thresholds,calibration_images=shards[0]['calibration_images'],
                      stage1b_mdu_fallback=sum(s.get('stage1b_mdu_fallback',0) for s in shards))
        json_write(args.output,report)
        return
    plan = load_manifest(args.manifest)
    if args.command=='solve':
        data = read_features(args.features)
        expected = {r['id']:r for r in plan['rows']}
        for k,v in data.items():
            row = expected[k]
            if row['role'] != v['role'] or row.get('owner') != v['owner'] or row['source_sha256'] != v['image']:
                raise ValueError('Development feature roles changed')
        refs = {}
        for p in args.references:
            entries = load_state(p)
            if set(refs)&set(entries):
                raise ValueError('Overlapping reference weight shards')
            refs.update(entries)
        locked = plan.get('projection_parity',{})
        if set(locked)!={k for k,v in data.items() if v.get('projection_parity')} or len(locked)!=5:
            raise ValueError('Five actual frozen-MED projection checks required before S0')
        if not any(v.get('zero_residual_parity') for v in data.values()):
            raise ValueError('Missing actual zero-residual parity')
        solutions = solve(data,refs,args.device)
        write_state(args.output,dict(manifest_digest=sha256(Path(args.manifest).read_bytes()).hexdigest(),
                                     solutions=solutions))
        return
    solutions = bound_solution(args.solution,args.manifest) if getattr(args,'solution',None) else {}
    if args.command=='s0':
        scored = scoped_payload(plan,read_features(args.features),solutions)
        owners = {r['id']:r.get('owner') for r in plan['rows']}
        calibration = set(plan.get('calibration_owners',[]))
        main = {k:v for k,v in scored.items()
                if v['K-TT']['role']!='official_ind' and owners[k] not in calibration}
        report = s0_report(main)
        report['calibration_owners_excluded'] = len(calibration)
        report['official_ind'] = {k:statistics.median(e[k]['score'] for e in scored.values()
            if e[k]['role']=='official_ind') for k in next(iter(main.values()))} if any(
                e['K-TT']['role']=='official_ind' for e in scored.values()) else {}
        report['solver'] = {str(o):{k:{m:v for m,v in s.items() if not isinstance(v,torch.Tensor)}
            for k,s in rec.items() if k.startswith('K-')} for o,rec in solutions.items()}
        report['solution_digest'] = sha256(Path(args.solution).read_bytes()).hexdigest()
        json_write(args.output,report)
        return
    if args.command in {'beta','train'}:
        h = sha256(Path(args.solution).read_bytes()).hexdigest()
        gate = json_read(args.gate)
        if gate['decision']!='S1_ADMITTED' or gate.get('solution_digest')!=h:
            raise ValueError('S0 gate does not admit this solution')
    runtime = FrozenRuntime(Path(args.output).parent,json_read(args.contract))
    runtime.load_prefixes(args.base_prefixes or [], plan['rows'])
    runtime.allow_base_generation = args.command in {'collect','prefix'}
    if args.command=='prefix':
        if args.part=='evaluation' and not solutions:
            raise ValueError('Lock S0 before the evaluation prefix preflight')
        allowed = DEV if args.part=='development' else TEST
        rows = [r for r in plan['rows'] if r['role'] in allowed and r['role'] not in POSITIVES
                and r.get('base_tokens') is None and (not args.roles or r['role'] in args.roles)]
        rejected = []
        for row in rows[args.shard::args.shards]:
            result = runtime.base_answer(row)
            if result['at_cap'] or not result['tokens'] or all(t==runtime.eos for t in result['tokens']):
                rejected.append(dict(id=row['id'],role=row['role'],at_cap=result['at_cap']))
        output = runtime.generated_prefixes()
        output.update(rejected=rejected,requested=len(rows[args.shard::args.shards]))
        json_write(args.output,output)
    elif args.command=='collect':
        if args.part=='evaluation' and not solutions:
            raise ValueError('Lock S0 solutions before collecting held-out features')
        cache = acquire(runtime,plan,args.part,args.shard,args.shards,args.roles)
        if args.references:
            owners = {v['owner'] for v in cache.values() if v['role']=='native'}
            write_state(args.references,reference_arm_states(runtime,owners))
        write_state(args.output,cache)
    elif args.command=='beta':
        owners = plan['beta_validation_owners']
        if not owners:
            raise ValueError('Need a locked beta calibration cohort')
        events = {r['id']:r for r in plan['rows']}
        validations = {o:[r for r in events.values() if r.get('owner')==o and r['role'] in {'val_pos','val_neg'}] for o in owners}
        positive_grad,negative_grad = [],[]
        if len(set(owners)) != len(owners) or any(not {'val_pos','val_neg'}<={r['role'] for r in validations[o]} for o in owners):
            raise ValueError('Incomplete beta validation support')
        for owner in owners:
            runtime.seed(runtime.by_order[owner]['seed'])
            warm,_ = train(runtime,runtime.expert(owner),writer_initialization(solutions,owner),events,
                           get_schedule(plan,owner)[:8],0.,False)
            p,n = gradient_ratio(runtime,runtime.expert(owner,warm),validations[owner])
            positive_grad.extend(p)
            negative_grad.extend(n)
        pg,ng = statistics.median(positive_grad),statistics.median(negative_grad)
        if not math.isfinite(pg+ng) or pg<=0 or ng<=0:
            raise ValueError('KL/CE gradient scale unidentified after warmup')
        cap = statistics.median(validate_kl(runtime,r,runtime.expert(o,runtime.checkpoint('P4',o)))
                 for o in owners for r in validations[o] if r['role']=='val_neg')
        candidates = {}
        selected = None
        for multiplier in (.3,1.,3.):
            beta = multiplier*pg/ng
            kl = {}
            for joint in (False,True):
                observations = []
                for o in owners:
                    runtime.seed(runtime.by_order[o]['seed'])
                    weights,_ = train(runtime,runtime.expert(o),writer_initialization(solutions,o),events,
                                      get_schedule(plan,o),beta,joint)
                    editor = runtime.expert(o,weights)
                    observations.extend(validate_kl(runtime,r,editor) for r in validations[o] if r['role']=='val_neg')
                kl['V-TT-ft' if joint else 'V-TT'] = statistics.median(observations)
            candidates[str(multiplier)] = dict(beta=beta,medians=kl)
            if selected is None and max(kl.values())<=cap:
                selected = beta
        json_write(args.output,dict(beta=selected,reference=cap,candidates=candidates,ce_gradient=pg,kl_gradient=ng,
                                    manifest_digest=sha256(Path(args.manifest).read_bytes()).hexdigest(),
                                    solution_digest=h,candidate_steps=160,warmup_steps=8))
    elif args.command=='train':
        beta = json_read(args.beta)
        if (beta['beta'] is None or beta.get('solution_digest')!=h
                or beta['manifest_digest']!=sha256(Path(args.manifest).read_bytes()).hexdigest()):
            raise ValueError('Missing passing beta bound to this manifest')
        by_id = {r['id']:r for r in plan['rows']}
        bank = {}
        curves = {}
        for owner in sorted(solutions)[args.shard::args.shards]:
            runtime.seed(runtime.by_order[owner]['seed'])
            steps = get_schedule(plan,owner)
            if any(by_id[e]['owner']!=owner for step in steps for e in step['views']+step['negatives']+[step['native']]):
                raise ValueError('Schedule crosses expert ownership')
            state = writer_initialization(solutions,owner)
            bank[owner],curves[owner] = {},{}
            for arm,joint in (('V-TT',False),('V-TT-ft',True)):
                bank[owner][arm],curves[owner][arm] = train(runtime,runtime.expert(owner),
                    state,by_id,steps,beta['beta'],joint)
            curves[owner]['fit_readin_ratio'] = state['fit_readin_ratio']
            curves[owner]['writer_input_scale'] = state['writer_input_scale']
            snapshots = Path(args.output).with_suffix('.owners')
            snapshots.mkdir(exist_ok=True)
            write_state(snapshots/f'{owner}.pt',bank[owner])
            json_write(snapshots/f'{owner}.json',curves[owner])
            print('TRAINED',owner,flush=True)
        write_state(args.output,bank)
        json_write(Path(args.output).with_suffix('.curves.json'),curves)
    elif args.command=='eval':
        json_write(args.output,eval_workers(args,runtime,plan,solutions))
    if args.command=='collect' and runtime.generated_base:
        json_write(Path(args.output).with_suffix('.base.json'),runtime.generated_prefixes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    for name in ('d0','prefix','collect','solve','s0','beta','train','eval','score-input','summarize'):
        p = commands.add_parser(name)
        p.add_argument('--output',required=True)
        if name in {'prefix','collect','solve','s0','beta','train','eval'}:
            p.add_argument('--manifest',required=True)
        if name in {'prefix','collect','beta','train','eval'}:
            p.add_argument('--contract',required=True)
            p.add_argument('--shard',type=int,default=0)
            p.add_argument('--shards',type=int,default=1)
            p.add_argument('--base-prefixes',nargs='+')
        if name in {'prefix','collect','s0','beta','train','eval'}:
            p.add_argument('--solution',required=name in {'s0','beta','train'})
        if name in {'solve','s0','eval'}:
            p.add_argument('--features',nargs='+',required=name!='eval')
        if name in {'beta','train'}:
            p.add_argument('--gate',required=True)
        if name=='train':
            p.add_argument('--beta',required=True)
        if name=='d0':
            p.add_argument('--diagnostics',required=True)
            p.add_argument('--images',required=True)
            p.add_argument('--reference-audit')
        if name in {'collect','prefix'}:
            p.add_argument('--part',choices=('development','evaluation'),required=True)
        if name=='collect':
            p.add_argument('--references')
        if name in {'prefix','collect'}:
            p.add_argument('--roles',nargs='+')
        if name=='solve':
            p.add_argument('--references',nargs='+',required=True)
            p.add_argument('--device',default='cpu')
        if name=='eval':
            p.add_argument('--policy',choices=('stage1b','writer','bank'),required=True)
            p.add_argument('--roles',nargs='+',required=True)
            p.add_argument('--bank',nargs='+')
            p.add_argument('--bank-arm',choices=('V-TT','V-TT-ft'),default='V-TT')
            p.add_argument('--writer-arms',nargs='+',
                           choices=('V-TT','V-TT-ft','V-U','V-U-ft','V-SGD-dense','V-SGD-TT'),
                           default=('V-TT','V-TT-ft'))
            p.add_argument('--bank-pilot',action='store_true')
            p.add_argument('--stage1b-arms',nargs='+',choices=('P1','P2','P3','P4','MDU'),
                           default=('P1','P2','P3','P4','MDU'))
            p.add_argument('--owner-only',action='store_true')
            p.add_argument('--decisions')
        if name in {'summarize','score-input'}:
            p.add_argument('--inputs',nargs='+',required=True)
        if name=='summarize':
            p.add_argument('--decisions',nargs='+')
    args = parser.parse_args()
    if hasattr(args,'shards') and not 0<=args.shard<args.shards:
        parser.error('Invalid sharding contract')
    run(args)


if __name__=='__main__':
    main()
