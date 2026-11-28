"""Fit the lockable S0 candidates using only development features."""
from collections import defaultdict
from time import monotonic
import statistics

import torch

from .evaluate import fit_scale
from .features import DEV
from .scope_solver import BackgroundInverse, CovarianceMoment, Moment, fit, input_matrix, objective, tt_svd, unrestricted


def keyed(data):
    groups = defaultdict(lambda:defaultdict(list))
    for v in data.values():
        groups[v['owner']][v['role']].append(v['feature'])
    return groups


def solve(data, states, device, rounds=10, steps=50, *, owners=None, background_gram=None, save_owner=None, profile=None):
    if any(v['role'] not in DEV for v in data.values()):
        raise ValueError('Solve requires development-only features')
    groups = keyed(data)
    if not groups[None]['bg']:
        raise ValueError('Missing independent background')
    background = groups.pop(None)['bg']
    bg_cpu = Moment.from_queries(background) if background_gram is None else None
    if background_gram is None:
        bg = CovarianceMoment((bg_cpu.rows.T @ bg_cpu.rows).to(device))
    else:
        bg = CovarianceMoment(background_gram.to(device))
    selected_owners = sorted(groups) if owners is None else list(owners)
    if len(set(selected_owners))!=len(selected_owners) or not set(selected_owners)<=groups.keys():
        raise ValueError('Invalid owner shard')
    inverses = {}
    result = {}
    for owner in selected_owners:
        group = groups[owner]
        print('SOLVING',owner,flush=True)
        if len(group['fit']) != 4 or not group['val_pos'] or not group['val_neg']:
            raise ValueError(f'FIT or validation support missing for owner {owner}')
        p = Moment.from_queries(group['native']+group['fit']+group['train_eq'],device)
        n = [(1.,Moment.from_queries(group[role],device)) for role in ('train_ind','train_vis')]
        fits = torch.stack([v[0] for v in group['fit']])
        def score(a, rows):
            basis,den = fit_scale(a,fits)
            qs = torch.stack([v[0] for v in rows]).to(basis)
            return ((qs @ basis.T).norm(dim=-1)/den).tolist()
        def gap(a):
            return statistics.median(score(a,group['val_pos']))-statistics.median(score(a,group['val_neg']))
        initial = {}
        for arm in ('MED','MDU','P4'):
            state = states[owner][arm]
            initial[f'K-{arm}'] = dict(A=input_matrix(state['G3'].double(),state['G4'][:,:,0].double()))
        selected = {}
        candidates = defaultdict(list)
        for wv,gamma,keys in [(1.,g,('K-U','K-TT','K-TT-med','K-TT-p4')) for g in (.1,1.,10.)]+[
                              (0.,g,('K-TT-novis',)) for g in (.1,1.,10.)]+[(1.,0.,('K-TT-nobg',))]:
            ridge = .001*(1+wv+gamma)
            negatives = [(1.,n[0][1]),(wv,n[1][1])]
            tag = gamma,ridge
            if profile is not None:
                started = monotonic()
            if tag not in inverses:
                inverses[tag] = BackgroundInverse(bg_cpu,gamma,ridge,p.rows.size(1),background_gram)
            eigs,u = unrestricted(p,negatives,inverses[tag])
            if profile is not None:
                profile['unrestricted'] += monotonic() - started
            if 'K-U' in keys:
                candidates['K-U'].append(dict(A=u,eigenvalues=eigs.tolist(),val_gap=gap(u),
                    gamma=gamma,objective=float(eigs.sum())))
            for name in keys:
                if name=='K-U':
                    continue
                source = {'K-TT-med':'MED','K-TT-p4':'P4'}.get(name)
                init = states[owner][source]['G4'][:,:,0] if source else tt_svd(u)
                trained = fit(p,negatives+[(gamma,bg)],ridge,init,rounds=rounds,steps=steps,profile=profile)
                a = input_matrix(trained['G3'],trained['G4'][:,:,0])
                candidates[name].append(dict(trained,A=a,val_gap=gap(a),gamma=gamma,visual_weight=wv))
        for name, choices in candidates.items():
            selected[name] = max(choices,key=lambda x:x['val_gap'])
        anchor = selected['K-TT']
        weights = n+[(anchor['gamma'],bg)]
        for arm, rec in initial.items():
            rec.update(val_gap=gap(rec['A']),objective=float(objective(
                rec['A'].to(device),p,weights,.001*(2+anchor['gamma']))))
        value = dict(selected,**initial,fits=fits,prototypes=torch.stack(
            [v[0] for v in group['native']+group['fit']+group['train_eq']]))
        if save_owner is None:
            result[owner] = value
        else:
            save_owner(owner,value)
    return result
