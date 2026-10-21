"""All-target token risk; no hand-selected tokens or generated-answer supervision."""
import math
import torch
import torch.nn.functional as F
from .brc_pair import solve as constrained_solve
from .contrast_basis import build_bases
from .editor import finite


def token_statistics(logits, labels, temperature):
    if temperature<=0 or not (labels!=-100).any():
        raise ValueError('Positive temperature and nonempty target required')
    flat=logits.reshape(-1,logits.shape[-1]).float()
    target=labels.reshape(-1)
    mean=finite(F.cross_entropy(flat,target,ignore_index=-100,reduction='mean'))
    losses=F.cross_entropy(flat,target,ignore_index=-100,reduction='none')[target!=-100]
    risk=finite(temperature*(torch.logsumexp(losses/temperature,dim=0)-math.log(losses.numel())))
    return mean,risk


def source_measure(model,batch,arm,config):
    labels=batch.labels[:,1:]
    if not (labels!=-100).any():
        raise ValueError('Empty target before model call')
    logits=model(**batch.forward_kwargs()).logits[:,:-1]
    mean,risk=token_statistics(logits,labels,config['token_risk_temperature'])
    if arm not in ('MEAN4','RISK4'):
        raise ValueError('Unknown frozen token objective')
    return mean,mean if arm=='MEAN4' else risk


def shared_bases(*args):
    basis=build_bases(*args)['CONTRAST4']
    return dict(MEAN4=basis,RISK4=basis)


def solve(runtime,layer,pair,factor,config,admission,progress,*,budget):
    if config['version']!='PUREW_TOKEN_RISK_V1':
        raise PermissionError('Frozen all-token risk required')
    return constrained_solve(runtime,layer,pair,factor,config,admission,progress,budget=budget,
                            basis_builder=shared_bases,source_measure=source_measure)
