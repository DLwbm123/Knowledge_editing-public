"""Two-anchor finite-response vectors and one analytic constrained direction."""
import math
import torch
from .editor import finite
from .bdecompose_ops import StopRound


def calibrate_vectors(gradients,anchors,responses,a,b,save):
    fv=[q.double()*b.double()[:,None]*a.double()[None,:] for q in anchors]
    G=torch.tensor([[float((anchors[i].double()*fv[j]).sum()) for j in range(2)] for i in range(2)],dtype=torch.float64)
    diagonal=G.diag()
    state=dict(Gram=G.tolist(),condition_max=10000.,responses=responses)
    save(state)
    if not torch.isfinite(G).all() or not (diagonal>0).all():raise StopRound('STOPPED_CALIBRATION_GRAM_INVALID')
    norm=diagonal.sqrt();normalized=G/norm[:,None]/norm[None,:]
    eigen=torch.linalg.eigvalsh(normalized);condition=float(eigen[-1]/eigen[0]) if eigen[0]>0 else float('inf')
    state.update(normalized_Gram=normalized.tolist(),eigenvalues=eigen.tolist(),condition=condition if math.isfinite(condition) else None,condition_infinite=not math.isfinite(condition));save(state)
    if not math.isfinite(condition) or condition>10000. or eigen[0]<=0:raise StopRound('STOPPED_CALIBRATION_ILL_CONDITIONED')
    fitted=[];state['vectors']=[]
    for i,g in enumerate(gradients):
        before=torch.tensor([float((g.double()*q.double()).sum()) for q in anchors],dtype=torch.float64)
        y=torch.tensor(responses[i],dtype=torch.float64);coef=torch.linalg.solve(G,y-before)
        bar=finite((g.double()+float(coef[0])*fv[0]+float(coef[1])*fv[1]).float())
        actual=[float((bar.double()*q.double()).sum()) for q in anchors];error=[abs(x-z) for x,z in zip(actual,responses[i])]
        state['vectors'].append(dict(quantity=['NLL','margin','null'][i],before=before.tolist(),observed=responses[i],coefficients=coef.tolist(),fitted=actual,absolute_errors=error));save(state)
        if any(not math.isfinite(x) or x>1e-5 for x in error):raise StopRound('STOPPED_CALIBRATION_FIT_FAILED')
        fitted.append(bar)
    state['status']='PASS';save(state);return fitted


def solve_geometry(En,Em,cross,cost,floor,margin_reference):
    data=dict(NLL_energy=En,margin_energy=Em,cross=cross,cost=cost,NLL_floor=floor,required_margin_gain=margin_reference)
    if not all(math.isfinite(x) and x>0 for x in (En,Em,cost,floor)) or not math.isfinite(cross):data['status']='STOPPED_SOLVER_GEOMETRY_INVALID';return data
    minimum=floor*floor/(2*En);free_scale=math.sqrt(2*cost/Em);free_nll=cross*free_scale
    data.update(minimum_NLL_cost=minimum,free_linear_NLL_change=free_nll)
    if minimum>cost:data['status']='STOPPED_SOLVER_INFEASIBLE';return data
    if free_nll<=-floor:
        data.update(status='ELIGIBLE',constraint_active=False,free_scale=free_scale,predicted_NLL_change=free_nll,predicted_margin_gain=math.sqrt(2*cost*Em))
    else:
        ep=Em-cross*cross/En;data.update(constraint_active=True,perpendicular_energy=ep)
        if not math.isfinite(ep) or ep<=1e-6*Em or cost-minimum<=0:data['status']='STOPPED_SOLVER_DEGENERATE';return data
        data.update(status='ELIGIBLE',predicted_NLL_change=-floor,predicted_margin_gain=-(floor/En)*cross+math.sqrt(2*(cost-minimum)*ep),NLL_scale=-floor/En,perpendicular_scale=math.sqrt(2*(cost-minimum)/ep))
    if data['predicted_margin_gain']<=margin_reference+1e-5:data['status']='STOPPED_PREDICTED_MARGIN_NOT_ABOVE_A'
    return data


def materialize_solution(qn,qm,geometry):
    if geometry['status']!='ELIGIBLE':raise StopRound('STOPPED_NO_ELIGIBLE_ANALYTIC_S')
    if not geometry['constraint_active']:return finite(geometry['free_scale']*qm)
    perp=finite(qm-(geometry['cross']/geometry['NLL_energy'])*qn)
    return finite(geometry['NLL_scale']*qn+geometry['perpendicular_scale']*perp)
