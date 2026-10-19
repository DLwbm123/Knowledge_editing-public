"""One bounded empirical quadratic response model, not a Hessian estimator."""
import math
import numpy as np
from scipy.optimize import minimize


def value(model,x):
    a,b=map(float,x);g=model['linear'];r=model['quadratic']
    return g[0]*a+g[1]*b+r[0]*a*a+r[1]*b*b+r[2]*a*b


def gradient(model,x):
    a,b=map(float,x);g=model['linear'];r=model['quadratic']
    return np.array([g[0]+2*r[0]*a+r[2]*b,g[1]+2*r[1]*b+r[2]*a])


def fit(linear,axes,mix):
    models=[];errors=[]
    for g,y,z in zip(linear,axes,mix):
        aa=y[0]-g[0];bb=y[1]-g[1]
        ab=4*(z-.5*(g[0]+g[1])-.25*(aa+bb))
        m=dict(linear=list(g),quadratic=[aa,bb,ab]);models.append(m)
        errors.extend([abs(value(m,[1,0])-y[0]),abs(value(m,[0,1])-y[1]),abs(value(m,[.5,.5])-z)])
    if len(models)!=3 or not all(math.isfinite(x) and x<=1e-5 for x in errors):
        raise ValueError('Quadratic finite-response fit failed')
    return dict(models=models,fit_errors=errors,max_fit_error=max(errors),status='PASS',interpretation='quantized_response_surrogate_not_pure_curvature')


def admissibility(models,x,bias,floor,margin,cost_gram,cost_max):
    x=np.asarray(x,dtype=float);G=np.asarray(cost_gram,dtype=float)
    predicted=[value(m,x) for m in models];cost=float(.5*x@G@x)
    return dict(coefficients=x.tolist(),predicted_NLL=predicted[0],predicted_margin=predicted[1],predicted_null=predicted[2],predicted_base_null=bias+predicted[2],continuous_F0_cost=cost,
        in_hull=bool(np.isfinite(x).all() and (x>=-1e-10).all() and x.sum()<=1+1e-10),NLL_pass=predicted[0]<=-floor+1e-5,margin_pass=predicted[1]>margin+1e-5,null_pass=abs(bias+predicted[2])<=.01,cost_pass=math.isfinite(cost) and cost>0 and cost<=cost_max*(1+1e-5))


def solve(fitted,bias,cost_gram,cost_max,floor,margin):
    models=fitted['models'];G=np.asarray(cost_gram,dtype=float)
    def inequalities(x):
        return np.array([-1.05*floor-value(models[0],x),.01-bias-value(models[2],x),.01+bias+value(models[2],x),1-x.sum(),1-float(.5*x@G@x)/cost_max])
    def jac(x):
        return np.stack([-gradient(models[0],x),-gradient(models[2],x),gradient(models[2],x),np.array([-1.,-1.]),-(G@x)/cost_max])
    # A single fixed initialization and invocation; CPU optimizer iterations
    # do not trigger model reads or form native candidate tensors.
    result=minimize(lambda x:-value(models[1],x),np.array([.25,.25]),jac=lambda x:-gradient(models[1],x),method='SLSQP',bounds=[(0.,1.),(0.,1.)],constraints=[dict(type='ineq',fun=inequalities,jac=jac)],options=dict(maxiter=200,ftol=1e-12,disp=False))
    data=dict(status='STOPPED_QUADRATIC_SOLVE_FAILED',success=bool(result.success),solver_message=str(result.message),iterations=int(result.nit),initial_coefficients=[.25,.25],solver_invocations=1,max_iterations=200,coefficients=result.x.tolist(),inequalities=inequalities(result.x).tolist())
    if not result.success or not np.isfinite(result.x).all():return data
    check=admissibility(models,result.x,bias,1.05*floor,margin,G,cost_max);data['recheck']=check
    if not all(check[k] for k in ('in_hull','NLL_pass','margin_pass','null_pass','cost_pass')):
        data['status']='STOPPED_QUADRATIC_PREDICTED_CONSTRAINT_FAILED';return data
    data['status']='ELIGIBLE';return data
