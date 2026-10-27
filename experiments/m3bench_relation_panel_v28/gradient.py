"""Source-grounded update/preserve pair; no inference parameters."""
import math

def combine(update, preserve, relational):
    assert len(update)==len(preserve) and len(update)>0
    assert all(math.isfinite(x) for x in [*update,*preserve])
    dot=sum(a*b for a,b in zip(update,preserve)); norm=sum(b*b for b in preserve)
    scale=dot/norm if relational and dot<0 and norm>0 else 0.
    return [(a-scale*b+b)/2 for a,b in zip(update,preserve)]

def selfcheck():
    assert combine([1.,2.],[0.,1.],True)==combine([1.,2.],[0.,1.],False)
    assert combine([1.,2.],[0.,0.],True)==[.5,1.]
    assert combine([-1.,0.],[1.,0.],True)==[.5,0.]
    assert combine([-1.,0.],[1.,0.],False)==[0.,0.]
    try:combine([float('nan')],[1.],True)
    except AssertionError:pass
    else:raise AssertionError('nonfinite accepted')
    return dict(status='PASS',nonconflict_identity=True,zero_preserve_identity=True,opposing_gradients=True,nonfinite_rejected=True,model_calls=0)
