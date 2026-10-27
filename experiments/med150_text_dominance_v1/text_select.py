"""Fixed text-score weak dominance guard on a previously selected candidate."""
import math


def select(original, previous, text_original, text_previous):
    if original is None:
        assert previous is None
        return None
    assert previous is not None and math.isfinite(text_original) and math.isfinite(text_previous)
    return previous if text_previous>=text_original else original


def selfcheck():
    assert select(None,None,0.,0.) is None
    assert select(0,1,.9,.8)==0
    assert select(0,1,.8,.9)==1
    assert select(0,1,.8,.8)==1
    assert select(0,0,.8,.8)==0
    try:select(None,1,.1,.2)
    except AssertionError:pass
    else:raise AssertionError('OFF must remain OFF')
    return dict(status='PASS',zero_is_weak_order=True,off_preserved=True,no_role_or_answer=True)

if __name__=='__main__':print(selfcheck())
