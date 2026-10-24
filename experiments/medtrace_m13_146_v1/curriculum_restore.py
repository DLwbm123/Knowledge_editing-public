"""Explicit one-time M13P recovery; never implicitly restart an existing owner."""
PHASES=(("native",140),("A2",80),("W0",320))

def remaining(completed):
    assert isinstance(completed,int) and 0<=completed<=540
    offset=0
    for phase,steps in PHASES:
        first=max(1,completed-offset+1)
        if first<=steps:yield phase,steps,first,first==1
        offset+=steps

def validate(saved):
    step=saved['completed_step'];assert 0<step<=540 and step%20==0
    assert len(saved['curve'])==step and saved['optimizer'] is None
    expected=[(phase,s) for phase,n in PHASES for s in range(1,n+1)][:step]
    assert [(x['phase'],x['phase_step']) for x in saved['curve']]==expected
    assert [x['step'] for x in saved['curve']]==list(range(1,step+1))
    assert saved['scalar_state']['step']==expected[-1][1]
    assert set(saved['state'])=={'G1','G2','G3','G4'}
    return step

def selfcheck():
    full=[(p,s) for p,n in PHASES for s in range(1,n+1)]
    for done in range(0,541,20):
        actual=[(p,s) for p,n,first,reset in remaining(done) for s in range(first,n+1)]
        assert actual==full[done:]
        for p,n,first,reset in remaining(done):assert reset==(first==1)
    assert list(remaining(140))[0]==('A2',80,1,True)
    assert list(remaining(160))[0]==('A2',80,21,False)
    assert list(remaining(220))[0]==('W0',320,1,True)
    print('PASS: all 28 checkpoint boundaries preserve remaining steps and phase resets')
if __name__=='__main__':selfcheck()
