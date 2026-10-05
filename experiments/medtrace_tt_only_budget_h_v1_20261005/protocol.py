"""Frozen registry: seeds are repetitions, not additional independent edits."""
STRUCTURES=('TT44','TT88','TT84','TT48')
CONDITIONS=('NO_H','H1')
BASE_ARMS=tuple(s+'_'+h for s in STRUCTURES for h in CONDITIONS)+('TT88_SMALL_LR_H','TT88_GUARDED_H')
ARMS=tuple(a+'_s'+str(k) for k in range(3) for a in BASE_ARMS)
PRIMARY=(('TT88_H1','TT44_H1'),('TT88_GUARDED_H','TT88_H1'))
PAIRS=PRIMARY+(('TT88_NO_H','TT44_NO_H'),('TT84_H1','TT44_H1'),('TT48_H1','TT44_H1'),('TT84_H1','TT48_H1'),('TT88_GUARDED_H','TT88_SMALL_LR_H'),('TT88_SMALL_LR_H','TT88_H1'))+tuple((s+'_H1',s+'_NO_H') for s in STRUCTURES)
def arms_for(structure,seedslot):
    return tuple(a+'_s'+str(seedslot) for a in BASE_ARMS if a.split('_')[0]==structure)
def decode(arm):
    base,slot=arm.rsplit('_s',1)
    return base.split('_')[0],base.split('_',1)[1],int(slot)
