"""Final attribution only: seven corrected rejections; S04 has no correction."""
from functools import partial
from . import precision_abc
from .stress_precision8 import prepare_control

POINTS=(('S01',12),('S02',13),('S03',13),('S05',14),('S06',11),('S07',10),('S08',16))
AUTH=dict(points=[list(p) for p in POINTS],extra_steps=0,extra_corrections=0,
    tune=False,replace=False,generation=False,judge=False,variant='secant_ABC7')


def main(config):
    return precision_abc.main(config,evaluation=partial(precision_abc.evaluate,compensate='secant'),
        authorization=AUTH,prepare=partial(prepare_control,compensate='secant'),label='SECANT_ABC7',
        points=POINTS,prefix_compensate='secant')
