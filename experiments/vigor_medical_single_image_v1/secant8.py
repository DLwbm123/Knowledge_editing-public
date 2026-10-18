"""One deterministic quantized-effect secant candidate at each frozen rejection."""
from functools import partial
from . import precision_abc
from .stress_precision8 import POINTS,prepare_control,AUTH as CONTROL_AUTH
from .amplitude_compensation import point

AUTH=dict(CONTROL_AUTH,variant='quantized_secant_one_shot',candidates_per_point=1)


def main(config):
    return precision_abc.main(config,evaluation=partial(point,compensated=True,mode='secant'),
        authorization=AUTH,prepare=prepare_control,label='SECANT8',points=POINTS,prefix_compensate=True)
