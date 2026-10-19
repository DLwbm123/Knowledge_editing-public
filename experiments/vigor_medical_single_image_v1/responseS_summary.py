"""Offline S accounting, retaining NA when calibration/analytic admission stops."""
from .hybridH_summary import summarize as previous


def summarize(result,ended,life):
    response=result.get('response',{});adapted=dict(result,hybrid=response)
    value=previous(adapted,ended,life)
    value['restoration']['S_endpoint_restored']=value['restoration'].pop('H_endpoint_restored')
    value.update(calibration=response.get('calibration'),anchor_checks=response.get('anchor_checks'),calibrated_geometry=response.get('calibrated_geometry'),analytic_recheck=response.get('analytic_recheck'),historical_endpoint_forward_calls=response.get('historical_endpoint_forward_calls',0))
    value['budget_checks']['historical_endpoint_forward_calls']=value['historical_endpoint_forward_calls']==0
    return value
