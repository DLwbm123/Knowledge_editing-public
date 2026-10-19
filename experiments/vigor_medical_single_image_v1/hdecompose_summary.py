"""Offline summary for the frozen complete-H decomposition."""
from .bdecompose_summary import summarize as previous


def summarize(result,ended,life):
    value=previous(result,ended,life)
    value['restoration']['dtype_restored']=result.get('dtype_restored')
    diagnostic=result.get('diagnostic',{})
    value['increment']=diagnostic.get('increment')
    value['frozen_ratio']=diagnostic.get('frozen_ratio')
    value['historical_F_rounded_reference']=None
    value['limitations'][-1]='No new accepted edit, generation result, T0 performance or safe-H claim'
    parts=diagnostic.get('decomposition')
    if result['status']=='COMPLETE' and parts and parts['delta_F_rounded']<-2e-5 and parts['delta_native']>2e-5:
        value['interpretation']='LOCAL_EXECUTION_INTERVENTION_CRITICAL_FOR_REVERSAL'
    return value
