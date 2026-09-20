"""R1 dependency contract; runtime must supply actual bindings, never file paths alone."""
import hashlib
import json

REQUIRED = ('prepared_image', 'prompt_ids', 'attention_mask', 'context',
            'tokenizer_preprocessor', 'base_parameters_buffers', 'W0',
            'route_decision_and_gate', 'selected_parameter_version', 'hook',
            'implementation', 'backend', 'generation', 'batch_padding', 'RNG_condition')

def key(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def execution_key(bindings):
    if set(bindings)!=set(REQUIRED) or any(v is None for v in bindings.values()):
        raise ValueError('Incomplete execution dependency binding')
    return key(bindings)

def judge_key(*,image,question,context,reference,candidate,protocol,prompt,model,config,normalization,task):
    return key(locals())

def check_repeat(previous,current):
    if previous['K_exec']==current['K_exec'] and any(previous[k]!=current[k] for k in ('tokens','submitted_text','K_judge')):
        raise ValueError('EXECUTION_IDENTITY_VIOLATION')

def self_check():
    b={k:'fixed' for k in REQUIRED}
    baseline=execution_key(b)
    for name in REQUIRED:
        assert execution_key(dict(b,**{name:'changed'}))!=baseline
    try: execution_key({})
    except ValueError: pass
    else: raise AssertionError('Missing dependencies accepted')
    a=dict(K_exec=baseline,tokens=[1],submitted_text='answer',K_judge='scored')
    check_repeat(a,dict(a))
    for field,val in [('tokens',[2]),('submitted_text','other'),('K_judge','changed')]:
        try: check_repeat(a,dict(a,**{field:val}))
        except ValueError: pass
        else: raise AssertionError('Identity violation accepted')
    kwargs=dict(image='i',question='q',context='',reference='r',candidate='a',protocol='p',prompt='p',model='m',config='c',normalization='n',task='correctness')
    assert judge_key(**kwargs)!=judge_key(**dict(kwargs,task='teacher_consistency'))
    assert judge_key(**kwargs)!=judge_key(**dict(kwargs,reference='new version'))

if __name__=='__main__':
    self_check()
    print('Execution dependency and scoring invalidation contract: PASS (CPU only)')
