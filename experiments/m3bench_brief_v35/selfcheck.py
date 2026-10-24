"""CPU checks for the global gate, prompt-only change, and exact-output rejection."""
import copy
import json
import tempfile
from pathlib import Path
from protocol import INSTRUCTION, brief_row, verify_original, gate_ready


def main():
    q = dict(question='Which finding?', reference='private target', image_path='private image', query_id='x')
    b = brief_row(q)
    assert b['question'] == q['question']+'\n'+INSTRUCTION
    assert all(b[k]==q[k] for k in q if k!='question') and 'private target' not in b['question']
    d = dict(binding=q, raw_answer='answer', raw_token_ids=[1,2], EOS=True, at_cap=False)
    verify_original(d, copy.deepcopy(d))
    for field, value in [('raw_token_ids',[2,1]),('raw_answer','changed'),('EOS',False)]:
        bad = dict(d, **{field:value})
        try: verify_original(bad,d)
        except AssertionError: pass
        else: raise AssertionError('Mismatch admitted: '+field)
    with tempfile.TemporaryDirectory() as td:
        r=Path(td)
        for part in ('0','1'): (r/part).mkdir()
        assert not gate_ready(r)
        (r/'0/ORIGINAL_PARITY.json').write_text(json.dumps(dict(status='PASS',outputs=30)))
        assert not gate_ready(r)
        (r/'1/ORIGINAL_PARITY.json').write_text(json.dumps(dict(status='PASS',outputs=32)))
        assert gate_ready(r)
        (r/'1/ORIGINAL_PARITY.json').write_text(json.dumps(dict(status='PASS',outputs=31)))
        try: gate_ready(r)
        except AssertionError: pass
        else: raise AssertionError('Incomplete global originals admitted')
    return dict(status='PASS', exact_output_mismatch_rejected=True, both_shards_required=True,
        prompt_only=True, answer_not_inserted=True, real_model_calls=0)


if __name__=='__main__': print(json.dumps(main()))
