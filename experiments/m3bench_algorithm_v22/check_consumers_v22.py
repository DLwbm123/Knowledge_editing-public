"""CPU-only scientific-accounting check, including uncertain losses and absent scores."""
import json,tempfile
from pathlib import Path
from consumers_v22 import ARMS,validate,blind,summarize,selfcheck

def main():
    selfcheck()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
        rows=[dict(query_id='q0',question='Q0',reference='A0',image_sha256='image0',roles=['T0']),dict(query_id='q1',question='Q1',reference='A1',image_sha256='image1',roles=['T1L'])]
        write(root/'DATA_ADMISSION.json',dict(tasks=[dict(key='E001',order=1,eval_rows=rows)]));write(root/'QUERY_ROWS.json',{r['query_id']:r for r in rows});write(root/'EVAL_BINDINGS.json',{});write(root/'ORIGINAL_BASE_MASK146.json',{})
        for row in rows:write(root/'base'/str(int(row['query_id'][-1])%5)/(row['query_id']+'.json'),dict(binding=row,raw_answer='base'+row['query_id'],at_cap=False))
        for arm in ARMS:
            folder=root/'0/E001'
            for row in rows:write(folder/'outputs'/arm/(row['query_id']+'.json'),dict(binding=row,raw_answer=arm+row['query_id'],at_cap=False))
            if arm in ARMS[:5]:write(folder/arm/'TRAIN_COMPLETE.json',dict(status='COMPLETE',accepted_steps=160))
            else:write(folder/(arm+'_TRAIN.json'),dict(steps=50))
        assert validate(root)['method_outputs']==14;blind(root)
        packet=json.loads((root/'BLIND_PACKET.json').read_text());verdict={}
        for x in packet['items']:
            answer=x['answer'];label='CORRECT'
            if answer=='baseq0':label='INCORRECT'
            elif answer=='RESERVEq1':label='UNCERTAIN'
            verdict[x['code']]=dict(label=label,reason='Fixed synthetic accounting case')
        write(root/'BLIND_VERDICTS.json',verdict);result=summarize(root)
        assert result['table']['RESERVE']['damage']==dict(Base_correct_denominator=1,lost_correct=1,explicit_incorrect=0)
        assert result['table']['SOURCE']['repair']['repaired']==1
        del verdict[next(iter(verdict))];write(root/'BLIND_VERDICTS.json',verdict)
        try:summarize(root)
        except AssertionError:pass
        else:raise AssertionError('Missing hidden score accepted')
    return dict(status='PASS',uncertain_loss_not_hidden=True,explicit_repair_denominator=True,missing_score_rejected=True,new_model_calls=0)
if __name__=='__main__':print(main())
