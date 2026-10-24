"""One Base answer per frozen training-source row, without training or retries."""
import signal
import time
import route_common as c
v = c.v


def main():
    with (v.ROOT/'WORKER_ONCE').open('x'): pass
    v.allocations = dict(v.limits['counts'])
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original admission clock')))
    signal.alarm(max(1, int(v.limits['hard_end']-time.time()-300)))
    v.budget()
    v.setup()
    rows = v.read(v.ROOT/'SOURCE_ROWS.json')
    assert len(rows) == 80 and len({r['source_group'] for r in rows}) == 20
    for row in rows:
        dest = v.ROOT/'outputs'/f"{row['qid']}.json"
        attempt = v.ROOT/'attempts'/f"{row['qid']}.json"
        assert not dest.exists() and not attempt.exists(), 'No implicit retry'
        item = v.prepare({}, [row], False)[0]
        v.budget()
        assert v.counts['evaluation_generations'] < v.allocations['evaluation_generations']
        v.write(attempt, dict(qid=row['qid'], started=time.time()))
        answer = c.generate(None, item)
        tokens = list(answer.raw_token_ids)
        v.write(dest, dict(query=row, binding=item[4], raw_answer=answer.decoded_text,
            raw_token_ids=tokens, EOS=tokens[-1]==v.runtime.adapter.tokenizer.eos_token_id,
            at_cap=len(tokens)>=1024))
        v.progress(stage='base_qualification', completed=len(list((v.ROOT/'outputs').glob('*.json'))))
        del item
    assert v.counts['updates'] == v.counts['backwards'] == 0
    v.write(v.ROOT/'QUALIFICATION_COMPLETE.json', dict(outputs=80, counts=v.counts, time=time.time()))

if __name__ == '__main__':
    c.run(main)
