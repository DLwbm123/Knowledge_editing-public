"""Source-only CPU admission for M13 configuration and reused TT mathematics."""
import json
import torch
import m13 as run
import m13_report as report
import module_check as existing

def main():
    assert run.r.mm.flags('M13') == (True, False, True)
    assert report.ARMS == ('M13',)
    assert report.PAIRS == (('C','M13'),('M1','M13'),('M3','M13'),('E','M13'))
    report.selfcheck()
    existing.main()
    run.admission()
    prior = run.c.read(run.PARENT / 'public/RESULTS.json')
    assert all(prior['panels'][arm]['NATIVE']['missing'] == 0 for arm in ('C','M1','M3','E'))
    assert all(len(run.c.read(run.PARENT / 'private/results/C' / f'{i}.json')['curve']) == 160 for i in range(1,9))
    checks=run.c.read(run.RUN / 'public/CPU_ADMISSION.json')
    checks.update(M13_flags=[True,False,True],M2_disabled=True,reference_pairs=list(report.PAIRS),
                  retained_C_reused=True,reference_Judge_reused=True,VLM_forwards=0,Judge=0)
    run.c.write(run.RUN / 'public/CPU_ADMISSION.json',checks)
    print(json.dumps(checks))

if __name__ == '__main__':main()
