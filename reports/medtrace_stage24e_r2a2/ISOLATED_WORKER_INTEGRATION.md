# R2-A2 isolated worker integration

The isolated control flow is in `reports/medtrace_stage24e_r2a2/isolated_worker/worker_r2.py`. It is not imported by or deployed to the active worker.

Implemented calls: frozen slot validation and list grouping; PREPARED→WRITTEN→EVALUATED→RELEASED/STOPPED transitions; exact teacher lookup without fallback; active/inactive/masked U handling on initial training and maintenance; consumer ID plus execution state ID; native verdict barrier; immutable event/version checks; restart recovery that does not repeat a committed write.

CPU integration test result: **8/8 passed** using injected synthetic engine, optimizer spy and verdict map. Test output is retained in `private/cpu_integration_test.log`. No medical model, CUDA, remote worker, new Judge request, or formal score was used.

Known boundary: this proves the intended production call sequence under deterministic CPU injection only. It does not prove real-model numerical behavior or R2-B acceptance.
