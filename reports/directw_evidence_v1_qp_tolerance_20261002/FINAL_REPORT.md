# E15 final: numerical progress, no accepted direct-answer edits

All8cases completed and rolled back. Accepted edits0/8, exact target sequences0/8; all Base/native control checks passed, no failure receipt, matrix, active worker or lease remains. The unchanged mechanical regression passed. Both declared criteria FAILED: numerical criterion required all8 to avoid SOLVER_NOT_CONVERGED (7/8 did), and efficacy required at least2 exact target outputs after accepted physical edits plus protection checks (0 did).

Six cases ended NOT_SATISFIED after three accepted intermediate steps. Case5 still ended SOLVER_NOT_CONVERGED; case7 ended BACKTRACK_REJECTED after one accepted step. Case5's recorded CG solves both converged (relative residuals6.696e-5 and5.803e-5 within1e-4), but primal KKT residual1.2397766e-5 exceeded the explicit1e-5 absolute component. This is a retained numerical limitation, not hidden as a scientific rejection. Increasing dual tolerance eliminated most earlier numerical stops but did not demonstrate useful editing.

| Case | Terminal status | Initial margin | Last accepted margin before rollback | Intermediate steps | Drift at last accepted step |
|---|---|---:|---:|---:|---:|
| 0 | NOT_SATISFIED | -6.750000 | -3.187500 | 3 | 0.073681 |
| 1 | NOT_SATISFIED | -13.843750 | -8.562500 | 3 | 0.071616 |
| 2 | NOT_SATISFIED | -10.250000 | -7.875000 | 3 | 0.041569 |
| 3 | NOT_SATISFIED | -14.500000 | -4.437500 | 3 | 0.163442 |
| 4 | NOT_SATISFIED | -14.968750 | -5.937500 | 3 | 0.140917 |
| 5 | SOLVER_NOT_CONVERGED | -13.687500 | -13.093750 | 1 | 0.017085 |
| 6 | NOT_SATISFIED | -8.562500 | -3.000000 | 3 | 0.140197 |
| 7 | BACKTRACK_REJECTED | -8.312500 | -8.000000 | 1 | 0.012044 |

The six step-limited cases were still improving, with last margins between-8.5625 and-3.0 and drift below0.3. Larger candidate steps often violated the unchanged referenceKL0.001, explaining backtracking. Because every final result restored Base, unchanged final generations and probe distributions do not establish successful protection. Full accepted/rejected trial summaries and available CG/KKT records are preserved.

Next separately frozen control: change max_edit_steps from3 to6, retain the complete8case denominator including known numerical/backtracking failures, and keep objective+0.01,QP dual_tol1e-6,CGsettings,trust/drift/KL,seed,prompt and199probes fixed. Compare the initial trial prefix against E15; no favorable case selection. This directly tests the six observed cap-limited trajectories; it is not a claim that more steps fix cases5/7. Finite7200seconds/1024GGN ceiling remains, estimated cost roughly1.8times E15 (about75minutes), not guaranteed. All incomplete/failed work must remain recorded if a hard cap is reached. No further tolerance relaxation is bundled into that test.

New cost290GGN/2426.387077791seconds; cumulative2851GGN/21306.461885863seconds.47CPU checks passed locally/remotely. Same8train-support targets/3images, unknown scopes, no verified replacement facts, Judge0. Raw medical materials private; no clinical or independent-generalization claim.
