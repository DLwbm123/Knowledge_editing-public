# E14 final: numerical QP gate blocks the stronger target objective

All8 cases terminated SOLVER_NOT_CONVERGED; none was accepted or generated the exact target answer. All edits rolled back, all Base/control checks passed, no generated matrices remain, and the worker exited without an exception. The unchanged mechanical regression passed. Thus the preliminary efficacy criterion failed, but this run cannot distinguish achievable answer editing from a numerical solver gate that prevented it.

Every terminal KKT record violates at least one fixed1e-7 absolute acceptance limit (10times dual_tol1e-8). The failing residuals are roughly1e-7 to2.18e-6, on target margins initially6.75 to14.97 below zero. The original terminal record did not retain CG subsolve receipts, so concurrent CG failures cannot be excluded from E14 alone. A separately reproduced CPU example with FP32 constraints and a14.97 RHS is rejected with primal residual1.91e-6, although both CG solves converge and the FP64 reference direction differs by only3.21e-9 max. This motivates a numerical-tolerance control, not a claim that the scientific objective has been fixed.

Three cases(0,2,3) made one accepted intermediate step before the QP gate stopped them; five failed at the first QP. Several early candidates improved margin but exceeded the unchanged reference KL0.001; drift stayed well below0.3. Therefore larger step or drift budgets are not the first justified change. After rollback all199 diagnostic probes and final generations reflect Base, which must not be presented as successful edit protection.

| Case | Initial margin | Accepted intermediate steps | Terminal status |
|---|---:|---:|---|
| 0 | -6.750000 | 1 | SOLVER_NOT_CONVERGED |
| 1 | -13.843750 | 0 | SOLVER_NOT_CONVERGED |
| 2 | -10.250000 | 1 | SOLVER_NOT_CONVERGED |
| 3 | -14.500000 | 1 | SOLVER_NOT_CONVERGED |
| 4 | -14.968750 | 0 | SOLVER_NOT_CONVERGED |
| 5 | -13.687500 | 0 | SOLVER_NOT_CONVERGED |
| 6 | -8.562500 | 0 | SOLVER_NOT_CONVERGED |
| 7 | -8.312500 | 0 | SOLVER_NOT_CONVERGED |

Next frozen run will expose the existing QP dual tolerance and set it to1e-6 for scientific cases (KKT absolute acceptance component becomes1e-5); mechanical regression keeps1e-8. This is an explicit numerical tolerance change, preserving the objective+0.01, native reference KL0.001, drift0.3, three steps, CG rtol1e-4, all data and rounding. CPU FP64 comparison and rejection of truly failed CG solves are required; future terminal receipts will include CG statuses/residuals and dual iterations. The original E14 failure remains unchanged and published.

New cost131GGN/1077.533234287seconds; cumulative2561GGN/18880.074805443seconds.46CPU tests passed locally/remotely. Raw QA/images/tokens remain private; Judge0, same3training images, unknown scopes and missing fact replacements remain BLOCKED_DATA for stronger claims.
