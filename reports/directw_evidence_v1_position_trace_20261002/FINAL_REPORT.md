# E21: exact replay localizes mostly fixed bottlenecks

All eight scientific trajectories exactly match E20; the mechanical replay, historical controls, vector/scalar/repeat checks and Base restoration pass. Observation validityPASS. Only case6 changes its worst-position set across accepted updates, so the frozen at-least2/8 switching hypothesisFAILS (observed1/8). Efficacy remains0/8, as expected under exact replay. This does not establish impossibility of direct weight editing.

| Cases | Worst native target position through accepted updates |
|---|---|
|0,1,2,5,7|Position0 (first content token), unchanged|
|3,4|Final position1 (EOS), unchanged|
|6|Position0 initially, then final position1 (EOS) after the second accepted update|

Case6 progresses from[-8.5625,-3.5] to[-0.75,-1.0625]. Case1 ends with two near-equal negative margins[-4.0625,-4.03125] but never switches its worst position on accepted states. Rejected candidates are not counted as accepted states. All sets are exact minima with ties retained; no tie-breaking is used.

Among the five terminal backtracking stages, cases0/7 have no factor passing protection. In cases2/3/4 only the smallest factor passes protection and its minimum stays flat. Cases3/4 have completely unchanged vectors; case2 leaves the first three positions unchanged but EOS worsens from-3.75 to-3.8125. Case7's smallest rejected candidate improves EOS from-2.9375 to-2.875 while its first-token minimum stays-6.625, but protection still fails. Thus widespread bottleneck switching is not supported; jointly adding token constraints is not justified solely by this trace.

All final generations and199probe slots are unchanged after rollback, not evidence of successful editing with protection. No scientific matrix was generated, no matrix/lease/process/runtime error remains, and GPU5 is free at completion. New519GGN4265.08870057296seconds; cumulative5916GGN44670.303452876746seconds. FINAL_RESULTS.json contains all numeric traces and trial diagnostics; raw QA/images/tokens/weights stay private.

A more discriminating successor is a matched simple direct-gradient control on the same W, native minimum-margin acceptance, BF16 write rule, protection thresholds and finite step budget. It replaces the QP/curvature direction with a gradient direction while retaining the same FP32 derivative surrogate and native score deficit. This tests the current update rule, not capacity of all possible weight edits. A failed finite baseline cannot prove infeasibility; a successful baseline would be evidence against the necessity of the current QP direction in these cases. No blind repeat, seed sweep, extra steps or protection relaxation is justified. Freeze the baseline definition and criteria before execution. Clinical/new-fact/independent-image claims remain unsupported; scientific data remainsBLOCKED_DATA,Judge0.
