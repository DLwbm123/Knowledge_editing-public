# Interrupted exploratory run: 20261001E1

Live inspection on 2026-10-01 at 12:10 China time confirmed that the detached process had exited. The final status is **STOPPED_ON_ERROR**, not experiment completion. It stopped at 11:48:39 China time after 845.75 seconds (14.10 minutes).

Four of eight cases completed: three BACKTRACK_REJECTED and one NOT_SATISFIED. None was finally accepted. Intermediate accepted steps (1, 0, 1, 3 respectively) are not successful edits: all four transactions rolled back to Base. Completed cases had exactly zero same-precision native/functional logit discrepancy. Peak PyTorch allocation was approximately 57.78 GiB.

The fifth case (zero-based index 4) failed with CUDA OOM when requesting 66 MiB with 39.69 MiB free. The runtime error recorded approximately 56.49 GiB for this process and 14.58 GiB for a concurrent process. The concurrent process's ownership and purpose were not established; no other process was modified. The failed case rolled back and the final Base restoration receipt is true. Cases 5–7 were not attempted. No native edited-matrix export was produced because no case reached final acceptance.

The preliminary native regression was marked PASS for the implemented mechanical checks: same-precision parity zero, finite exact GGN, symmetry error 6.0174e-14, nonnegative PSD probe and exact repeatability. Its edit itself was BACKTRACK_REJECTED with restoration; PASS does not mean successful native editing or new export/reload qualification.

New GGN calls: 115, including 11 regression calls and one call in the failing case. Historical cumulative calls: 161. Prior native time remains 144.3932 seconds; combined with this run it is 990.1431 seconds. Judge calls: zero. The negative completed cases and interrupted remainder are retained; this partial run cannot be summarized as a full eight-case success rate.

Machine-readable sanitized evidence: [INTERRUPTION_RESULTS.json](INTERRUPTION_RESULTS.json). Private source rows, prompts, generated tokens and raw logs remain on the permitted private storage. The earlier [launch report](RUN_REPORT.md) is historical and is superseded by this status. No retry, parameter change or subsequent experiment was launched during this status check.
