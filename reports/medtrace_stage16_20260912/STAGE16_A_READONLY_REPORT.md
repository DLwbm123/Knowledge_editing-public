# Stage16 A — read-only routing/writer decomposition

COMPLETE: 400 historical writer checkpoints, 2514 completed writer-probes, 2426 frozen Judge tuples. No training, generation or rescoring.

Stage15 primary RC_FIXED_OLD16 and all original results remain unchanged. All new threshold curves are post-hoc diagnostics, not calibration or independent confirmation.

Old kappa=0.7696741135364367; accepted iff R0_ON and margin >= kappa, margin=(r-d)/max(r,1e-12). Equivalent positive-radius d/r <= 0.23032588646356333.

| Writer | I-General R0 | I-General old RC | I-Local R0 preservation | I-Local old RC |
|---|---:|---:|---:|---:|
| C_NO_H | 158/164 | 2/164 | 53/67 | 67/67 |
| BE | 159/164 | 2/164 | 49/67 | 67/67 |

The old RC image-locality preservation is predominantly rejection-to-Base. It does not demonstrate a writer that preserves locality or a transferable cross-image acceptance region.

R0 image-locality: 49 OFF in both methods; among 18 ON, C_NO_H preserves 4 and damages 14, BalancEdit preserves 0 and damages 18. Paired discordants 4 vs 0 yield supplemental two-sided exact p=0.125. The original edit-bootstrap interval is retained separately, not overwritten.

PORTABILITY_BASE_DELTA.csv compares each hop and exact/semantic score against Base. Source semantic accuracy is not output locality; counterfactual target adherence is not clinical truth. Exact target-copy proportions use whole normalized answers only, not substring or semantic inference.

Known observed standard-probe source components: 151. SOURCE_GROUP_SENSITIVITY.csv supplements the edit-based estimates; patient/study identity and unobserved near-duplicate relations remain UNKNOWN.

ROUTER_PAIRWISE_AUROC.csv keeps native, text-generality and image-generality denominators separate and compares each with each official locality category. Official categories are scope proxies, not newly clinically reviewed H labels. No combined native+image-general coverage or H-fit-as-held-out claim.

All 1354 attainable nonnegative inclusive/crossing breakpoints are reported. No threshold is adopted from the old 200 edits. Independent Base-only development support is required before RC_EXTCAL_V1.

## Threshold-migration diagnosis — post hoc only

Image-general vs image-local margin AUROC is 0.981616 (164 positives, 67 negatives), not a pooled native-plus-image-general metric. The other separately defined positive/locality comparisons have AUROC 1.0 on this historical support. These are benchmark-role diagnostics, not independent clinical scope labels.

Among reported historical breakpoints that retain all 200 native activations, at least 190/200 text-general activations, and at least 152/164 image-general activations (R0: 160/164; at most 5 pp loss), 118 breakpoints are feasible. The minimum observed image-local activation is 5/67, occurring over the listed breakpoints from 0.1658721541392988 to 0.1712490825803072; image-general activation there is 152/164. This suggests a threshold-transfer problem worth independent Base-only calibration, with residual overlap. It does not establish a new threshold, guarantee future performance, or justify searching new router representations. The historical primary remains kappa=0.7696741135364367.

At the historical R0, C_NO_H's locality output-preservation gain over BE is +5.97 pp, original edit-bootstrap interval [1.49, 11.94] pp, supplemental exact p=0.125. The source-semantic accuracy gain is only +2.99 pp, interval [0, 7.46] pp, exact p=0.5. These are different endpoints.

For one-hop portability, exact success rises from Base 0/73 to 1/73 for both writers, but semantic success falls from Base 21/73 to 7/73 (C_NO_H) or 8/73 (BE). Two-hop semantic success falls from Base 8/24 to 1/24 or 0/24. Thus the small exact gains are not evidence of reliable multi-hop transfer.
