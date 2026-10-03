# A1: frozen AlphaEdit VLM baseline

Authorized by the user's 2026-10-03 request to reproduce AlphaEdit on physical
GPUs 5, 6, 7 of pro5000. The older Direct-W campaign and its automation remain
paused. This is a separate method campaign; its cost does not reset that ledger.

## Scope and fixed comparison

- Native LLaVA-Med v1.5 Mistral 7B, existing local model/vision assets and input
  path, native BF16 deployment, deterministic greedy generation up to 64 tokens.
- Two prespecified arms: single layer `[21]`, and layers `[19,20,21,22,23]`.
  The five-layer placement is a fixed architectural adaptation around the
  previous site, not a tuned or independently validated layer selection.
- Same eight target/reference pairs and 199 probe slots as E23. Each case starts
  from Base, so history covariance is zero. No sequential-edit claim.
- Existing examples are original training answers on three previously developed
  images. This evaluates answer reinforcement, not independently held-out new
  facts, visual scope, or clinical correctness. Exact-match failures do not prove
  semantic answer errors; raw private generations and lexical F1 are retained.

## Method fidelity and explicit VLM changes

Reference: jianghoucheng/AlphaEdit commit
`b84624f44dfe8fc6cd9e41df916c44124a0c46dc`, MIT license.

Keep latent Adam (`lr=0.1`, 25 objective evaluations, official last-iteration
stop behavior), norm penalty `0.5 * ||delta|| / ||initial||^2`, clamp factor 0.75,
KL factor 0.0625 with official `D(current || Base)` direction, small-eigenvalue
projection threshold 0.02, and weight-solve L2=10. Residual is divided by the
number of remaining layers. The single-request closed form is algebraically
equivalent to the original solve with empty history; CPU test compares both.

Adaptations are native image+question input, last expanded prompt-token lookup,
one native context instead of synthetic text contexts, answer plus EOS NLL, and
a distinct-image native QA prefix replacing the original textual KL prompt.
Vision components stay frozen. Collect keys from the **input of down_proj**,
and optimize the target at the **output of the last edited transformer block**.
No target answer token is included when extracting the editing key.

Raw AlphaEdit updates are written into physical BF16 weights and evaluated.
There is no Direct-W QP, trust radius, line search, or protection-based rejection.
Record requested/actual update norms, BF16 rounding error and update components
outside the estimated nullspace; do not interpret approximate linear key
preservation as exact whole-model output protection.

## Projection statistics

SLAKE train contains 447 other eligible English-QA images after excluding all
three target/reference/probe/smoke images. Use **all 447 images**, one English QA
per image selected by seed 20261003 from sorted source QIDs. Collect every valid
expanded prefix token, including image tokens and question/template tokens;
exclude all answer tokens. Estimate the uncentered mean second moment at each
of layers 19–23 and retain eigenvectors with eigenvalues below 0.02.

This is a bounded multimodal substitute for the official Wikipedia statistics,
not a claim to have reproduced the original 100,000-document statistics. Record
token count, complete eigenvalue spectrum, nullspace rank, and a sampled basis
orthogonality check. Empty nullspaces are retained as outcomes, not retuned.
Evaluation probes never fit P. Image disjointness does not imply that these
training images are unseen by the pretrained model or past project development.

## Evaluation and controls

Before statistics, use the two reserved mechanical rows to verify finite,
nonzero native latent gradients at layers 21 and 23, no frozen-weight mutation,
and hook cleanup. This check does not establish editing efficacy.

For all eight cases, record Base/edited full generated tokens including EOS,
normalized exact match, token F1, truncation, target NLL, all 199 probe answer
log-probability changes and Base-to-edited KL, elapsed time, peak GPU allocation,
and every latent objective value. No external judge calls.

Primary shared criterion: at least 2/8 previously nonexact targets become full
target-token exact (including EOS), no probe score drop below -0.1, and all
independent native consumers pass. Also report efficacy and preservation
separately regardless of this composite outcome. KL>0.001 is a diagnostic.
Compare each arm with its own fresh BF16 Base; E23 FP32 is contextual evidence,
not a precision-matched superiority comparison.

Each physical edit is saved and reloaded in a fresh native model process without
editor import or hooks. Require identical target logits and full generated
output, audit untouched parameter/buffer versions, then restore Base exactly.
Controls failing stop the affected worker and retain evidence; do not silently
count infrastructure failure as scientific failure or omit it from denominators.

## Execution, finite budget and lifecycle

GPU 5: native mechanical qualification and shared statistics. GPU 6: single
layer; GPU 7: five layers. Editing workers wait without loading models until
statistics are ready. No use of GPUs 0–4, no stopping existing processes.
All main/consumer commands are neutral `python -` entries. Require at least
52,000 MiB free per authorized GPU at launch and 28 GiB free on the actual data
mount. Four-hour campaign wall deadline, conservatively at most 12 GPU-hours;
no automatic retries, parameter sweeps or successor experiments.

At most one generated independent edit per arm is retained at a time. The last
required consumers are edited generation, all probes and independent reload.
After these complete, delete that case's generated matrices and save a deletion
receipt. After both arms finish, delete shared projection tensors; retain the
spectra, manifests, code, configuration, raw private outputs and metrics.
Incomplete/failed consumer artifacts remain for diagnosis. Peak output allowance
20 GiB plus 8 GiB safety margin; shared Base assets are never deleted.

Launch detached, check startup once, and return status. No new monitoring
automation. When completion is subsequently inspected, publish source, protocol,
sanitized results and report to the project GitHub repository through the proxy;
do not publish private data, images, questions/answers, checkpoints or raw logs.
