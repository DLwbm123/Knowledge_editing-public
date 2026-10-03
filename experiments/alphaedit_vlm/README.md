# AlphaEdit on native LLaVA-Med

This is a bounded **VLM adaptation**, not a reproduction of the original LLM
benchmark scores. The reference implementation is
[AlphaEdit b84624f](https://github.com/jianghoucheng/AlphaEdit/tree/b84624f44dfe8fc6cd9e41df916c44124a0c46dc).
Its MIT notice is retained in `UPSTREAM_LICENSE`.

Preserved: latent-vector Adam optimization, the official KL direction, norm
penalty and clamp; small-eigenvalue second-moment projection; the AlphaEdit
weight solve and residual distribution over layers. For independent single
requests the history covariance is zero. The exact rank-one identity used here
is tested against the official dense solve. This implementation does not claim
sequential editing support or evaluate cumulative-edit retention.

Explicit adaptations: native image/question chat input, lookup at the last
expanded prompt token, a single native context instead of synthetic textual
contexts, target answer plus EOS supervision, a distinct-image QA prefix for
latent KL, and statistics from all 447 other English-QA SLAKE training images rather than
100,000 Wikipedia documents. Layers are fixed at `[21]` and `[19,20,21,22,23]`;
the latter is an architectural choice around the existing editing site, not a
validated optimum or the official Llama3 layer configuration.

Model parameters stay frozen during latent optimization. Updates are written
to original BF16 `down_proj` weights. Raw edit outcomes are evaluated without
the Direct-W QP, trust radius, preservation acceptance gate, or line search.
Every case starts from Base and is independently reloaded in an ordinary native
model process with no editor import or hooks. Complete generation, preservation
metrics, and reload parity are retained before generated matrices are deleted.

Run the CPU check with `python -m experiments.alphaedit_vlm.tests`. The remote
runner is configured by a private manifest; launch via the existing environment
using neutral `python -` entry points and only the authorized physical GPUs.
No model/data downloads, judge calls, sweeps, automatic retries, or ongoing
Codex monitoring are part of this campaign.
