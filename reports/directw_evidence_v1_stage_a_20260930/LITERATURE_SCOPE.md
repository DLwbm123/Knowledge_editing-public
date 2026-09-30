# Actual literature scope (2026-09-30)

The named `knowledge-editing-literature-list.md` was not found in the provided
Downloads material, nominal checkout or active checkout. No mechanisms were
inferred from missing titles. This missing supplementary input does not block
the explicitly specified CPU implementation.

[CrispEdit arXiv:2602.15823v2](https://arxiv.org/html/2602.15823v2), revised
2026-05-01, was read in its abstract and method sections 2/3. The paper motivates
capability preservation through Bregman/Gauss-Newton geometry and uses K-FAC
and a matrix-free low-curvature projector for scale. DirectW borrows the goal
of functional protection. It uses exact categorical GGN matvecs, an explicit
preservation gradient, L2-slack QP and actual-forward gates instead of that
projector. Its visual-pair and fixed-capacity-history rules are project design.
It is not a CrispEdit replication and does not inherit its empirical claims.

ScopeEdit, M-ORE, DOW-KE and MetaKE were not independently mechanism-verified in
this pass. No reproduction, equivalence or novelty claim is based on their titles.

[Official torch.func documentation](https://docs.pytorch.org/docs/2.14/func.html)
was consulted for functional transforms and their limitations. The accessible
online page is PyTorch 2.14; this run actually used installed PyTorch **2.6.0**.
The 2.6 documentation URL was unavailable through the browser. Actual CPU tests,
not newer documentation alone, establish local synthetic JVP/VJP/functional_call
behavior. No existing environment was upgraded. The original multimodal kernels
and native deployment backend remain unverified.
