# Mathematics and numerical interpretation

For a fixed expanded prefix, label at position t+1 is predicted by logits at t.
Content scores average log p of non-padding answer tokens per answer; EOS is
excluded by the proposed content policy and tested separately. The actual native
template and any EOS policy change must be frozen before native admission.

For a legal pair, s(I)=ell(y+|I,q)-ell(y-|I,q). The cross-image constraint is
s(I+)-s(I-) >= m_vis. Both correct-endpoint margins are required separately,
and the negative end has an independently budgeted anchored KL. If endpoint
thresholds sum to at least m_vis, the cross constraint is redundant; its presence
cannot count as additional evidence. A matched question is not causal identification.

For group g, D_g is the full-vocabulary KL(q_ref || p_W), averaged equally by
source, then sample, then effective predictor position. Group coefficients are
explicit, not implied by group sample counts. q_ref and its prefix are immutable.
For unrelated facts q_ref is clean Base; for corrected history it is the accepted
distribution. Current p_k is used for local geometry, not as a moving actual teacher.

`FrozenGGN` computes Fv = J^T{w_pos * p_k * [Jv - <p_k,Jv>]} using JVP and VJP.
There is no parameter-sized covariance or explicit Jacobian in production. The
bounded CPU oracle alone constructs a Jacobian to independently check this result.
The local categorical GGN is PSD. It is not the exact full KL parameter Hessian
away from an anchor, and the actual KL's first derivative is retained.

Let Q=tau I+sum_g lambda_g F_g, g=sum_g lambda_g grad D_g,
A_j=grad c_j, b_j=m_j-c_j. The hard reference minimizes
0.5 d^T Qd+g^T d subject to Ad>=b. Its bounded exhaustive active-set oracle is
restricted to 64 variables and 10 constraints; it is not a large-model solver.

The operational solver instead minimizes

    0.5 d^T Qd + g^T d + nu/2 ||xi||^2
    subject to Ad+xi >= b, xi >= 0.

For alpha>=0, differentiating the Lagrangian gives
d=Q^-1(A^T alpha-g), xi=alpha/nu. Substitution yields the dual

    alpha^T(b+A Q^-1 g)
    - 0.5 alpha^T(A Q^-1 A^T + I/nu)alpha,

up to an alpha-independent constant. CG computes Q^-1g and Q^-1A_j, never an
inverse. Nonnegative coordinate descent operates in the <=8 constraint space.
Reports include stationarity, primal infeasibility, complementary slackness,
projected dual residual, CG actual relative residual/iterations/calls and a
Rayleigh-quotient condition surrogate (not a true condition number).

The soft optimum can have nonzero slack even when CG and dual KKT converge.
This is not a successful edit. A finite nu can deliberately fail a hard fit
threshold while preserving an anchor. The visual feasibility fixture uses
nu=1e8 specifically to test a near-hard solution; this is not a native tuning
recommendation. Native nu, tau and acceptance thresholds remain unapproved.

CG repeats a fixed probe and checks actual residuals to catch a changing operator;
this is a diagnostic, not a proof against every possible impure callback. Runtime
inputs and Torch CPU RNG are frozen. GPU RNG/backend determinism is pending native
qualification; no alternative curvature backend is silently substituted.

Trust-radius clipping changes the QP step. Every finite backtracking factor
is rechecked using normal forward at the rounded deployed weight. Merit is the
sum of squared violations over **all** fit constraints; its roundoff tolerance
uses squared-score units, independently of score satisfaction tolerance.
Protection comparison has 1e-12 CPU numerical allowance, not an implicit budget
increase; real BF16/FP16 KL and parity tolerances require review and native measurement.

W_KEY_QP computes DeltaW -> (DeltaW K^T * w_pos) K; output metric is identity.
W_EUCLIDEAN_QP uses only tau I, but keeps g and actual KL checks. W_FT has an
explicit update penalty and bounded gradient steps. None is named as an author
implementation of a paper method.

For the linear diagnostic only, ||DeltaW k+-r||+||DeltaW k-|| is bounded below
by [||r||-rho||k+-k-||]+ when ||DeltaW||_op<=rho. The CPU test checks 100 random
fixtures with dimensionally consistent r. This does not prove an autoregressive
VLM uneditable, justify sample removal, or define a required hidden target.

The full vocabulary is retained in teachers and curvature. No K-FAC, top-k,
diagonal approximation or low-rank factorization has been inserted. Runtime
host reset/transaction copies and large gradient rows are counted in resource
estimates. Bounded historical retention is finite replay, not unlimited memory.
