# Fixed TT write coordinates and reference boundaries

Let i,j in 0..63, a in 0..7, b in 0..3. The actual TT contracts B[(i,j),b]=sum_a G1[0,i,a] G2[a,j,b], A[b,(u,v)]=sum_c G3[b,u,c] G4[c,v,0]. For normalized input x, residual=x A^T B^T. Define C[(a,j),b]=G2[a,j,b] and implicit F[(i,j),(a,k)]=G1[0,i,a] delta(j,k); then B=F C and residual=F C A x in column convention. F is not stored or registered.

Take H=A A^T=L L^T without truncation. T=L^-1; G3'=T G3 and C'=C L. Thus A'=T A, C'A'=CA. All new arms use this same state. A' A'^T=I and the first/last two coordinates are orthogonal local/shared subspaces. G1 is fixed but need not be orthogonal. This is an input-coordinate gauge, not a claim that arbitrary TT cores are semantic scope bases.

The only trained tensor is the existing G2. Adam generates D in its view C; joint uses D P_joint and split uses [D_local P_local, gamma D_shared P_shared]. Histories are absorbed after the write and once per unique native/FIT input. Gating never participates in inference.

Reference paper: [ScopeEdit v1](https://arxiv.org/html/2607.01978v1), equations 14–24. It describes fixed orthogonal branches, evidence gates and recursive branch statistics. Reference code at [e90aa13](https://github.com/lab-klc/ScopeEdit/tree/e90aa1326ab1e2200e9dd8b9d9e4ed98c2102a6f): lora_layers.py uses gamma_self=gamma*mix_self, normalized mixed shared keys, and recursion coupled to enabled writes; MORE.py selects participating groups. The yaml enables locality-key accumulation, and the evaluation script accumulates loc/loc_image. These details are not copied into this experiment.

This implementation keeps the original TT/routing/layer, uses question-only features, gates the actual Adam candidate update, and absorbs identical ungated native/FIT keys across all arms. No replay CE or locality evaluation data are admitted. It is ScopeEdit-inspired, not formula-for-formula reproduction. The user amendment changes the attachment's replay objective and reduces 24 experts to a fixed first-eight pilot; historical CE192 is the relevant continuation reference.

Historical evidence: TT_FIXED_A gave no established gain; intrinsic route variants degraded; U/HOLD results did not establish qualified protection. The previously proposed U_ONLY/HOLD stage was partly blocked and must not be represented as fully executed merely because the attachment calls it a negative result. W0+MARGIN_002 remains the primary baseline.
