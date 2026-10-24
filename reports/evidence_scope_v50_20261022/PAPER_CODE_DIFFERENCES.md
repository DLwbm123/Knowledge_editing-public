# Formula and implementation provenance

| Component | Verified source | Implemented adaptation / limitation |
|---|---|---|
| BE radius | BalancEdit v2 Eq3 vs official commit83749e52a1d27331d21cfec845b6089294730c2f `mid_epsilons` | Code alpha=.2 weights negative distance .8 and positive .2; equivalent paper-alpha=.8. Endpoints and numerical equivalence tested. Not two runs followed by selection. |
| BE key | Official adapter averages layer input across sequence | Frozen Base prompt-only down-projection inputs; excludes target answer, EOS and padding by separate prompt preprocessing. Original code has full-sequence averaging and answer-dependent indexing; not copied as inference leakage. |
| BE black branch | Official debug branch copies `locality_tokens` and replaces image | Explicitly retains source question and normal RGB preprocessor. Black is an input proxy, never medically verified negative. |
| BE editing | Official transformation/codebook | Shared TT W0 delta; two anchors for two explicitly supported facts. Single-edit scope adaptation, not original full reproduction or multi-edit deployment. |
| LDKE localization | v1 Eq1–5 | Final model norm + unembedding, pre-FFN residual and residual+MLP; source first target token only; layers16–31, top1. No whole-answer probability assertion, no causal proof. |
| LDKE route | v1 Eq8–18 | Residual projection, L2 embedding, dual margin, absolute similarity, BCE. Threshold.5 fixed. Auxiliary classifier reads residual representation as sec3.4 specifies; Eq17 notation uses normalized h, ambiguity disclosed. |
| LDKE training | Paper appendix reports meta-editor schedules but not complete router widths/loss weights | Explicit per-edit source-local engineering settings, 100steps, lr.01, margin.1, coefficients1, bottleneck32, embedding64. No official repository verified. BLACK_PROXY training arm named LDKE_ROUTER_TT_PROXY, not true medical-scope reproduction. |
| B evidence | User-specified new design | Frozen projected patch tokens and question embeddings, cosine pooling, fixed block means. WEAK_EVIDENCE and weak relation representation; no ROI or anatomy claim. |
| B writer | User-specified central rank modulation | Gate frozen and shared; conditioner zero-final-map, rho.5, central bond4. Paired independent writer training. TT canonical coordinates change under the inherited canonical RMS update; gamma is defined in the stored canonical basis, not a basis-invariant operator modulation. |
| B evidence loss | User-specified optional term | Mean answer-token logp excludes EOS. Unit-tested rejection without verified medical pair; disabled in this run. |

Sources: [BalancEdit](https://arxiv.org/pdf/2505.01343), [official code](https://github.com/donglgcn/BalancEdit/blob/83749e52a1d27331d21cfec845b6089294730c2f/easyeditor/models/BalancEdit/balancedit.py), [LDKE v1](https://arxiv.org/html/2605.29826v1).

Implementation failure: initial real admission encountered nonfinite router loss because the inherited LLaVA loader disables Linear/LayerNorm reset_parameters globally. New modules now explicitly initialize their own tensors. No pretrained weights are reinitialized. The failure happened before formal writer updates or generation; original receipts and costs retained. Added regression test simulates this exact loader behavior.
