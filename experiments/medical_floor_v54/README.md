# Explicit positive increment floor

V54 adds only relu(0.1-delta_positive) to V53's relative pair and negative
increment constraints. The threshold reuses the frozen pair margin, weight1.
`objective.selfcheck()` checks gradients and explicitly detects the case in which
both old hinges pass despite a decreasing positive score.

Use the existing guarded staging workflow with V54_ROOT. Both real admission
receipts are required before STAGE1_GO. No control retraining; Base and CE outputs
and exact prior verdicts are reused. Per worker160 updates/2632 backwards including
72 admission;32 new formal outputs+2 smoke across both workers. Run evaluate.py
validate/blind; grade only new blinded outputs, freeze, then summarize. The fixed
development set has prior exposure and cannot establish independent confirmation.
