# Appendix E1: matched Direct-LR4 supervision

This overlay runs after the parent decomp-24h phase closes. Copy the parent runtime Python modules, then overlay this directory's worker, training, bindings, report and controller modules. Reuse the existing environment; do not upgrade dependencies. `bootstrap.py` audits parent teacher bindings, creates a new run once, carries forward lifetime Judge attempts, and reserves complete E1 scoring capacity. Runtime/private inputs and weights are never published.

The parent W0 checkpoints were reclaimed. E1 therefore rebuilds Direct-W0 and forks **both M6 and M7** from that same state; the old parent M6 is not claimed as an initialization-matched reference. M7 uses the parent's S+G+D objective, optimizer, sampler, 80 updates and verified CP-M0 teacher. “SGD” names supervision, not the optimizer.

Frozen order: two REG24 canary edits, then the remaining REG24 edits, then real M6/M7 bank prefixes 12/24. Only GPU 5/6/7. A complete E1 pair reserves 1700 Judge attempts (1440 formal worst-case consumers plus 260 canary diagnostic reserve), within the inherited lifetime 6000 ceiling. The canary requires training/save/reload/generation/Judge closure before expansion. Existing failed keys remain missing; no retry.

E2, DEV24 expansion, E3 and E4 are **not implemented or admitted by this E1 controller**. They require the already-authorized plan's separate complete-pair budget audit. E1 completion is not completion of the entire appendix. No independent CONFIRM. The original clocks and actual consumption are preserved; user-authorized compute/storage quota waivers do not authorize extra paid Judge attempts.
