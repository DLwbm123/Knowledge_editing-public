# E9A launch receipt; paired experiment incomplete

2026-10-01 hourly follow-up. E8 final results publishedb36b554 with public HTTP200 verification. Both E9 arms were frozen in protocol commit200371a before armA launch. Exact source bindings matchbfe5c46;42 CPU tests reused. Static assertions confirm deterministic selection, target uniqueness, old-ID/question-hash exclusion and distinct-image references. All18 remaining candidate QA rows match original local train annotations;3images readable. Public selection digests are in SELECTION_AUDIT.json.

20261001E9A launched detached on pro5000 GPU5 UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca, PID1333578, neutral argv `/data/bmw/envs/s0/bin/python -`. Minimum66000MiB free precondition passed. New RUN_ROOT and source snapshot preserve prior runs. /data is ext4 on/dev/sdc1 with17.1TiB available; write/read probe and28GiB free-space requirement passed.

Initial check19seconds after launch: PID alive, stateNATIVE_REGRESSION,0/8 scientific cases, native result pending. GPU5 used15548MiB/free57283MiB at device snapshot; process query moments later15550MiB. Log34282characters, no traceback marker or failure receipts. These are startup observations, not native PASS or completion.

ArmA new budget7200seconds/1024GGN, prior1749GGN/11677.34334912477seconds. Pair total ceiling14400seconds/2048GGN, sequential only; Judge0/storage20GiB/cache2GiB per arm. After auditing/publishing A, the next hourly wake-up executes already-frozen E9B on identical QA/references with seed20261002 and A's actual cumulative ledger. Do not call the pair complete at A, change membership after results, or select a favorable seed.

No model code changed, no other GPU/process altered. All new QA share old image groups and remain exploratory train-support. Native consumers remain mandatory for successful edits. Raw QA/images/tokens/configs/logs stay private; formal W_EVIDENCE/data admission remains BLOCKED_DATA.
