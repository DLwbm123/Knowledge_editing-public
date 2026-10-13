# C/O/L checkpoint lifecycle

C and all original146/CPOL assets are read-only. Only physical GPU5 is authorized. The original146 frozen run is unchanged.

The finite local dependency consumer must observe original TRAINING_GENERATION_COMPLETE, FINAL_CONSUMED, EXIT_gpu=0, EXIT_reload=0, and no open GPU lease before installing this run. Its installation starts the new 72-hour clock once; waiting is excluded. O GPU, O fresh reload, L GPU, L fresh reload execute serially, with a shared 36 GPU-process-hour cap. A failed admission stops, never retunes.

Each step atomically saves native W, FP32 master, Adam m/v/step (O), RNG including NumPy, source step/owner clock, cumulative committed count and loss/norm event. Recovery requires identity, count and journal reconciliation; no blind process restart. Attempted generation without durable output and attempted Judge are never automatically replayed.

The active state lives under private/arms/<arm>/private/active and is kept until all 146 panels and a native final capture are durable. Final capture is only the last single edit, not a continuous bank. Delete it only after the fresh native reload reproduces the two registered outputs exactly and records schema/parameter/storage checks. Keep logs, masks, bindings, scores, failure costs and compact receipts. All newly owned matrices, including atomic temporary files, stay below 2 GiB; free space at least 8 GiB. Do not delete historical checkpoints or old outputs.

The four one-attempt score workers stream completed panels; terminal scoring requires the global generation barrier and no pending/reserved payload. Final reporting validates all 7,116 C/O/L consumers, shared missing identities, individual protection and per-edit gates, and retired GPU leases. Publish only source and anonymous aggregates; verify pushed SHA and anonymous access. A scientific failure or permanent missing is a valid reported outcome, never an excuse to retry. Engineering/budget partial runs retain all state and require an explicit partial report before closure.
