# MedTRACE CP-independent visual pooling: pre-start review snapshot

Source implementation: research commit `f19ecb7b7127eaf838448b65abe293794d638593`, branch `medtrace-cp-independent-pooling-20260907T060435Z`, based on `2f142769a26971b12f6e089653e4065d697fbdf6`.

**Not an executed experiment.** The GPU occupancy guard stopped initialization before any model load or fit. This snapshot is published at the user's request for GPT Pro review. Historical failure conclusions are unchanged.

Start with the [review and status](../../reports/medtrace_pooling_20260907T060435Z/GPT_PRO_REVIEW.md) and [pooling configuration](../../reports/medtrace_pooling_20260907T060435Z/POOLING_ABLATION_CONFIG.json).

Use the existing research environment; no installation or training is needed for the CPU regression:

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m unittest discover -s tests/medtrace -p 'test_v*.py' -v
```

Actual execution requires the private frozen input/cache/checkpoint/runtime bindings. They are intentionally absent from this source-only snapshot. Do not infer completed reproducibility or clinical validity from source availability. The conditional new-edit confirmation execution path is not yet connected; see the review limitations.

The runner reuses the existing bounded coordinator and queue. This snapshot does not authorize GPU sharing, start jobs, or modify any historical report directory.
