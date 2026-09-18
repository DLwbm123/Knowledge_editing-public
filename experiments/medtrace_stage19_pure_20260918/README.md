# Stage19 FASTTRACK source overlay

Apply the prior Stage15-19 published source overlays, then this overlay. Reuses the frozen FP16 LLaVA-Med runtime, FP32 writers, HSIC Top1 and original BalancEdit recipe. Private source manifests and authorized model/image files are required; none are distributed.

Run CPU checks with `python -m scripts.medtrace.test_stage19_fasttrack` and the existing Stage18/19 checks. The supervisor charges every GPU worker attempt against one 28800-second ledger. The local controller performs frozen source-agreement scoring and publishes only aggregate reports. Two arms share actual background experts in track S; total bank N is not FACT coverage K. See the corresponding report directory for realized N, coverage, resource usage and limitations.
