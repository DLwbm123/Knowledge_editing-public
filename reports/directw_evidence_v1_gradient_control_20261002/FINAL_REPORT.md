# E22: direct-gradient control also fails protected target generation

All eight cases completed with zero accepted final edits and zero actual full-target sequences after accepted edits. Six ended BACKTRACK_REJECTED, two reached six steps. All initial score/token controls, mechanical exact replay, trace scalar/repeat checks and Base restores passed. No scientific GGN call was used;11GGN came from mechanical validation. This is a negative result for this bounded direct-gradient baseline, not a proof that all direct-weight methods are infeasible.

|Case|E20 QP best accepted-state margin|E22 gradient best margin|Difference|E22 status|
|---|---:|---:|---:|---|
|0|-3.8125|-6.25|-2.4375|BACKTRACK_REJECTED|
|1|-4.0625|-13.0|-8.9375|BACKTRACK_REJECTED|
|2|-10.0|-10.0625|-0.0625|BACKTRACK_REJECTED|
|3|-13.9375|-3.8125|+10.125|NOT_SATISFIED|
|4|-6.3125|-6.625|-0.3125|NOT_SATISFIED|
|5|-9.6875|-10.5|-0.8125|BACKTRACK_REJECTED|
|6|-1.0625|-1.875|-0.8125|BACKTRACK_REJECTED|
|7|-6.625|-7.9375|-1.3125|BACKTRACK_REJECTED|

Best-state metrics include clean Base and accepted intermediate updates only. Only1/8 cases improves by>=0.01 versus E20, failing the frozen>=5/8 diagnostic prediction. Case3's gain of10.125 is retained, but it still ends at-3.8125, below the+0.01 acceptance margin. The other seven cases are worse by this metric. No learning-rate sweep is triggered.

All six terminal backtracking stages fail native reference-KL protection at every factor. At their smallest factors, native KL is approximately0.001934/0.001802/0.001867/0.002297/0.002025/0.001817 for cases0/1/2/5/6/7. Five of these six have FP32 functional KL evaluated at the same rounded matrix below0.001 (case6 is the exception). These FP32 values are diagnostic, not valid substitutes for native protection: the forward precision/input representations and precision-specific teacher distributions differ. This recurring disagreement motivates a full native precision control, not relaxation of the native KL gate. In particular, case2's last native KL0.001867 contrasts with FP32 diagnostic KL0.0000036513.

All199 final probe slots and final generations are unchanged after rollback. This does not establish successful editing with protection. No scientific matrix was generated, no worker/lease/runtime error remains, and GPU5 is free at completion. New11GGN2957.0459919100394seconds; cumulative5927GGN47627.349447437795seconds. This run is cheaper than the sequential E20 reference but has no demonstrated efficacy or general throughput advantage. Full numeric trials/position traces are retained in FINAL_RESULTS.json. Raw QA/images/tokens/weights remain private.

Next justified diagnostic: hold this update algorithm, data, six steps and numerical target/protection thresholds fixed, and promote the initialized native model and input pipeline toFP32 using the existing E6 loader. This jointly changes write precision and end-to-end computation, including image features and Base scores; it cannot isolate those causes. E6's earlier positive likelihood-gain results do not answer the current full-generation question. Any success in a new FP32 control must be labeled FP32-only feasibility, never original BF16 deployment success. Freeze resource headroom, new-precision controls and success criteria before launch; do not preserve obsolete BF16 absolute-score fingerprints across an intentional precision change. No seed selection, extra steps, protection relaxation or repeated gradient baseline. Formal medical/new-fact/independent-image claims remainBLOCKED_DATA,Judge0.
