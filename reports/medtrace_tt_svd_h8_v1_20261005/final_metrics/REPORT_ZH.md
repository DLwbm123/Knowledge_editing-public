# TT/SVD结构与FREE配对消融结果

原146登记中的固定H8，64续训，非146全库。TT/SVD独立初始化；同路线KEEP/FREE共享warm函数，匹配更新/样本但不匹配坐标或墙钟。SVD不是比FREE更紧凑的参数化。

|臂|任务|编辑|分母|已知正确|missing|宏界|微界|
|---|---|---:|---:|---:|---:|---|---|
|TTKEEP_NO_H|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|TTKEEP_H1|T0|8|8|7|0|[0.875, 0.875]|[0.875, 0.875]|
|TTFREE_NO_H|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|TTFREE_H1|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|SVDKEEP_NO_H|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|SVDKEEP_H1|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|SVDFREE_NO_H|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|SVDFREE_H1|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|TTKEEP_NO_H|T1G|8|31|31|0|[1.0, 1.0]|[1.0, 1.0]|
|TTKEEP_H1|T1G|8|31|27|0|[0.875, 0.875]|[0.8709677419354839, 0.8709677419354839]|
|TTFREE_NO_H|T1G|8|31|31|0|[1.0, 1.0]|[1.0, 1.0]|
|TTFREE_H1|T1G|8|31|31|0|[1.0, 1.0]|[1.0, 1.0]|
|SVDKEEP_NO_H|T1G|8|31|31|0|[1.0, 1.0]|[1.0, 1.0]|
|SVDKEEP_H1|T1G|8|31|31|0|[1.0, 1.0]|[1.0, 1.0]|
|SVDFREE_NO_H|T1G|8|31|31|0|[1.0, 1.0]|[1.0, 1.0]|
|SVDFREE_H1|T1G|8|31|29|0|[0.9375, 0.9375]|[0.9354838709677419, 0.9354838709677419]|
|TTKEEP_NO_H|T2G|8|27|25|0|[0.9270833333333334, 0.9270833333333334]|[0.9259259259259259, 0.9259259259259259]|
|TTKEEP_H1|T2G|8|27|17|0|[0.65625, 0.65625]|[0.6296296296296297, 0.6296296296296297]|
|TTFREE_NO_H|T2G|8|27|25|0|[0.9375, 0.9375]|[0.9259259259259259, 0.9259259259259259]|
|TTFREE_H1|T2G|8|27|24|0|[0.8958333333333334, 0.8958333333333334]|[0.8888888888888888, 0.8888888888888888]|
|SVDKEEP_NO_H|T2G|8|27|23|0|[0.8541666666666666, 0.8541666666666666]|[0.8518518518518519, 0.8518518518518519]|
|SVDKEEP_H1|T2G|8|27|21|0|[0.7916666666666666, 0.7916666666666666]|[0.7777777777777778, 0.7777777777777778]|
|SVDFREE_NO_H|T2G|8|27|24|0|[0.8958333333333334, 0.8958333333333334]|[0.8888888888888888, 0.8888888888888888]|
|SVDFREE_H1|T2G|8|27|25|0|[0.9375, 0.9375]|[0.9259259259259259, 0.9259259259259259]|
|TTKEEP_NO_H|T1L|0|0|0|0|None|None|
|TTKEEP_H1|T1L|0|0|0|0|None|None|
|TTFREE_NO_H|T1L|0|0|0|0|None|None|
|TTFREE_H1|T1L|0|0|0|0|None|None|
|SVDKEEP_NO_H|T1L|0|0|0|0|None|None|
|SVDKEEP_H1|T1L|0|0|0|0|None|None|
|SVDFREE_NO_H|T1L|0|0|0|0|None|None|
|SVDFREE_H1|T1L|0|0|0|0|None|None|
|TTKEEP_NO_H|T2L|0|0|0|0|None|None|
|TTKEEP_H1|T2L|0|0|0|0|None|None|
|TTFREE_NO_H|T2L|0|0|0|0|None|None|
|TTFREE_H1|T2L|0|0|0|0|None|None|
|SVDKEEP_NO_H|T2L|0|0|0|0|None|None|
|SVDKEEP_H1|T2L|0|0|0|0|None|None|
|SVDFREE_NO_H|T2L|0|0|0|0|None|None|
|SVDFREE_H1|T2L|0|0|0|0|None|None|
|TTKEEP_NO_H|H_fit|8|12|0|0|[0.0, 0.0]|[0.0, 0.0]|
|TTKEEP_H1|H_fit|8|12|4|0|[0.5, 0.5]|[0.3333333333333333, 0.3333333333333333]|
|TTFREE_NO_H|H_fit|8|12|0|0|[0.0, 0.0]|[0.0, 0.0]|
|TTFREE_H1|H_fit|8|12|6|0|[0.5833333333333334, 0.5833333333333334]|[0.5, 0.5]|
|SVDKEEP_NO_H|H_fit|8|12|0|0|[0.0, 0.0]|[0.0, 0.0]|
|SVDKEEP_H1|H_fit|8|12|6|0|[0.5833333333333334, 0.5833333333333334]|[0.5, 0.5]|
|SVDFREE_NO_H|H_fit|8|12|0|0|[0.0, 0.0]|[0.0, 0.0]|
|SVDFREE_H1|H_fit|8|12|6|0|[0.5833333333333334, 0.5833333333333334]|[0.5, 0.5]|

主比较与CI：[{"mask": "original", "task": "T2G", "comparison": "TTFREE_NO_H-TTKEEP_NO_H", "micro_delta_bounds": [0.0, 0.0], "edits": 8, "macro_delta_bounds": [0.010416666666666666, 0.010416666666666666], "macro_delta": 0.010416666666666666, "edit_ci95_envelope": [-0.09375, 0.125], "source_cluster_ci95_envelope": [-0.075, 0.14285714285714285], "source_groups": 5, "known_transitions": {"1_to_1": 23, "1_to_0": 2, "0_to_1": 2}, "unknown_pairs": 0, "known_net_correct": 0, "known_improved_edits": 2, "known_worsened_edits": 2}, {"mask": "original", "task": "T2G", "comparison": "SVDFREE_NO_H-SVDKEEP_NO_H", "micro_delta_bounds": [0.037037037037037035, 0.037037037037037035], "edits": 8, "macro_delta_bounds": [0.041666666666666664, 0.041666666666666664], "macro_delta": 0.041666666666666664, "edit_ci95_envelope": [0.0, 0.125], "source_cluster_ci95_envelope": [0.0, 0.16666666666666666], "source_groups": 5, "known_transitions": {"1_to_1": 23, "0_to_0": 3, "0_to_1": 1}, "unknown_pairs": 0, "known_net_correct": 1, "known_improved_edits": 1, "known_worsened_edits": 0}]

结论标签：DESCRIPTIVE_EXPOSED_H8_ONLY_NO_AUTOMATIC_PROMOTION；不自动更换主线或追加实验。Hfit仅训练诊断，H_eval NA，空locality不能证明安全。

- 8 H-supported edits previously exposed; not146 full-method evidence
- 12 H relationships from7 distinct source QA/images; Hfit not independent H_eval
- Original shared U and patient independence limitations unchanged
- Fixed per-edit seed; no multi-seed robustness claim
- Matching updates/tokens does not equal matching wall-time or optimization geometry
- SVD orthogonal and FREE rank4 have the same function family; difference is optimization coordinates, not compression or capacity
- TT/SVD native warmups are independent; only within-route KEEP/FREE share identical warm function
- Two primary contrasts, descriptive confidence intervals without multiple-testing correction
