# CP初始化配对消融结果

原146登记中的固定H8，32续训，非146全库。CP和DIRECT两路线匹配训练更新/样本，不匹配优化坐标或墙钟。

|臂|任务|编辑|分母|已知正确|missing|宏界|微界|
|---|---|---:|---:|---:|---:|---|---|
|CP_NO_H|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|CP_H1|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|DIRECT_NO_H|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|DIRECT_H1|T0|8|8|8|0|[1.0, 1.0]|[1.0, 1.0]|
|CP_NO_H|T1G|8|31|31|0|[1.0, 1.0]|[1.0, 1.0]|
|CP_H1|T1G|8|31|31|0|[1.0, 1.0]|[1.0, 1.0]|
|DIRECT_NO_H|T1G|8|31|31|0|[1.0, 1.0]|[1.0, 1.0]|
|DIRECT_H1|T1G|8|31|31|0|[1.0, 1.0]|[1.0, 1.0]|
|CP_NO_H|T2G|8|27|26|0|[0.96875, 0.96875]|[0.9629629629629629, 0.9629629629629629]|
|CP_H1|T2G|8|27|23|0|[0.8645833333333334, 0.8645833333333334]|[0.8518518518518519, 0.8518518518518519]|
|DIRECT_NO_H|T2G|8|27|22|0|[0.8333333333333334, 0.8333333333333334]|[0.8148148148148148, 0.8148148148148148]|
|DIRECT_H1|T2G|8|27|24|0|[0.8958333333333334, 0.8958333333333334]|[0.8888888888888888, 0.8888888888888888]|
|CP_NO_H|T1L|0|0|0|0|None|None|
|CP_H1|T1L|0|0|0|0|None|None|
|DIRECT_NO_H|T1L|0|0|0|0|None|None|
|DIRECT_H1|T1L|0|0|0|0|None|None|
|CP_NO_H|T2L|0|0|0|0|None|None|
|CP_H1|T2L|0|0|0|0|None|None|
|DIRECT_NO_H|T2L|0|0|0|0|None|None|
|DIRECT_H1|T2L|0|0|0|0|None|None|
|CP_NO_H|H_fit|8|12|0|0|[0.0, 0.0]|[0.0, 0.0]|
|CP_H1|H_fit|8|12|6|0|[0.5833333333333334, 0.5833333333333334]|[0.5, 0.5]|
|DIRECT_NO_H|H_fit|8|12|0|0|[0.0, 0.0]|[0.0, 0.0]|
|DIRECT_H1|H_fit|8|12|6|0|[0.5833333333333334, 0.5833333333333334]|[0.5, 0.5]|

主比较与CI：{"mask": "original", "task": "T2G", "comparison": "CP_NO_H-DIRECT_NO_H", "micro_delta_bounds": [0.14814814814814814, 0.14814814814814814], "edits": 8, "macro_delta_bounds": [0.13541666666666666, 0.13541666666666666], "macro_delta": 0.13541666666666666, "edit_ci95_envelope": [0.03125, 0.22916666666666666], "source_cluster_ci95_envelope": [0.03125, 0.25], "source_groups": 5, "known_transitions": {"1_to_1": 22, "0_to_0": 1, "0_to_1": 4}, "unknown_pairs": 0, "known_net_correct": 4, "known_improved_edits": 4, "known_worsened_edits": 0}

结论标签：SUPPORTED_ON_EXPOSED_H8_ONLY；不自动更换主线或追加实验。Hfit仅训练诊断，H_eval NA，空locality不能证明安全。

- 8 H-supported edits previously exposed; not146 full-method evidence
- 12 H relationships from7 distinct source QA/images; Hfit not independent H_eval
- Original shared U and patient independence limitations unchanged
- Fixed per-edit seed; no multi-seed robustness claim
- Matching updates/tokens does not equal matching wall-time or optimization geometry
