# removed软KL与实际硬约束路径首源20槽对照

PARTIAL_SCIENTIFIC_STOP。SH11只读复用R；其他三臂独立从Base开始。缺分不补判，未知不通过逐题零损伤。

|臂|任务|正确/分母|未知|微平均界%|
|---|---|---:|---:|---|
|SH11|T0|0/1|1|[0.0, 100.0]|
|SH11|T1G|0/4|2|[0.0, 50.0]|
|SH11|T2G|0/4|2|[0.0, 50.0]|
|SH11|T1L|0/0|0|None|
|SH11|T2L|0/0|0|None|
|SH00|T0|1/1|0|[100.0, 100.0]|
|SH00|T1G|4/4|0|[100.0, 100.0]|
|SH00|T2G|3/4|0|[75.0, 75.0]|
|SH00|T1L|0/0|0|None|
|SH00|T2L|0/0|0|None|
|SH10|T0|1/1|0|[100.0, 100.0]|
|SH10|T1G|4/4|0|[100.0, 100.0]|
|SH10|T2G|2/4|0|[50.0, 50.0]|
|SH10|T1L|0/0|0|None|
|SH10|T2L|0/0|0|None|
|SH01|T0|0/1|1|[0.0, 100.0]|
|SH01|T1G|0/4|2|[0.0, 50.0]|
|SH01|T2G|0/4|2|[0.0, 50.0]|
|SH01|T1L|0/0|0|None|
|SH01|T2L|0/0|0|None|

主机械指标为共同的完整native+allFIT CE。各臂CE/接受拒绝与零比例：{"SH11": {"final_complete_CE": 2.3723029944461094, "last_available_CE": 2.3723029944461094, "slots": 20, "accepted": 13, "rejected": 7, "zero_fraction": 0.35, "first_hit_CE": "NA_HISTORICAL_NOT_MEASURED"}, "SH00": {"final_complete_CE": 0.01851115074300988, "last_available_CE": 0.01851115074300988, "slots": 20, "accepted": 20, "rejected": 0, "zero_fraction": 0.0, "first_hit_CE": {"1": 6, "0.5": 9, "0.1": 13}}, "SH10": {"final_complete_CE": 0.5192088006322929, "last_available_CE": 0.5192088006322929, "slots": 20, "accepted": 20, "rejected": 0, "zero_fraction": 0.0, "first_hit_CE": {"1": 7, "0.5": 17, "0.1": null}}, "SH01": {"final_complete_CE": 2.375626972471239, "last_available_CE": 2.375626972471239, "slots": 20, "accepted": 10, "rejected": 10, "zero_fraction": 0.5, "first_hit_CE": {"1": null, "0.5": null, "0.1": null}}}

四个条件效应及交互：{"SH10-SH00": {"final_complete_CE": 0.500697649889283, "accepted": 0, "zero_fraction": 0.0}, "SH11-SH01": {"final_complete_CE": -0.003323978025129737, "accepted": 3, "zero_fraction": -0.15000000000000002}, "SH01-SH00": {"final_complete_CE": 2.3571158217282293, "accepted": -10, "zero_fraction": 0.5}, "SH11-SH10": {"final_complete_CE": 1.8530941938138166, "accepted": -7, "zero_fraction": 0.35}, "SH11-SH10-SH01+SH00": {"final_complete_CE": -0.5040216279144127, "accepted": 3, "zero_fraction": -0.15000000000000002}}

T2G共享缺分交互：{"comparison": "SH11-SH10-SH01+SH00", "delta_macro_bounds_pp": [25.0, 25.0], "edit_ci": [25.0, 25.0], "source_ci": [25.0, 25.0], "source_groups": 1, "interpretation": "Development uncertainty only; missing scores bounded, not dropped"}。逐题保护及相对损伤/修复、真实插入端点与固定34T2L+2T1L见RESULTS.json。所有界先合并同一payload变量，未假设缺分独立。单来源10000次配对重采样不是总体推广证据。

H0仍有Base16/removed4 Fisher保护，但不执行五视图真实KL方向和硬门，明确HARD_KL_NOT_ENFORCED。H1保留原阈值.01和R求解器。H是整个实际约束路径因子，不能只归因末端硬门。新信息投影目标未测试；有限Fisher、粗框及非线性误差不构成临床或隐私保证。原146/M暂停，O/L等待，不自动扩展。

实际调用成本：{"forwards": 13648, "vision_forwards": 336, "backwards": 223, "committed_updates": 53, "attempt_slots": 60, "generation_attempts": 276, "active_set_checks": 0, "candidate_checks": 191}；GPU进程小时=0.4817956217130025；新Judge=30。科学止步、永久缺分与未生成分开记录。
