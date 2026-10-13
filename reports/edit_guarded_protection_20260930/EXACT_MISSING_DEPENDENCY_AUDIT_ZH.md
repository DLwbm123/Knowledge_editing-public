# PR8 精确缺失依赖审计

零训练、零模型前向、零新Judge；PR8原判定不变。同一缺失judge_key在任何方法或面板中均为同一z_j，系数先合并再计算精确上下界。

| 方法 | PR8原状态 | 本阶段精确门槛状态 |
|---|---|---|
|P_01|FAIL|FAIL|
|SP_01|INCONCLUSIVE|FAIL|
|L|FAIL|FAIL|
|G_0.75|FAIL|FAIL|

- P_01 / single/1/T2G/A0: INCONCLUSIVE → PASS
- P_01 / single/1/T2G/E_orig: INCONCLUSIVE → PASS
- P_01 / sequential/8/T2G/A0: INCONCLUSIVE → PASS
- P_01 / sequential/8/T2G/E_orig: INCONCLUSIVE → PASS
- P_01 / sequential/12/T2G/A0: INCONCLUSIVE → PASS
- P_01 / sequential/12/T2G/E_orig: INCONCLUSIVE → PASS
- P_01 / sequential/24/T2G/A0: INCONCLUSIVE → PASS
- P_01 / sequential/24/T2G/E_orig: INCONCLUSIVE → PASS
- SP_01 / single/1/T2G/A0: INCONCLUSIVE → FAIL
- SP_01 / single/1/T2G/E_orig: INCONCLUSIVE → FAIL
- SP_01 / single/1/generality/H: INCONCLUSIVE → PASS
- SP_01 / sequential/8/T2G/A0: INCONCLUSIVE → FAIL
- SP_01 / sequential/8/T2G/E_orig: INCONCLUSIVE → FAIL
- SP_01 / sequential/8/generality/H: INCONCLUSIVE → PASS
- SP_01 / sequential/12/T2G/A0: INCONCLUSIVE → FAIL
- SP_01 / sequential/12/T2G/E_orig: INCONCLUSIVE → FAIL
- SP_01 / sequential/12/generality/H: INCONCLUSIVE → PASS
- SP_01 / sequential/24/T2G/A0: INCONCLUSIVE → FAIL
- SP_01 / sequential/24/T2G/E_orig: INCONCLUSIVE → FAIL
- SP_01 / sequential/24/generality/H: INCONCLUSIVE → PASS
- L / single/1/generality/H: INCONCLUSIVE → FAIL
- L / sequential/8/generality/H: INCONCLUSIVE → FAIL
- L / sequential/12/generality/H: INCONCLUSIVE → FAIL
- L / sequential/24/generality/H: INCONCLUSIVE → FAIL

这些是同一批暴露DEV结果的精确重算，未修复或重判永久缺失，不是新的模型效果。各门槛的精确界不意味着可以把不同最优赋值拼成一套模型；joint PASS仍要求所有门槛都对共享变量安全。
