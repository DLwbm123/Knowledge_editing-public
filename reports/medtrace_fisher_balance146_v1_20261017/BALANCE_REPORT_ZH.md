# Fisher梯度平衡单候选：完整有限实验

联合门槛：{'complete_no_missing': True, 'FIT': True, 'T2L': True, 'T2G_improves': True, 'damage': True, 'passed': True}

| 方法 | FIT | T2G | T2L | damage / Base正确 |
|---|---:|---:|---:|---:|
| C | 100.0 | 82.07762557077625 | 87.09677419354838 | 9 / 85 |
| M1 | 100.0 | 84.13242009132419 | 82.7956989247312 | 12 / 85 |
| M3 | 100.0 | 62.100456621004575 | 97.31182795698926 | 4 / 85 |
| M13 | 100.0 | 61.3013698630137 | 98.38709677419355 | 3 / 85 |
| M13B | 100.0 | 73.91552511415526 | 100.0 | 2 / 85 |

T2G配对差与区间：{"M13B_minus_M13_macro_pp": 12.614155251141552, "edit_CI_pp": [9.30365296803653, 16.095890410958905], "source_CI_pp": [9.317129629629628, 16.19385342789598], "source_unit": "Original source groups; not independently verified patients", "M13B_preserves_each_M13_owner": false}

Source机制诊断：{"baseline_endpoint_owners": 146, "baseline_cap_would_activate": 146, "baseline_source_conflict_groups": 10, "baseline_median_scale": 0.5475800357456524, "new_steps": 23360, "new_cap_active_steps": 13686, "new_median_scale": 0.18072060211829866, "limitation": "Endpoint diagnostics are source-only, not direct T2G gradients or proof of causal mediation; no finite-step locality guarantee"}

已暴露DEV；T2L只有31支持编辑/54观测。不把拟合成功、source梯度性质或T2G正号单独解释为联合成功；无独立确认或医学安全结论。
