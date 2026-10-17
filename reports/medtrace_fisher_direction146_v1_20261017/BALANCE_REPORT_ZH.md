# M13P：完整有限实验

联合门槛：{'complete_no_missing': True, 'FIT': True, 'T2L': True, 'T2G_improves': True, 'damage': True, 'passed': True}

| 方法 | FIT | T2G | T2L | damage / Base正确 |
|---|---:|---:|---:|---:|
| C | 100.0 | 82.07762557077625 | 87.09677419354838 | 9 / 85 |
| M1 | 100.0 | 84.13242009132419 | 82.7956989247312 | 12 / 85 |
| M3 | 100.0 | 62.100456621004575 | 97.31182795698926 | 4 / 85 |
| M13 | 100.0 | 61.3013698630137 | 98.38709677419355 | 3 / 85 |
| M13B | 100.0 | 73.91552511415526 | 100.0 | 2 / 85 |
| M13P | 100.0 | 76.48401826484019 | 100.0 | 2 / 85 |

T2G配对差与区间：{"M13P_minus_M13B_macro_pp": 2.5684931506849313, "edit_CI_pp": [0.7990867579908676, 4.394977168949772], "source_CI_pp": [0.7270693512304252, 4.432624113475177], "source_unit": "Original source groups; not independently verified patients", "M13P_preserves_each_M13B_owner": false}

Source机制诊断：{"parent_trajectory_summary": {"group_steps": 46574, "negative_Fisher_source_cosine": 46123, "cosine_nonzero": 46428, "median_cosine": -0.3248298537162111, "median_source_dot_retention": 0.8712424344373542, "below_half_source_dot": 1330, "below_tenth_source_dot": 0, "early_median": 0.9623174025483979, "late_median": 0.7700063111358271, "owners_median_retention_below_half": 1}, "new_steps": 23360, "projected_groups": 46043, "minimum_source_dot_ratio_after": 0.9999999939172554, "zero_directions": 0, "limitation": "Source-only pre-clipping and pre-matching geometry; not finite-step generalization or locality guarantee"}

已暴露DEV；T2L只有31支持编辑/54观测。不把拟合成功、source梯度性质或T2G正号单独解释为联合成功；无独立确认或医学安全结论。

T1L补充：全部六臂100%，仅2条观测/1个支持编辑，无缺失；T1G全部100%，578条/146编辑。T1L不作为稳健保护证据。
