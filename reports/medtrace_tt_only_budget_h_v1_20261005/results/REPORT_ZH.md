# TT-only预算与H更新结果

96 TT暖启动，240条续训，8编辑×3seed。每edit先对seed平均后bootstrap；不是24独立病例。全部部署专家仅TT核。

|方式|任务|编辑|seed重复分母|正确|missing|宏界|微界|
|---|---|---:|---:|---:|---:|---|---|
|TT44_NO_H|T0|8|24|24|0|[1.0, 1.0]|[1.0, 1.0]|
|TT44_H1|T0|8|24|21|0|[0.875, 0.875]|[0.875, 0.875]|
|TT88_NO_H|T0|8|24|24|0|[1.0, 1.0]|[1.0, 1.0]|
|TT88_H1|T0|8|24|20|0|[0.8333333333333334, 0.8333333333333334]|[0.8333333333333334, 0.8333333333333334]|
|TT84_NO_H|T0|8|24|24|0|[1.0, 1.0]|[1.0, 1.0]|
|TT84_H1|T0|8|24|20|0|[0.8333333333333334, 0.8333333333333334]|[0.8333333333333334, 0.8333333333333334]|
|TT48_NO_H|T0|8|24|24|0|[1.0, 1.0]|[1.0, 1.0]|
|TT48_H1|T0|8|24|23|0|[0.9583333333333334, 0.9583333333333334]|[0.9583333333333334, 0.9583333333333334]|
|TT88_SMALL_LR_H|T0|8|24|16|0|[0.6666666666666666, 0.6666666666666666]|[0.6666666666666666, 0.6666666666666666]|
|TT88_GUARDED_H|T0|8|24|24|0|[1.0, 1.0]|[1.0, 1.0]|
|TT44_NO_H|T1G|8|93|93|0|[1.0, 1.0]|[1.0, 1.0]|
|TT44_H1|T1G|8|93|81|0|[0.875, 0.875]|[0.8709677419354839, 0.8709677419354839]|
|TT88_NO_H|T1G|8|93|93|0|[1.0, 1.0]|[1.0, 1.0]|
|TT88_H1|T1G|8|93|75|0|[0.8125, 0.8125]|[0.8064516129032258, 0.8064516129032258]|
|TT84_NO_H|T1G|8|93|93|0|[1.0, 1.0]|[1.0, 1.0]|
|TT84_H1|T1G|8|93|77|0|[0.8333333333333334, 0.8333333333333334]|[0.8279569892473119, 0.8279569892473119]|
|TT48_NO_H|T1G|8|93|93|0|[1.0, 1.0]|[1.0, 1.0]|
|TT48_H1|T1G|8|93|89|0|[0.9583333333333334, 0.9583333333333334]|[0.956989247311828, 0.956989247311828]|
|TT88_SMALL_LR_H|T1G|8|93|63|0|[0.6666666666666666, 0.6666666666666666]|[0.6774193548387096, 0.6774193548387096]|
|TT88_GUARDED_H|T1G|8|93|93|0|[1.0, 1.0]|[1.0, 1.0]|
|TT44_NO_H|T2G|8|81|75|0|[0.9305555555555556, 0.9305555555555556]|[0.9259259259259259, 0.9259259259259259]|
|TT44_H1|T2G|8|81|51|0|[0.6597222222222222, 0.6597222222222222]|[0.6296296296296297, 0.6296296296296297]|
|TT88_NO_H|T2G|8|81|75|0|[0.9305555555555556, 0.9305555555555556]|[0.9259259259259259, 0.9259259259259259]|
|TT88_H1|T2G|8|81|51|0|[0.6701388888888888, 0.6701388888888888]|[0.6296296296296297, 0.6296296296296297]|
|TT84_NO_H|T2G|8|81|80|0|[0.9895833333333334, 0.9895833333333334]|[0.9876543209876543, 0.9876543209876543]|
|TT84_H1|T2G|8|81|55|0|[0.71875, 0.71875]|[0.6790123456790124, 0.6790123456790124]|
|TT48_NO_H|T2G|8|81|74|0|[0.9201388888888888, 0.9201388888888888]|[0.9135802469135802, 0.9135802469135802]|
|TT48_H1|T2G|8|81|56|0|[0.7118055555555556, 0.7118055555555556]|[0.691358024691358, 0.691358024691358]|
|TT88_SMALL_LR_H|T2G|8|81|48|0|[0.6458333333333334, 0.6458333333333334]|[0.5925925925925926, 0.5925925925925926]|
|TT88_GUARDED_H|T2G|8|81|78|0|[0.9548611111111112, 0.9548611111111112]|[0.9629629629629629, 0.9629629629629629]|
|TT44_NO_H|T1L|0|0|0|0|None|None|
|TT44_H1|T1L|0|0|0|0|None|None|
|TT88_NO_H|T1L|0|0|0|0|None|None|
|TT88_H1|T1L|0|0|0|0|None|None|
|TT84_NO_H|T1L|0|0|0|0|None|None|
|TT84_H1|T1L|0|0|0|0|None|None|
|TT48_NO_H|T1L|0|0|0|0|None|None|
|TT48_H1|T1L|0|0|0|0|None|None|
|TT88_SMALL_LR_H|T1L|0|0|0|0|None|None|
|TT88_GUARDED_H|T1L|0|0|0|0|None|None|
|TT44_NO_H|T2L|0|0|0|0|None|None|
|TT44_H1|T2L|0|0|0|0|None|None|
|TT88_NO_H|T2L|0|0|0|0|None|None|
|TT88_H1|T2L|0|0|0|0|None|None|
|TT84_NO_H|T2L|0|0|0|0|None|None|
|TT84_H1|T2L|0|0|0|0|None|None|
|TT48_NO_H|T2L|0|0|0|0|None|None|
|TT48_H1|T2L|0|0|0|0|None|None|
|TT88_SMALL_LR_H|T2L|0|0|0|0|None|None|
|TT88_GUARDED_H|T2L|0|0|0|0|None|None|
|TT44_NO_H|H_fit|8|36|0|0|[0.0, 0.0]|[0.0, 0.0]|
|TT44_H1|H_fit|8|36|12|0|[0.5, 0.5]|[0.3333333333333333, 0.3333333333333333]|
|TT88_NO_H|H_fit|8|36|0|0|[0.0, 0.0]|[0.0, 0.0]|
|TT88_H1|H_fit|8|36|12|0|[0.5, 0.5]|[0.3333333333333333, 0.3333333333333333]|
|TT84_NO_H|H_fit|8|36|0|0|[0.0, 0.0]|[0.0, 0.0]|
|TT84_H1|H_fit|8|36|12|0|[0.5, 0.5]|[0.3333333333333333, 0.3333333333333333]|
|TT48_NO_H|H_fit|8|36|0|0|[0.0, 0.0]|[0.0, 0.0]|
|TT48_H1|H_fit|8|36|13|0|[0.5138888888888888, 0.5138888888888888]|[0.3611111111111111, 0.3611111111111111]|
|TT88_SMALL_LR_H|H_fit|8|36|12|0|[0.5, 0.5]|[0.3333333333333333, 0.3333333333333333]|
|TT88_GUARDED_H|H_fit|8|36|2|0|[0.08333333333333333, 0.08333333333333333]|[0.05555555555555555, 0.05555555555555555]|

主对比与CI：[{"mask": "original", "task": "T2G", "comparison": "TT88_H1-TT44_H1", "seed_slot": "mean", "primary": true, "micro_delta_bounds": [0.0, 0.0], "edits": 8, "macro_delta_bounds": [0.010416666666666666, 0.010416666666666666], "macro_delta": 0.010416666666666666, "edit_ci95_envelope": [-0.1875, 0.21180555555555555], "source_cluster_ci95_envelope": [-0.08641975308641976, 0.17129629629629628], "source_groups": 5, "known_transitions": {"1_to_1": 38, "0_to_0": 17, "1_to_0": 13, "0_to_1": 13}, "unknown_pairs": 0, "known_net_correct": 0, "known_improved_edits": 5, "known_worsened_edits": 5}, {"mask": "original", "task": "T2G", "comparison": "TT88_GUARDED_H-TT88_H1", "seed_slot": "mean", "primary": true, "micro_delta_bounds": [0.3333333333333333, 0.3333333333333333], "edits": 8, "macro_delta_bounds": [0.2847222222222222, 0.2847222222222222], "macro_delta": 0.2847222222222222, "edit_ci95_envelope": [0.05902777777777777, 0.5104166666666666], "source_cluster_ci95_envelope": [-0.03968253968253969, 0.537037037037037], "source_groups": 5, "known_transitions": {"1_to_1": 49, "0_to_1": 29, "1_to_0": 2, "0_to_0": 1}, "unknown_pairs": 0, "known_net_correct": 27, "known_improved_edits": 6, "known_worsened_edits": 2}]

保护更新与小学习率的机制表：[{"arm": "TT44_NO_H_s0", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 22.760338665947533, "cumulative_function_displacement": 3453.3047472834587, "extra_guard_forwards": 0}, {"arm": "TT44_H1_s0", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 25.19282019116284, "cumulative_function_displacement": 4165.539890617132, "extra_guard_forwards": 0}, {"arm": "TT88_NO_H_s0", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 31.706757840063602, "cumulative_function_displacement": 6678.546172738075, "extra_guard_forwards": 0}, {"arm": "TT88_H1_s0", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 32.56160125299181, "cumulative_function_displacement": 7340.414636656642, "extra_guard_forwards": 0}, {"arm": "TT84_NO_H_s0", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 35.869195001778785, "cumulative_function_displacement": 6324.964559674263, "extra_guard_forwards": 0}, {"arm": "TT84_H1_s0", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 35.96762707861149, "cumulative_function_displacement": 6769.5779566168785, "extra_guard_forwards": 0}, {"arm": "TT48_NO_H_s0", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 21.767648762225804, "cumulative_function_displacement": 3536.4388157725334, "extra_guard_forwards": 0}, {"arm": "TT48_H1_s0", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 24.99196189190965, "cumulative_function_displacement": 4533.71147608757, "extra_guard_forwards": 0}, {"arm": "TT88_SMALL_LR_H_s0", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 18.07655032387517, "cumulative_function_displacement": 4219.4460411816835, "extra_guard_forwards": 0}, {"arm": "TT88_GUARDED_H_s0", "outer_attempts": 2560, "accepted": 173, "alpha_counts": {"1.0": 161, "0.5": 4, "0.25": 3, "None": 2387, "0.125": 5}, "failed_constraint_index_counts": {"0": 5744, "3": 5896, "1": 5710, "2": 4182, "4": 4531}, "cumulative_parameter_displacement": 6.953759577016403, "cumulative_function_displacement": 1804.7087886333466, "extra_guard_forwards": 58476}, {"arm": "TT44_NO_H_s1", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 26.48757512969116, "cumulative_function_displacement": 3612.710297971964, "extra_guard_forwards": 0}, {"arm": "TT44_H1_s1", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 24.68340557333112, "cumulative_function_displacement": 3966.9338499978185, "extra_guard_forwards": 0}, {"arm": "TT88_NO_H_s1", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 30.27366562592624, "cumulative_function_displacement": 7003.352720975876, "extra_guard_forwards": 0}, {"arm": "TT88_H1_s1", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 32.1617979235156, "cumulative_function_displacement": 7683.306714639068, "extra_guard_forwards": 0}, {"arm": "TT84_NO_H_s1", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 36.148709937633555, "cumulative_function_displacement": 5480.618774801493, "extra_guard_forwards": 0}, {"arm": "TT84_H1_s1", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 33.00380123661521, "cumulative_function_displacement": 5605.818836070597, "extra_guard_forwards": 0}, {"arm": "TT48_NO_H_s1", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 24.937035016816417, "cumulative_function_displacement": 4670.49411842227, "extra_guard_forwards": 0}, {"arm": "TT48_H1_s1", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 23.222242620581227, "cumulative_function_displacement": 4913.0879859775305, "extra_guard_forwards": 0}, {"arm": "TT88_SMALL_LR_H_s1", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 18.389618714030536, "cumulative_function_displacement": 4503.507185064256, "extra_guard_forwards": 0}, {"arm": "TT88_GUARDED_H_s1", "outer_attempts": 2560, "accepted": 240, "alpha_counts": {"1.0": 223, "0.25": 5, "0.125": 9, "None": 2320, "0.5": 3}, "failed_constraint_index_counts": {"3": 6753, "0": 6032, "1": 4962, "2": 4066, "4": 4439}, "cumulative_parameter_displacement": 9.602641156288005, "cumulative_function_displacement": 2483.105664551258, "extra_guard_forwards": 57360}, {"arm": "TT44_NO_H_s2", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 25.719426563467497, "cumulative_function_displacement": 3957.909140229225, "extra_guard_forwards": 0}, {"arm": "TT44_H1_s2", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 24.476679252771792, "cumulative_function_displacement": 4095.2540646120906, "extra_guard_forwards": 0}, {"arm": "TT88_NO_H_s2", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 33.04082727528678, "cumulative_function_displacement": 7104.486085474491, "extra_guard_forwards": 0}, {"arm": "TT88_H1_s2", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 34.22834433078619, "cumulative_function_displacement": 7943.0439270585775, "extra_guard_forwards": 0}, {"arm": "TT84_NO_H_s2", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 33.20864487109528, "cumulative_function_displacement": 6096.48883163929, "extra_guard_forwards": 0}, {"arm": "TT84_H1_s2", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 35.07960164903324, "cumulative_function_displacement": 6692.268271155655, "extra_guard_forwards": 0}, {"arm": "TT48_NO_H_s2", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 26.466630414801493, "cumulative_function_displacement": 4665.363259136677, "extra_guard_forwards": 0}, {"arm": "TT48_H1_s2", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 26.65594098672075, "cumulative_function_displacement": 5342.919160582125, "extra_guard_forwards": 0}, {"arm": "TT88_SMALL_LR_H_s2", "outer_attempts": 2560, "accepted": 2560, "alpha_counts": {"unguarded": 2560}, "failed_constraint_index_counts": {}, "cumulative_parameter_displacement": 18.658476438809252, "cumulative_function_displacement": 4515.4027625620365, "extra_guard_forwards": 0}, {"arm": "TT88_GUARDED_H_s2", "outer_attempts": 2560, "accepted": 230, "alpha_counts": {"1.0": 214, "0.5": 6, "None": 2330, "0.25": 4, "0.125": 6}, "failed_constraint_index_counts": {"3": 6955, "2": 6085, "0": 5745, "1": 4398, "4": 4136}, "cumulative_parameter_displacement": 9.309797568419674, "cumulative_function_displacement": 2372.562255203724, "extra_guard_forwards": 57492}]

H_fit仅训练诊断。没有独立H_eval/locality支持，完整方法NOT_CONFIRMED。保护与小学习率相近时不能排除更小步长解释；高拒绝且不学习H不是成功。不自动选主线、扩秩或追加146。

- 8 exposed development edits;3seeds are optimization repeats not24 independent cases
- 12Hrelations from7QA/images; Hfit diagnosis only,H_evalNA
- H8 locality eligible denominator0=NA; patientUNKNOWN; sharedU not independentCAL
- rank4 middle remainsfixed; structures plus fixed optimizer differ,not purecapacity;84vs48 unequalparams
- smallLR not parameter-displacement matched;guard benefits may be smallerupdates or learningstall
- multiple secondary comparisons descriptive;CI containing0 not equivalence
- no146bank;future146 must allTT,not H8hybrid
