# MDU-v2 第二骨干机制复现

27 个 Base 错题以普通 CE 各训练 160 步。读入分母为 Qwen 自身四条 FIT 的读入中位数。教师强制读取最后提示位，CORE 评分沿用原合同。

| 问法 | 事件 | CORE 正确 | 未知 | 读入比 | 整答 NLL | KL |
|---|---:|---:|---:|---:|---:|---:|
| native | 27 | 23 | 4 | 1.0342 | 0.01035 | 5.448 |
| fit | 108 | — | 0 | 1.0000 | 0.01328 | 5.261 |
| holdout_modal | 27 | 18 | 9 | 0.9059 | 0.03313 | 4.652 |
| holdout_context | 27 | 19 | 8 | 0.9742 | 0.01869 | 5.089 |
| independent | 27 | 11 | 4 | 0.8004 | 3.822 | 0.8463 |
| official_numeric_perturbation | 26 | 17 | 8 | 0.9591 | 0.04894 | 3.575 |
| official_negation_uncertainty | 27 | 21 | 4 | 0.9720 | 0.01922 | 4.351 |
| official_paraphrase | 27 | 21 | 6 | 1.0054 | 0.01803 | 4.641 |
| official_distractor_insertion | 27 | 19 | 8 | 0.9890 | 0.01993 | 4.779 |

留出正负例选择性间隔为 0.1353。官方 T2G 控制问法组的读入系数、图像聚类 bootstrap 及各分组残差见结果 JSON。开发数据已有第一骨干的实验曝光。

## 2026-11-28 冻结输出补评

只补评上一轮 45 个未知去重载荷，保留 140 个已有判定。45 个载荷全部返回有效判定，覆盖 51 个原未知事件。未重新生成或训练。评分沿用同一方法盲 CORE 合同与 `gpt-6-astra` medium，服务使用模型别名。

| 问法 | CORE | 补齐事件 |
|---|---:|---:|
| holdout_context | 27/27 | 8 |
| holdout_modal | 25/27 | 9 |
| independent | 15/27 | 4 |
| native | 27/27 | 4 |
| official_distractor_insertion | 27/27 | 8 |
| official_negation_uncertainty | 25/27 | 4 |
| official_numeric_perturbation | 24/26 | 8 |
| official_paraphrase | 27/27 | 6 |

官方四类 T2G 合计 103/107。原回归程序在完整评分上仍返回拟合不可用，读入关联没有得到可报告的回归估计。

[冻结输出补评聚合](MDU_V2_QWEN_CORE_SUPPLEMENT.json)
