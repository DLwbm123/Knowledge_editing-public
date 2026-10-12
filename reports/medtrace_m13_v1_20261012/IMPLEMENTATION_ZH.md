# 实现边界

复用PR67已验证的训练、数学、FP32 runtime、实际组合算子步长匹配、Fisher、生成、原R0路由与评分隔离。新增m13.py仅组织M13-only有限链，m13_queue.py沿用完整payload身份复用，m13_report.py沿用全部角色/逐owner/原63双聚类区间并加入C/M1/M3/E四比较和兼顾门槛，m13_check.py核对配置与真实7168参数CPU数值。没有新依赖。

共享module_run仅新增M13的旧系数一致性检查，并让evaluate/route冻结候选数量作为显式参数；旧默认48保持，M13传8。冻结记录始终如实记8个新增候选，不把8伪装成48。Fisher重建是实际新增成本，C保留状态和参考输出只读复用。

CPU配置、参考来源、M1坐标/标量状态、实际算子Fisher值及梯度、真实TT FP32检查通过；M13 flags明确true/false/true，M2不会调用。CPU检查无VLM前向或Judge。GPU中首个Fisher pullback、重建系数对原冻结值、M13每步canonical函数范数、8个160步保存加载、原路由219一致性和完整消费者计数仍需实际屏障通过。尚未运行或声称M13收益。
