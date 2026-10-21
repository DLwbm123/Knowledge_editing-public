# 独立FT-M启发教师的源内资格 TQ3（运行前冻结）

目的：检验从W0独立拟合的参数教师能否在不提供答案参考的原问题+图像条件下完整回答源问题。只做教师资格，不进行学生蒸馏，不动旧H/F/N轨迹，不声称完整AlignEdit复现或性能提升。

固定E01/E02/E03各一个教师、按原顺序独立从W0重建，不按旧教师资格删E02。对照为已缓存原Base完整源输出和前轮P资格2/3；只引用同源已有评分，不重新生成Base/P/R。不得读评估QA或locality作训练目标/选择教师。

教师仅model.layers.6.mlp.down_proj.weight可写；layer6/25步/5e-4借鉴论文Table10，单矩阵、医疗多模态原chat模板与原源full平均因果NLL为明确适配。Adam(lr=.0005,betas=.9/.999,eps1e-8,weight_decay0)，不梯度裁剪、不坐标截断，无新候选/层/学习率扫描。每步loss<.01则停止且不更新，否则Adam一次，最多25次；不按生成/Judge挑步。选择矩阵与Adam状态FP32、其余骨干原生精度，最后写回FP16、一次完整生成前核有限native loss。这不是全模型FP32，不把FP32训练loss当native生成表现。教师拟合不使用Pure-W的null/F0/trust门；它是隔离的教师构建，不能把无约束教师称最终受保护写入结果。每例生成后恢复W0和原native损失/dtype/flags/modes/hooks/grad/config，再处理下一例。仅一个教师在内存，不存teacher checkpoint。

每例最后教师原问题+图像一次官方生成，不输入参考/草稿；原max_new_tokens1024不降低，每请求128forward，最多3完整输出。一次源语义agreement和一次严格source-faithfulness资格，原revision3 rubric/Astra medium、每criterion最多3payload一批、每条一次/600秒/工具0隔离；总新payload≤6，精确同question+reference+完整输出且同criterion的旧缓存可复用，冲突停不挑分。完整分母3、缺失NA和上下界保留，两个criterion都true才资格；失败不重新拟合/改prompt重采。不能把teacher资格当T0学生性能、不能据此启动146。先完成教师证据公开，再决定是否必要独立冻结学生对照。

GPU总600forward、工作560，全部loss/原基座复核/生成/恢复计入，最多75backward；GPU从首次CPU检查起≤900秒（工作840、独立guard870 TERM/900 KILL、PID出生tick绑定/同进程组），恢复预留40forward。整个独立实验首次CPU实际调用起4小时，含评分和失败，不重置。内部每服务器新增256MiB：root写入上限240MiB、临时/评分/源码复核/日志/余量16MiB，不下载模型/数据、不存checkpoint、不删历史。首次root不存在基线0，预计实际<10MiB，独立storage gate与过程统计。每服务器实际新增仍必须严格<50,000,000,000字节、总原始历时≤24h。

pro5000固定物理GPU2/既有UUID，≥48GiB可用余量，既有环境/模型/官方Llava只读。新root单次安装，CPU合成实际fit/Adam因果损失下降、native提交、flags/hooks/grad/异常回滚检查全过才GPU；存储挂载/余量/写读探针通过。任何绑定/数值/恢复/CPU/GPU预算异常留现场终止、不重试；不完整输出不评分、不补生成，其他未执行保留NA。独立进程组与中性主子argv，启动后ps+nvidia-smi核。唯一pure-w静默半小时跟进，无子代理。

代码先经7897代理公开再部署；运行终态匿名完整代码/数值轨迹/资格/失败/缺失/成本/中文报告公开并核远端SHA/匿名HTTP200。QA、答案、token、权重、私有路径及第三方源码/PDF不发布。上游EasyEdit只记录链接和审阅结论，不拷贝发布。
