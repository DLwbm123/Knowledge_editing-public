# TT 编辑的作用范围子空间

实现基于项目阶段 1 提交 **f4acb43e** 和作用范围子空间规格。当前提交供代码审阅。真实数据的 D0、阶段 1b、S0 以及后续 GPU 运行有各自的执行回执，合成测试与实验结果分别记录。

## 实现入口

| 文件 | 作用 |
|---|---|
| [features.py](features.py) | Base 前向采集最后提示位及全部答案位，RMS 归一化、FP16 缓存、角色与图像隔离检查 |
| [scope_solver.py](scope_solver.py) | 查询等权二阶矩、固定 G4 的广义特征解、带回退的 G4 更新、行正交化、共享背景逆与 K-U |
| [solve_panel.py](solve_panel.py) | 三种初态、两种成分消融、已有 MED／MDU／P4 比较与开发集选参 |
| [train_value.py](train_value.py) | 保持原问与四视图损失的 V-TT 和 V-TT-ft，沿 Base 续写测同图 KL |
| [router.py](router.py) | 规范不变读入打分、原型近似、按图像组的保形阈值 |
| [evaluate.py](evaluate.py) | D0、S0 和路由／写出评估，按图像配对并保存未知 |
| [experiment.py](experiment.py) | 阶段执行、状态绑定、评分包与聚合 |
| [make_panel.py](make_panel.py) | 从冻结阶段 1 输入和补充清单编译角色、训练顺序与视觉负例 |

## 运行依赖与角色清单

从仓库根目录执行，使用历史 LLaVA-Med 运行环境及原链路绑定的模型、数据和 expert hook。在实际 GPU 作业中，启动器以中性进程和 stdin 传递入口参数，每个进程仅暴露一张物理卡，GPU 侧设置独立截止时间和新写入上限。下列命令入口用来说明参数和依赖顺序。

- 私有补充清单含阶段 5 排除图像散列、开发图像散列、训练和留出句型列表、T1G 变换列表、校准图像 owner、β 验证 owner 和扩展事件。补充事件用于绑定官方间接组、独立于拟合的验证负例、答案保持已核验的图像正例、背景医学问答、T1G 及范围外查询。背景按约 1,000 个问答锁定，重复图像与阶段 5 图像均需排除。工程输入样例可由函数参数和测试查看。
- 每条事件包含 id、owner（背景及范围外为 null）、role、dataset、question、answer（如有参考）、source_sha256、image_sha256、image_path。复用的 Base token 为续写 token。输入文件中的 projection_parity 按事件 ID 给出五条冻结 MED 正例投影值。GPU 采集还会检验实际置零专家的完整 logits。
- 先用阶段 1 快照与补充清单运行 make_panel。视觉负例的全黑和强噪声进入训练，强模糊按图像分到校准或评测。在正式实验前确认模糊破坏视觉证据、正例变换保留答案。保存新增图像的实际散列、关系标签和构造器版本。
- GPU contract.json 含 outputs、p_weights、deadline、memory_bytes、max_forwards、max_generations。deadline 用 Unix 时间。运行环境复用历史 M3BENCH 模型、官方 LLaVA 源及图像根路径变量，并配置 CUBLAS_WORKSPACE_CONFIG。P1–P4 使用原阶段 1 的权重和身份绑定。

## 命令顺序

命令入口为 **experiments.med_scope_subspace_v1.experiment**。运行入口的 --help 可查看完整参数。训练和评估的 --shard / --shards 按可见的 GPU 进程分别传入。

1. **d0** 读取冻结 JSONL 和 owner 到图像组的映射，输出残差条件组效应和原尺度敏感性。**eval --policy stage1b --roles official_ind** 在冻结 R0 下生成 P1–P4 和 MDU 的官方间接题。两个步骤均可先行。
2. **collect --part development** 分片采集开发特征，另用 --references 保存 MED、MDU、P4 的绑定权重。**solve** 合并分片，输出带 manifest 摘要的锁定解。使用 --device 指定用于 TT 交替求解的单一设备。K-U 的共享背景分解在 CPU 执行，行空间小于原始维数时使用对应的 Woodbury 行空间分解。
3. **collect --part evaluation --solution** 再打开校准／留出特征。**s0** 用新特征比较锁定解，输出配对区间、解的轨迹、特征值和 S1 判定。官方间接题的读入另列。
4. **beta** 在通过 S0 后以开发验证负例和 P4 上限定公共系数。**train** 验证 S0 和 β 与同一锁定解的绑定，再运行 V-TT、V-TT-ft。**eval --policy writer** 固定使用查询所属专家，评估写出层的能力并与 P4 配对。
5. **eval --policy bank** 在完整 150 库上比较冻结原库 R0、相同 R0 更换写入器、原型门控与耦合自路由。--bank-arm 指定使用的写出臂。小库必须显式标为 --bank-pilot。阶段 1 的 P4 权重只有 30 位。全库在 S1 后还需冻结并准入其余编辑的数据和扩展训练合同，再传入 150 位 V-TT 权重。阈值在校准负例上按图像取最大分数，报告有效校准图像数、留出图像最坏激活和事件级激活。α 为 0.05 时，少于 19 个校准图像会得到无穷阈值。
6. **score-input** 从私有生成回执导出项目既有 CORE 评分器的输入，**summarize** 合并分片与授权评分记录。输出含按路由及类型分组的 CORE、未知、精确一致率、双前缀 KL 和 NLL，以及配对增损和区间。逐问输出、输入、分片身份及权重保持私有。

## 数值和协议说明

- G3 子问题有闭式精确解，G4 采用 Adam 上升并回退，完整 TT 问题是交替优化。回退同时还原参数和优化器状态。阈值按实际使用的 A 行空间重新计算，因此对 V-TT-ft 也保持基变换不变。
- 零残差初态处 $D_{\mathrm{KL}}(p_0\|p_\theta)$ 的一阶梯度为零。β 的量级在八步无 KL 开发热身后估计。候选比较和正式两臂仍各自从零残差开始，暖启动只用于梯度标尺。
- 路由和聚合实现支持分组误激活与双前缀局部性。保形保证依赖校准和未来图像组可交换，且负例采样与组内最大值的定义一致。域外集合的激活率为外部实测结果。
- D0 的区间跨零对应写出失配证据不足。它无法单独确认读入是唯一瓶颈。S0 低于准入时保留初始化比较结果，正式写出训练需要独立的后续方案。
- 首次实际试验还需绑定背景、验证负例、T1G 变换族及封存散列，并通过五条真实投影和真实模型零残差复核。CPU 合成单元测试位于 [tests/test_scope.py](tests/test_scope.py)。
