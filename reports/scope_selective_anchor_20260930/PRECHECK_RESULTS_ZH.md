# 选择性正例锚定：预检与机械验证

当前状态：WAITING_FOR_JUDGE_RESERVATION。正式 pilot、全 DEV、REG 均未启动；没有新的泛化或压力效果结论。此文件是本轮预检回执，不是最终实验报告。

独立工作区为 Knowledge_editing-scope-selective-anchor-20260930，分支 experiment/scope-selective-anchor-20260930，起点 e6270d4。前轮原件、权重与报告未修改；未恢复小时监测。

## 已完成

- 48 个共同初态均使用保留 E_orig 载体，与原 M1 INIT_POST80 张量逐项相等。初始化来源、实际 native/S/U/G 支持绑定、代码和权重校验通过。
- 144 份历史 A0/AH/AHS 对照通过绑定审计。代表编辑 4 的 AH 与 AHS 均通过新路径真实模型 80 步最终张量完全一致检查。旧 pilot 的 8 份低 λ 权重已删除，本轮按补做计费，不宣称复用。
- 主 seed 固定 20260929，沿用 derive_seed(edit, seed)+1、原 CE 抽样、负例混合、Adam/clip/normalize 顺序和 c_minus。R0 保持原 nearest-first 和并列规则。
- CPU 单测通过：功能向量锚定、同范数反向响应、独立 gauge 变换、固定正尺度、冻结引用、有限学生梯度、next-token/padding mask、空 mask 拒绝、异常 hook 清理、beta=0、整体缩放、optimizer/RNG 恢复、诊断开关轨迹、选择规则的缺失与并列处理。
- 真实模型机械 smoke 通过：上游 h、predictor mask、AH/AHS 禁用新项的最终权重、诊断开关轨迹、gamma=1 与原路径 tokens、gamma=0 显式 Base-off tokens。
- 完成真实 R0 的 bank8/9/10/11/12/24 缓存决定重放，覆盖 DEV/REG 旧 47 输入及固定早期 8 编辑正例，形成 56 组路由/结果汇总。未以专家 ID 相同为由复用反事实答案。

## 数据边界

正例锚定只读取本编辑 native 和四个 S_fit，按 0.5 / 0.125×4 计算 c_plus。CHECK 为已暴露选择资料，48 个正例和 17 个负例，不进入训练或挖掘。实际继承 U_bg 为 127 行，区别于前轮最早报告中的初始辅助池清单。CHECK 与 fit/formal 无输入重叠，U_bg 与 formal 无输入重叠。

DEV、REG、原 47 输入面板均标为暴露诊断/回归，不称 CONFIRM。没有合法 CAL 正例，不拟合门控；fit-based anchor 也不构成跨图像泛化保证。

## 已有路由证据

旧 47 输入在 REG bank8→12 有 16 次 OFF→ON、11 次专家切换。按相邻插入检查，第 9/10 个无变化，第 11 个新增激活 1 个，第 12 个新增激活 15 个并切换专家 12 个。DEV bank8→12 则净有 5 次切换、无新增激活。前缀之间净变化数与逐次变化数不是同一计数单位。

路由变化并不自动等于新增答错；对应正确→错误、错误→正确、same 和 missing 已分别汇总。第 9/10/11 的实际答案未额外生成，不从专家 ID 推断答案。移除第 9/10/11/12 个条目的四条件生成仍未准入。

## 成本与准入阻塞

模型变更更新：用户随后要求改用 GPT‑6.1 sol，本轮已改为 `gpt-6.1-sol` / `high`，模板不变。采用独立评分命名空间，旧 Astra 分数不直接用作 Sol 评分。为同一 Judge 下比较，需重新评分的已知参考/Base payload 共 349 个，因此当前 pilot 保守预留从 2271 调整为 2620 次。无需为此重训或重新生成旧回答。模型等价性尚未建立，评分调用仍未启动。详见 JUDGE_MODEL_AMENDMENT.json。以下 2271 是模型切换前的预检快照。

机械 smoke 共四条 80 步续训，合计 320 步，约 0.03417 GPU 小时，0 次 Judge 尝试。这些步骤不计入正式 DEV 的 88 条续训，也不提升为正式方法效果。

实时继承 Judge 累计为 5779 次。旧无限额回执明确限定上一轮 P0–P5；本轮未收到新增授权。首个 pilot 的保守评分预留为 2271 次：5 个新训练分支的 single、bank4/8、CHECK，加 3 个非单位 gamma 的 CHECK；gamma=1 使用已核验 H 参考。未知 payload 去重率按零折扣计算。后续全 DEV/REG 仍需按完整块计算和预留，不把这个数当作全轮费用。

GPU 5/6/7 的 UUID 与空闲显存已核验；实际 smoke 仅占 GPU5。两次 smoke 会话已结束，无本轮 GPU 租约仍在运行。新目录约 23.8 MiB，soft/hard 为 10/20 GiB；已按明确清单删除 1 个已完成 smoke 恢复文件，保留源码、初始化绑定、梯度曲线和回执。旧权重未删除。

当前阻塞是本轮 Judge 预算授权，不是 DEV 科学门槛失败。只有预算明确、完整 pilot 准入并结束后，才能选择 beta/gamma；DEV 至少一个预注册候选明确通过联合门槛才进入完整 REG 比较。待准入阶段的编排与统计收口仍需继续完成，不将已有 worker 文件等同于完整实验已经运行。
