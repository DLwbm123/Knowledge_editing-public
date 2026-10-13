# Scope-safe Router 执行计划

基于 PR9 冻结公开提交 f5c629ccb8b7eac2e5792ee250b740ad876ba487，新建独立阶段。旧 PR7/8/9 只读。本轮零训练，不改 loss、optimizer、rank、layer、precision/backend 或 expert weights。

## 冻结与顺序

1. P0 审计 U_bg 与 DEV/REG/old47 的 input/source 隔离，positive 仅 native+四个 S_fit；锁定 ROUTER_CONFIG。
2. P1 在合法冻结 U_bg latent 上审计 DEV24、REG24 的全部历史 R0 insertion。分别按 H_count/H_source/H_switch 排名，不加权，不用 correctness。完成与保存后才读取 PR9 暴露损伤，单列 REG bank12 expert12。
3. 构造 R0、RCAP、NEG0、SAFE；RCAP 的 B前一库使用本方法此前安全半径，当前 candidate 用原半径试插入，再以 float32 nextafter 排出 captures；native 无法保留标记 UNSAFE_NO_RADIUS_SEPARATION，禁止替换规则。singleton 独立从空库插入。NEG0 固定 tau=0，复用原 hard-negative 16/source-cap2/hash tie 规则；缺少合法 negatives 标记 UNSUPPORTED_NEGATIVE_PROTO。
4. P3 完成全部机械检查。R0 保持历史路径；绑定一致时复用历史 H 原输出。其余路由与已验证有效 expert 相同的输入可复用同一确定生成，不能按相同 expert ID 代替绑定验证。至少实际生成机械样本验证 token exact parity、RNG、hook 和 Base 不变。
5. P4 一次性冻结并运行 DEV24 四 router 的 single、bank4/8/12/24，以及四 bank prefix 的 EXPOSED_REGRESSION old47。所有 router 完成后再判定，不按早期效果删分支。
6. P5 按 task/prefix/route bucket、switch matrix 与正确性输出 Positive Rejection、Negative Rescue、Negative Damage 和 NetRescue。单列 bank8→12、12→24 的固定共享输入，不把新增输入混入 transition。
7. P6 用原冻结 Base 资格及 shared-key exact Boolean missing。T0/T1G 最多0新增错误，T2G最多1；bank24 Pressure 相对 H/R0 至少+3已知正确且 exact下界≥0；old35各prefix最多1损伤；正式T2L≥2新增已知错 FAIL。缺失无法决定则 INCONCLUSIVE；空资格分母明确不可评估。至少一处已知 route rescue，否则 NO_ROUTING_EFFECT；不构造总分。
8. P7 多个明确PASS时按 RCAP→NEG0→SAFE 复杂度顺序选一个。P8 只有明确 DEV PASS 才运行 REG24 的 R0 reference + best candidate；不追加不必要的 control，不调参数，REG 为暴露回归，非独立确认。
9. P9 独立盘点可合法隔离且有 verified lineage 的 CAL_SCOPE_POS。未验证改写或其他正式面板不能充作新 cross-image positive；不足标为 CAL_SCOPE_UNAVAILABLE。本轮参数不可因 side-track 改变。

## 运行与交付

GPU仅5/6/7，进程中性入口；本轮不创建 adapter/optimizer。source/models/旧 experts复用只读路径；新输出仅新run。软10GiB/硬20GiB、最小8GiB余量。Judge继承7156，固定sol/high及原prompt/key；不重判41永久missing，新payload统一最多一次attempt，连续3新transport故障停止新请求。后台有限controller/scorer按依赖推进；本轮未要求恢复旧小时监测。

本轮结束后公开 source/config/tests/脱敏聚合与限制，新review分支和draft PR；不公开医疗输入、raw responses、逐题私有评分、weights或凭证。不为6–12小时目标补实验；没有明确PASS即收口。

## 最终执行状态

2026-10-01T12:02:52.525163+08:00：20/20 DEV任务完成；三个候选全部FAIL，REG=NOT_ADMITTED，按冻结stop condition提前收口，未扩展实验。12项实际机械检查PASS。专家冻结、训练0、新Judge0、未解决积压0。
