# V25实现与准入

CPU及五分片native真实模型机械准入全部PASS后，依赖屏障自动放行正式训练。CPU随机合成检查不是训练后的效果证据。公开CPU_ADMISSION.json与REAL_MODEL_ADMISSION.json仅含匿名机械结果；私有保存完整绑定/计数/执行身份。

- 复用V24 TT、canonical scalar RMS、PrefixCache与相同生成路径。C160保存参数和完整scalar状态，延长不重置二阶历史；停止检查针对完成更新数。每编辑停止前不评价T2G/T1L。
- CPU：诊断开关180步C参数/损失/scalar精确，160步参数及scalar保存加载、159/160/640停止边界、无EOS监督时NA与RNG不变；原TT/mask/save-load合成检查、缺输出拒绝。
- 五真实分片：仅native输入的诊断开关三步损失/梯度/参数/scalar精确；zero logits差0、teacher-prefill差0、KV差最大2.956390380859375e-5，未放宽rtol1e-5/atol5e-4。准入合计55 LM forward/35反向/0正式更新/0生成。五REAL_PASS先于各TRAINING_RELEASE。
- 诊断从既有logits与梯度detach提取；gate/down额外线性诊断明确计数。未观察的位置保留NA，不补额外训练目标。native实际生成前缀1/2/4/8 cached/uncached检查属于后置消费者，尚不能用启动准入宣称其全部完成；新增forward进同一账本。
- 180 OFF绑定检查后只复制冻结输出，0新Base生成/0key；仅C160/C_STOP，最大432消费者。两状态相同精确复用，记录no_new_generation。新增唯一回答待完整输出后隐藏四分类及reason，缺失不能完成。
- 独立guard原始24小时及各分片2小时窗口、固定操作上限、远2GB/软1.5GB/8GiB余量先到先止。精确PID/ticks/独立PGID/SID注册先于GO；中性父子入口。外部中性入口在同文件系统硬链接进受测root，guard覆盖实际字节，不产生模型或权重副本。
- 每编辑step160/active/final三个文件只在两终态forced/native缓存检查/save-load/绑定/token消费者全部PASS后按清单退休。未注册V26或bank依赖；失败保留现场，不盲重试已尝试生成。

CPU/真实准入通过不等于C160历史重建核对通过，更不等于泛化改善。各编辑另有161点源损失重建检查与终态消费者回执；最终报告必须区分源拟合、生成稳定性、T2G及T1L范围。
