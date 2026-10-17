# KLOD推导、官方实现与本项目适用边界

状态：只读文献/代码审查与新协议准备；没有模型调用、安装、GPU队列或Judge请求。官方仓库仅临时检出用于阅读，没有执行其训练程序。

论文：[KLOD, arXiv:2608.27839v1](https://arxiv.org/html/2608.27839v1)。官方代码：[Hostoday/KLOD](https://github.com/Hostoday/KLOD/tree/f1db53037d4d521cf97fb32104e7e39b8a3f628c)，本次固定commit `f1db53037d4d521cf97fb32104e7e39b8a3f628c`，MIT许可。下面的实现判断以该commit为准，不保证其他版本相同。

## 论文主张与实际支持

KLOD在答案位置采用逐token有界logit-odds目标，并保持排除当前目标token后的条件分布；在prefix位置保持完整分布。论文主设置alpha=.85、non-target系数.6、prefix系数1.2。附录E明确主实验是整批编辑请求跨epoch的Breadth-First训练，主表主要使用teacher-forced token指标；附录O另有自由生成检验。它支持一种编辑—局部性折中，不支持医学逐题零损伤或单病例在线无重放的保证。来源：论文§2、附录E/F/O。

这里采用其分布分解思路，不迁移论文的结果数字，也不把alpha=.85作为本项目CE<.1的替代标准。新的数学推导见THEORY_ZH.md，阶段一对照仍使用原CE。

## 逐入口实现审查

全部链接固定上述官方commit；行号是本次实际阅读的位置。

|入口|已核对行为|对本项目的要求|
|---|---|---|
|[train_klod.py L554–578](https://github.com/Hostoday/KLOD/blob/f1db53037d4d521cf97fb32104e7e39b8a3f628c/klod/train_klod.py#L554)|分别tokenize prompt与拼接序列，以左padding及prompt长度构造target mask，再shift一位|医学VLM必须用processor实际label/attention映射；首答案token不能落进prefix；图像patch/sentinel不能作为语言监督token。拼接边界重新分词需fixture，不照搬字符串长度假设|
|[L580–622](https://github.com/Hostoday/KLOD/blob/f1db53037d4d521cf97fb32104e7e39b8a3f628c/klod/train_klod.py#L580)|teacher先临时换回参考editable W，no_grad前向后恢复；non-target把两侧目标logit置dtype最小值后重归一化；log_target=True、batchmean|KL方向是reference→current。target rows被展平后batchmean是token平均；不等于本项目先按视图加权再按token平均|
|[L624–720](https://github.com/Hostoday/KLOD/blob/f1db53037d4d521cf97fb32104e7e39b8a3f628c/klod/train_klod.py#L624)|odds hinge；prefix mask同时排除前后padding与target；三个loss相加|理论上的坐标分离不保证共享W梯度正交。保留损失仍可使已达标token再次跌破阈值，因此必须在最终模型重新检查|
|[L657–681](https://github.com/Hostoday/KLOD/blob/f1db53037d4d521cf97fb32104e7e39b8a3f628c/klod/train_klod.py#L657)|alpha夹在[1e-6,1-1e-6]，rest-LSE通过总LSE和目标质量相减计算，质量夹到1-1e-7|高概率处不再精确等于直接排除目标的logsumexp。未来本地实现应直接masked-LSE，不能把数值截断当概率/医学间隔floor；非有限输入应显式失败|
|[L831–936](https://github.com/Hostoday/KLOD/blob/f1db53037d4d521cf97fb32104e7e39b8a3f628c/klod/train_klod.py#L831)|Adam；参考W在训练前快照；每epoch shuffle全部请求；model.train；异常/非有限batch可跳过；epoch平均总loss早停|不继承静默跳过、early-stop口径、Adam或参考权重替换路径。VLM需确定性teacher与原生FP16身份检查；dropout若非零，teacher/current噪声可能制造伪KL，须检查而非假定|
|[L245–295](https://github.com/Hostoday/KLOD/blob/f1db53037d4d521cf97fb32104e7e39b8a3f628c/klod/train_klod.py#L245)|目标文本规范化并追加EOS|不能悄悄更改原答案token/EOS约定；本阶段冻结原数据与token路径|
|[run_klod.sh](https://github.com/Hostoday/KLOD/blob/f1db53037d4d521cf97fb32104e7e39b8a3f628c/scripts/run_klod.sh)|脚本默认early_stop_loss=.1；YAML对应字段为null，trainer缺省路径另有处理|运行结果必须绑定实际命令/配置，不能只读论文或YAML声称精确复现|
|[train_klod_locft_150k.py L1330–1370](https://github.com/Hostoday/KLOD/blob/f1db53037d4d521cf97fb32104e7e39b8a3f628c/klod/train_klod_locft_150k.py#L1330)|分阶段只训练新slice，每阶段重建teacher W快照，阶段内部多epoch|代码确有跨批不重放旧批次的扩展；不能说KLOD完全没有no-replay模式。但它仍不是每次仅访问一个病例、不可重锚累计保护的本项目设定|

审查覆盖loss、mask、teacher、优化/停止、主配置、分阶段入口及评价口径。未安装依赖、未跑作者模型或数据、未独立复现论文性能；没有把作者整个仓库复制到本项目。

## 与K结果连接，但不越过证据

本项目K在同前100连续面板把T2L损伤30/34降到4/34；T2G相对C共享缺分差界[-1.0833,-.0833]pp。K同时引入61训练保护源和KL目标，并保留历史native重放。其机制是训练中直接看保护源当前函数变化，与只保留有限参数统计不同。不能据此认定KLOD或同样称作KL的摘要能复制K的效果。

K带来的设计要求是：明确保护对象、承认覆盖缺口、验证实际行为。KLOD带来的可借鉴点是：在同一分布里区分必要的目标质量变化和不必要的非目标重排。两者不是已经验证的等价机制。

证据：[本项目截至R汇总](../purew_ecw_r1_v1_20261015/EXPERIMENT_SUMMARY_ZH.md)、[K协议](../purew_protect100_v1_20261013/PROTOCOL.json)。所有旧结果、失败和永久缺分不改写。
