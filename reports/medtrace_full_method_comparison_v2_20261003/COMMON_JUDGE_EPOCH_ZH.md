# 原146共同评分epoch：执行前冻结

本记录在本轮首个新Judge请求之前形成。它实现既定计划，不更改数据、模型训练、路由、门槛或历史结果。模型固定gpt-6-astra/high，沿用原Stage17 `MEDTRACE_STAGE17_SOURCE_AGREEMENT_V1` 提示、四个可见字段、strict JSON schema/顺序校验、每批最多50条、无工具/记忆/旧分数/项目指令的独立临时CLI。模型没有可绑定的immutable snapshot，后续结果不得声称版本固定的独立确认。

epoch为 `MEDTRACE_ORIGINAL146_COMMON_ASTRA_20261003_V1`。Base与三个方法所有需要的raw输出使用共同新epoch；原Base eligibility/masks继续固定，新Base mask只用于另列敏感性分析。成功历史分数不混入新主表。匹配同一query、原图身份、prompt tokens/attention、runtime、generation、raw text及raw tokens和Judge模型/提示/协议的载荷可共用一个布尔判定；不能按答案字符串共用。每个消费者另保留方法、single/sequential、prefix、编辑、全output/training/插入祖先绑定，不把共享判定当独立重复。

永久missing仅按同完整模型/提示/载荷身份继承。原Astra Base2465条均已有布尔判定，无永久missing；历史scope的41永久missing使用gpt-6.1-sol和不同协议，仍原样永久锁定，不修改/重判/抹除，也不把不同模型协议当同一个Astra key。四基线phase原始输出准入PASS，尚无本轮共同评分。原成功评分全部只作历史参考。

远端SQLite私有队列先接原Base和完整已准入基线，再接每个已经完整生成的MedTRACE single编辑，最后接完整sequential原prefix面板。不查看性能决定入队。每个新载荷key请求前原账本原子预留，最多一次；格式有效才发布，任何已请求但没有有效结果的key永久missing，不补判。连续三次新的transport故障停止新请求；非transport/绑定/隔离异常立即停新请求。进程恢复只发布已经保留的有效响应，不能再请求同key。

运行使用原已授权登录和127.0.0.1:7897代理；操作员不复制凭据。原Stage17隔离读取边界保持，预检child通过环境传递路径以满足中性argv，不改Judge提示或可见数据。每次CLI上限600秒，首次72h截止及32000载荷预算继续继承，失败照计。本地每次仅中性/tmp有界批次，256MiB上限；终结私有批次及CLI证据上传/data/bmw后清理本地临时副本。患者图像不进入云评分，只有既定四字段文本；私有文本和原始评分不得公开。

GPU健康worker与已冻结私有SOURCE不热改；新增队列与评分工具保存在私有tools目录，版本单独记录。生成、评分、指标、报告和公开交付分别记状态，生成完成不等于实验交付完成。沿用原完整146分母、shared-key missing精确界、edit-macro/probe-micro、10000次paired bootstrap(seed20260912)、来源组敏感性、H_eval NA与成本限制。

启动机械检查发现CLI实际argv由应用launcher重写入口路径；当前健康scorer9579未重启，由绑定parent启动身份与本批output路径的中性watchdog9769执行同600秒上限，后续scorer代码已修正为output路径匹配。隔离预检的模块组装、cwd和/tmp真实路径修复均在首个请求前完成，零请求；私有PREFLIGHT_RECOVERY明示记录时间晚于预检修复，未虚构计时。原GPU SOURCE未改。
