# Pure-W 十二小时自主窗口最终报告

窗口：北京时间2026-10-19 23:32至2026-10-20 11:32。窗口已结束，不再新增实验；20轮均已终态、恢复并逐轮公开交付，没有待运行任务或未完成交付。半小时静默监测保持ACTIVE，不自行暂停，后续新人工指令按其实际范围处理。本次最终汇总不调用模型、生成或评分。

## 最重要的结果

正式八例性能保持原报告：T0 0/8、T1G修复0/29（PostAcc1/32）、T2G2/32；T1L/T2L保持率有效支持0/0，NA。原评分246条齐全，不重判。后续E01机制开发不能回填这些正式数值，也不能证明真实locality保护。

前19轮证明E01存在原门内实际NLL/竞争余量双收益更新：旧16步共同终点NLL1.510729909/margin−5.8671875，到OrthantBalance为1.293198824/−3.734375，null相对最初基座−.005858421，官方首目标rank3、p.019042853，仍未翻转。这是预优化路径的局部开发结果。

连续提案必须做native有效更新准入：SmallDirection中FP16参数写入使一阶NLL符号从负变正；但native准入仍不能保证有限步。SignedRead固定q完整步参考链路NLL上升.021308661，native上升.021849036，margin均退.1953125；可编辑层FP32有限响应内已反号，不能仅归因额外native执行，也不能将R唯一解释为纯曲率或全链路FP32结果。null的执行差异−.003023386仍重要。

第20轮来自用户最新明确授权，第一次把固定有界控制器用于从原基座独立开始的两臂8步比较，不延续旧失败，也不选择旧只读半步：

|指标|B普通旧F0保护NLL方向|C同方向加native筛选/平衡|
|---|---:|---:|
|接受步数|8/8|8/8|
|最终NLL|2.454109669|2.776510954|
|首目标最强竞争余量|−8.8671875|−8.6953125|
|最初基座null漂移|−.000664711|−.008582830|
|原secant|6|2|
|编辑forward|112|62|
|完整官方生成token|51|47|
|目标文本精确匹配|否|否|
|语义T0评分（Judge0）|NA|NA|

两臂最终原门均过，全部恢复。B第7步根据预注册双收益实际/预测比拒绝全步、半步后接受quarter，展示有界响应控制的具体作用；无controller-off消融，不声称其因果性能增益。C较B少调用和补偿，但NLL较高，margin仅略好，实际成本和进度不同，没有明确特殊方向优势。完整输出自然结束，原1024token配置不变，精确不匹配不能冒充语义失败。该8步结果也不能与旧16步/预优化端点直接判优劣。

## 方法判断与下一阶段边界

保留旧F0/原native门/null最初基座参照/旧secant，将连续方向、native准入、实际有限步验收和完整生成评价分开。Secant只处理已双收益且trust合格的null失败，补偿后全部条件重验。固定有限回溯可以在新预注册算法中合法使用；旧只读位置不能追溯变成已接受更新。

下一阶段优先保持简单控制器作为基线，只有证明特殊方向在可比较预算/进度和多个编辑上带来额外完整生成收益，才保留复杂机制。还需独立建立非零Base正确支持的保护评价；本窗口没有授权扩样本/新保留集或新增评分，因此未实施。当前不自动追加16步、八例、Judge、全链路FP32、尺度/掩码/预留搜索或历史失败重放。

## 总成本与公开交付

20轮累计GPU驻留 **1688.634602秒（28.14分钟）**，模型forward累计 **9668**；均计入失败及控制/恢复成本，未重置预算。GPU累计限4小时、最多24轮均未用满，未为填时间增加调用。窗口无新增Judge、模型/数据下载或checkpoint；第20轮新增两次完整生成，其余生成是对应协议的单token诊断。此成本不包含窗口前正式八例及历史机制实验。

空间账本最近已计量本地净增、所属远端运行根、临时入口与本地临时余量为 **65315392字节**；该值不是精确所有新增字节证明。全窗口保守预留4,516,582,400字节，低于内部18GB和用户所有本地/远程/临时新增20GB限制；保留历史文件，不删除或搬运模型。

所有20轮均有公开报告/精确远端SHA与匿名访问回执，运行主/GPU已退出。公开匿名完整标量轨迹、代码和中文报告；私有QA/答案/token身份/路径/生成token/梯度向量/mask/索引/权重不公开。所有GitHub操作经7897代理，无直连。

## 逐轮索引

|轮次|运行|冻结终态|GPU秒|公开结果|
|---|---|---|---:|---|
|1|responseReserve|STOPPED_MAIN_ACTUAL_NLL_FAILED|79.937|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/4d2aa017375dd805cd7ca9567c6aa5605e7faafe/reports/vigor_responseReserve_v1_20261019/completion/REPORT_ZH.md)|
|2|response3|COMPLETE|77.492|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/97dfff7cffcc8f3886ad636e0e7e5b9bf63a441a/reports/vigor_response3_v1_20261019/completion/REPORT_ZH.md)|
|3|responseFollow|STOPPED_MAIN_ACTUAL_NLL_FAILED|80.034|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/1e68d73e280cd88981b54bce9bdd8439d7568258/reports/vigor_responseFollow_v1_20261020/completion/REPORT_ZH.md)|
|4|responseRefresh|STOPPED_MAIN_ACTUAL_NLL_FAILED|78.885|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/5198d947790f5fd111231ffc4488ae526d0624fc/reports/vigor_responseRefresh_v1_20261020/completion/REPORT_ZH.md)|
|5|responseNcontrol|STOPPED_MAIN_ACTUAL_NLL_FAILED|77.481|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/4af6d409d860933fa199b67a005f9f48193b2e06/reports/vigor_responseNcontrol_v1_20261020/completion/REPORT_ZH.md)|
|6|responseQuad|STOPPED_QUADRATIC_SOLVE_FAILED|75.942|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/40f2fa42a1e393e7d5019c98b045a66988c0bd0b/reports/vigor_responseQuad_v1_20261020/completion/REPORT_ZH.md)|
|7|responseCenter|COMPLETE|80.075|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/c2383ce0dd71ee2063b31099a465c2d6e5477ba8/reports/vigor_responseCenter_v1_20261020/completion/REPORT_ZH.md)|
|8|responsePath|STOPPED_TRANSFER_MATCHED_PREDICTION_FAILED|112.805|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/ebab44097b8a84c2d22c4479779ec8eeeff1d694/reports/vigor_responsePath_v1_20261020/completion/REPORT_ZH.md)|
|9|responseNative|COMPLETE_ACCEPTED_NATIVE_BUDGET_PATH|80.062|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/8d9f1774d0ffdab7f0453f40aa04e69e739a2d4d/reports/vigor_responseNative_v1_20261020/completion/REPORT_ZH.md)|
|10|responseEndpoint|COMPLETE|98.488|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/4f76076bac63ac85ceac50223c237ab59ec6fca8/reports/vigor_responseEndpoint_v1_20261020/completion/REPORT_ZH.md)|
|11|responseNullstep|COMPLETE_ACCEPTED_NULL_HEADROOM_STEP|81.652|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/a5807e818c052ab30b31427ceda2a982dd717caa/reports/vigor_responseNullstep_v1_20261020/completion/REPORT_ZH.md)|
|12|responseSmallDirection|STOPPED_SMALL_NATIVE_ADMISSION_FAILED|79.945|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/1a47bcb71775c7a770f541fe28ae5fb8a9c639e3/reports/vigor_responseSmallDirection_v1_20261020/completion/REPORT_ZH.md)|
|13|responseQuantReserve|STOPPED_SMALL_NATIVE_ADMISSION_FAILED|81.033|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/2e277a5796aa3cd536ca9f0678c8c2ca2ff60dd1/reports/vigor_responseQuantReserve_v1_20261020/completion/REPORT_ZH.md)|
|14|responseNativeOrthant|COMPLETE_ACCEPTED_NATIVE_DUAL_SUBSET|82.041|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/30cb7ac63df3098c65a7953d1133d1a897fa56cf/reports/vigor_responseNativeOrthant_v1_20261020/completion/REPORT_ZH.md)|
|15|responseOrthantRadius|STOPPED_ORTHANT_NATIVE_ADMISSION_FAILED|80.941|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/1c3fd3c032e35af2c375557c87c81877b68d2f53/reports/vigor_responseOrthantRadius_v1_20261020/completion/REPORT_ZH.md)|
|16|responseOrthantBalance|COMPLETE_ACCEPTED_NATIVE_NULL_BALANCED_SUBSET|84.179|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/350b4ab0f895b66bad9c5ac69d1c155513df4d7d/reports/vigor_responseOrthantBalance_v1_20261020/completion/REPORT_ZH.md)|
|17|responseBalancedRefresh|STOPPED_BALANCE_NO_NEGATIVE_EXCURSION|99.691|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/ecb71f0ffcb7d2987e7fcfc8b072bb2714b42f30/reports/vigor_responseBalancedRefresh_v1_20261020/completion/REPORT_ZH.md)|
|18|responseSignedBalance|STOPPED_MAIN_ACTUAL_NLL_FAILED|88.945|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/79f9e02e3db4bd9e707341a56f6b297c54cf892e/reports/vigor_responseSignedBalance_v1_20261020/completion/REPORT_ZH.md)|
|19|responseSignedRead|COMPLETE|88.674|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/7f290748d49612753773846908e40f8eebcd409a/reports/vigor_responseSignedRead_v1_20261020/completion/REPORT_ZH.md)|
|20|controllerBC|COMPLETE|80.333|[报告](https://github.com/DLwbm123/Knowledge_editing-public/blob/39743dacd8e12cfbd16c4af437a214224abe6790/reports/vigor_controllerBC_v1_20261020/completion/REPORT_ZH.md)|
