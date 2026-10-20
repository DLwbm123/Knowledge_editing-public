# E01 B/C已完成输出的快速盲语义核验

最新人工请求“快速进行验证”，仅补现有B/C完整生成的语义评分缺口。原controllerBC轮Judge0和语义NA作为历史终态保持；本轮新建独立评分记录，不改写旧正式八例或重评原246条。没有新GPU、编辑、生成或样本。

只收集已终态controllerBC两条完整T0输出及原manifest同源问题/已验证目标答案。opaque随机标识与B/C映射仅operator可见，评分只见opaque_query_id/question/gold_answer/raw_base_answer四字段；不见臂身份、NLL/margin、先前结果、其他评测或私有文件。沿用原source-answer-agreement rubric/结构化格式和Astra medium，既有登录额度，不购买新额度。

最多1批2payload、每条仅一次、重试/fallback/重判0，输入≤64KiB，模型调用≤600秒，新增文件≤20MiB。调用前保留RESERVED；隔离/格式/网络/预算失败记missing，不自动重放未知状态。评分工具、memory、项目/其他batch访问全部关闭；原共享helper默认high以明确CLI配置覆盖medium并核实际参数，回执如实记载，不修改共享源文件。

仅source-answer agreement，非图像重阅、独立专家或临床正确性。通过与否只针对E01每臂一条完整输出；不证明泛化、locality保护、控制器因果增益或C优于B。结果缺失保留有效分母及上下界，不用精确匹配替代语义评分。

完成后仅匿名判定、评分用量/隔离证据与中文报告公开，QA/目标/原回答/token/身份/路径不公开；沿既有7897代理推送核远端及匿名访问。评分模型为无工具隔离Judge，不启动协作subagent。
