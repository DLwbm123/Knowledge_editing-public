# 真实 KV 生成调用组件

已实现官方 adapter 的 cache=True/False 固定两次完整生成调用及只读 pre/post observer；尚未加载或运行真实 VLM。本轮无数据搜索、SSH、GPU、Judge、合成模型或优化轨迹。

DynamicCache 会在 forward 中原地更新，故必须在 pre-hook 读取 past 长度。支持实际 get_seq_length 接口；未认证 legacy tuple 直接拒绝。缓存路径要求 prefill 后非空且递增 past、单 token decode；非缓存路径要求无 past 且输入长度递增。自然仅一个 token 则 CACHE_NOT_EXERCISED，不换问题。

沿用冻结 greedy/1024 上限和每请求128F；必须 EOS 完整结束、两次 continuation token 精确一致、每个生成 token 对应一次 forward，逐步末位置完整词表 logits 最大绝对差≤.05、relative RMS≤.001。缓存 logits 仅保存 detached CPU RAM，非缓存逐步累计误差，不保存完整序列位置 logits。观察 hook 在异常路径移除，不修改原 generation_config。每次调用前由外层提供持久独占预约，预算复用外层实例和唯一计数 hook。

8项纯控制检查通过：缓存轨迹、非缓存增长、无decode/无cache/停滞cache/错误复用/未认证tuple拒绝与语法。没有模型实例、forward、真实缓存或生成实测；不能据此称KV烟测PASS。组件只返回 KV_GENERATION_COMPONENT_PASS，不产生整个资格PASS。

剩余工作：实际加载caller、输入及来源绑定、系数导数/native合并诊断与完整恢复、冻结320F/4B/300秒预算适配、guard及硬件准入，统一为真实烟测入口后才可执行。当前不能运行IND2/JOINT2。科学参数和旧冻结协议未改。
