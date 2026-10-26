# 4编辑医学命题保护实验结果

**本轮为四编辑开发试验；不构成独立确认，不自动扩展到146。**

比较同一共同评分epoch的Base、原M13P（LEGACY）、FULL_CTX、MED和GEN；历史accepted标签未更改。
FULL按verified reference的保守来源支持判定，参考外的医学解释即使可能真实，也不能当作已验证。

|方法|T0 CORE|FIT CORE|T1G CORE|T2G CORE|T2G FULL|固定T2L CORE|
|---|---:|---:|---:|---:|---:|---:|
|BASE|0.00% (0/4)|0.00% (0/16)|0.00% (0/16)|0.00% (0/16)|0.00% (0/16)|100.00% (2/2)|
|LEGACY|100.00% (4/4)|100.00% (16/16)|100.00% (16/16)|87.50% (14/16)|81.25% (13/16)|100.00% (2/2)|
|FULL_CTX|100.00% (4/4)|100.00% (16/16)|100.00% (16/16)|81.25% (13/16)|81.25% (13/16)|100.00% (2/2)|
|MED|100.00% (4/4)|100.00% (16/16)|100.00% (16/16)|87.50% (14/16)|87.50% (14/16)|100.00% (2/2)|
|GEN|100.00% (4/4)|100.00% (16/16)|100.00% (16/16)|87.50% (14/16)|87.50% (14/16)|100.00% (2/2)|

百分比为编辑宏平均，括号为微计数；固定locality只使用原资格掩码，空支持为NA。

- MED_minus_LEGACY：T2G +0.00pp；0升/4同/0降；四编辑描述性bootstrap95%区间[0.00, 0.00]pp。
- MED_minus_FULL_CTX：T2G +6.25pp；1升/3同/0降；四编辑描述性bootstrap95%区间[0.00, 18.75]pp。
- MED_minus_GEN：T2G +0.00pp；0升/4同/0降；四编辑描述性bootstrap95%区间[0.00, 0.00]pp。

## 保护与完整回答

以下新增损伤只在本共同epoch的Base正确且LEGACY仍正确身份上计数；旧伤修复单列，不能抵消新伤。
这是小样本已有locality输入的审计，不能保证未测病例或全域安全。

|方法|CORE新增损伤/支持|FULL新增损伤/支持|CORE旧伤修复|FULL旧伤修复|
|---|---:|---:|---:|---:|
|LEGACY|0/2|0/0|0|0|
|FULL_CTX|0/2|0/0|0|0|
|MED|0/2|0/0|0|0|
|GEN|0/2|0/0|0|0|

## 机制与费用

31/61原BASIS项构造解剖结构出现命题，覆盖24来源组；其余项三臂共同保留原曲率。逐锚点匹配trace；MED/GEN逐项秩1谱匹配，不声称全局谱相同。
实际mask、有限差分、legacy一步、dense二次型及梯度、换基和save/load检查的回执与冻结配置保存在私有运行记录。
GPU进程驻留时间包含加载、等待和共享资源竞争，不是纯GPU计算时间；逻辑forward包括PrefixCache复用的调用，不能直接当作完整模型FLOPs。

```json
{
  "cost": {
    "counts": {
      "forwards": 7196,
      "backwards": 4956,
      "operator_backwards": 1920,
      "updates": 1920,
      "generations": 207,
      "vision_forwards": 371,
      "LM_input_positions": 2557338
    },
    "GPU_process_seconds": 3111.923309326172,
    "max_cuda_bytes": 33827649536,
    "owned_bytes_last_sample": 859803648,
    "Judge_attempt_items": 144
  },
  "curvature": {
    "mapped": 31,
    "common_fallback": 30,
    "mapped_source_groups": 24,
    "full_rank_min": 32,
    "full_rank_max": 32,
    "max_relative_trace_error": 1.0285939555654267e-07,
    "MED_GEN_cosine_min": -0.1735788125729138,
    "MED_GEN_cosine_max": 0.10777831748080632,
    "global_spectrum_matched": false
  },
  "quality": {
    "BASE": {
      "n": 69,
      "EOS": 69,
      "at_cap": 0,
      "unsupported_expansion": 45,
      "extra_contradiction": 4
    },
    "LEGACY": {
      "n": 69,
      "EOS": 69,
      "at_cap": 0,
      "unsupported_expansion": 12,
      "extra_contradiction": 0
    },
    "FULL_CTX": {
      "n": 69,
      "EOS": 69,
      "at_cap": 0,
      "unsupported_expansion": 11,
      "extra_contradiction": 0
    },
    "MED": {
      "n": 69,
      "EOS": 69,
      "at_cap": 0,
      "unsupported_expansion": 11,
      "extra_contradiction": 0
    },
    "GEN": {
      "n": 69,
      "EOS": 69,
      "at_cap": 0,
      "unsupported_expansion": 11,
      "extra_contradiction": 0
    }
  }
}
```

## 本轮判断

未观察到本轮预定联合方向性改善。
四编辑和单个GEN方向仅约束本轮；完整逐编辑计数、全部对照、损伤、生成完整性与费用见RESULTS.json。没有用本轮结果改变候选、种子、步数、权重或来源，也没有自动启动扩大实验。
