# 原始native来源绑定修复

第一次CPU审计因“Native QA source binding ambiguous/missing”停止。部分既有native来自SLAKE validation/test；错误是以train候选池同时充当全部native的来源索引。失败代码、日志、入口/启动身份、首次manifest和排除回执保存于私有private/recovery/SOURCE_BINDING_1，失败CPU wall上界6.694327116秒，GPU/Judge/训练/生成均0。

修复仅为既定8 pilot中已暴露、已作为native使用的QA核对原出处：保护分片中的条目必须与已知native的图像身份、问题、答案逐项匹配，才能记录该native的原始注释。它们只进入anchor provenance索引，绝不进入候选池、同图新材料或新的CAL。非native保护QA仍只提供排除身份。不更换pilot，不改匹配/排除/门槛，不读模型结果，首时钟与输入/CPU消耗累计保留。

启动包装器另有一次占位符CODE的全局字符串替换，误伤PYTHONDONTWRITEBYTECODE关键字，Python解析前失败。源数据读取0、GPU/Judge0，wall上界0.247093125秒，私有LAUNCH_SERIALIZATION_FAILURE回执保留；改为单一JSON编码包并先compile，累计失败上界6.941420241秒。
