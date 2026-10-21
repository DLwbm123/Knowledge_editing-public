# 既有开发源真实烟测：来源已绑定，KV兼容失败

结论：不需要寻找新数据才能执行方法开发。复用原已授权 E01 源记录、原图像/问题/参考、既有prompt与图像绑定及原F0校准排除证据后，真实模型已实际加载和执行。新数据不是当前阻塞。未改变保留角色，不声称独立泛化，未重放旧教师/控制器或评分。

## 实际执行

首次调用完成官方native加载、类路径核对、FP16层6形状核对、基座NLL、当前源梯度、系数loss与导数诊断、非零native写入。在首次生成前置观察hook中，兼容forward的VAR_KEYWORD未展开，导致错误的输入形状拒绝。7次前向计数含一次被hook拒绝的调用，2次梯度；权重和W0损失恢复完成，无实际生成forward/输出。

修复观察器解包后，执行唯一工程续接，沿用原开始时钟并累计前次7F/2B。开启cache的生成已返回且其trace通过；关闭cache的请求在首次forward前被拒绝，错误为 Actual cache flag differs from request。累计61次前向计数（含两个前置hook拒绝），4次梯度调用；恢复完成。无完整双请求KV PASS，IND2/JOINT2未启动，Judge0。

## 已定位的根因

安装版Transformers 4.51.3的generate对decoder-only、inputs_embeds入口强制设置generation_config.use_cache=True。官方LLaVA多模态generate先生成inputs_embeds再委托该入口，所以传入use_cache=False并不能保证实际禁用缓存。adapter确实透传了use_cache，不能将问题误归为缺数据或简单漏传参数。

本轮不放宽观察器、不伪造无缓存trace、不更换源或生成容差追通过。固定烟测在兼容失败处结束。下一必要工作是实现保持完整视觉前缀的真正无缓存输入准备路径，再单独冻结该接口修复的资格验证；不得以同一包装再次盲跑。只有真实cache开关、完整token/logits与恢复全部通过后，才运行已冻结两臂。

## 关闭与证据边界

两次worker均已核退出，GPU进程列表均无其PID；两次日志均记录RESTORED。运行期采样核中性argv及固定GPU；远端本轮目录2,931,172字节，另有很小的中性入口文件，低于256MiB。峰值未形成成功RESULT，保持未知，不由瞬时GPU采样替代。原错误、失败预约与全部私有绑定保留，无checkpoint。

dual_entry.py为后续两臂加载入口，已完成语法检查但未实际执行；不能称两臂生产资格通过。公开代码、协议和匿名结果，私有QA/输入/生成内容不公开。
