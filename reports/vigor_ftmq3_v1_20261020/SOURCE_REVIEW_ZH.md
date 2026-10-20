# 源码核对与适配边界

已通过本机7897代理取得EasyEdit main当时版本4c109870955a4522ac3d7cf10ad00f34de8e4f0d的ft_main.py、ft_hparams.py，均HTTP200；源码只存本地取证目录，不加入公共提交。论文引用EasyEdit（附录A脚注2），未公开绑定作者实际源码commit，不能把当前main称作者实验精确版本。

[固定上游FT实现](https://github.com/zjunlp/EasyEdit/blob/4c109870955a4522ac3d7cf10ad00f34de8e4f0d/easyeditor/models/ft/ft_main.py)区分prompt_last与target_new。prompt_last将多个目标token概率从prompt最后位置读取；target_new用拼接目标的因果位置及mask平均CE。本地旧版本没有objective_optimization分支，不能当完整回答NLL复用。当前上游Adam、loss<1e-2提前结束、可选float坐标clamp；未看到通用gradient clipping。模型/配置实际选用哪种objective、module及norm需独立确认；本次不冒称作者配置。

[论文](https://openreview.net/forum?id=RebJSGsREI)本地全文附录Table10教师layer6/25步/lr5e-4，正文强调原始编辑目标及每批W0教师，无回答参考的生成；学生全层全词表reverseKL/多epoch。本次只借鉴独立参数教师和三个数值设置，明确改为当前原chat完整源full-NLL、单down_proj、FP32选择层/FP16骨干；Adam细节/无clamp是独立事前选择。FT-M不是“只要一个Adam就忠实复现”。原Chat/Base两链的作者模板边界尚不够明确，保留该限制。

读取qwen2.5配置时一次代理TLS EOF失败，未绕代理；不是模型调用或科学失败。该可选配置不能解决作者实际版本绑定，未继续猜测。OpenReview网页读取工具不可达，但本地完整PDF/text已存在并复用，不下载第二份。

现有editor.editable_fp32已负责选择矩阵cast、原生提交及异常回滚，nll负责原目标mask的完整因果CE；本次直接复用，fit只加Adam循环及状态检查。教师无旧F0保护不是修改旧保护门：独立teacher阶段先检验是否具备源内知识访问能力，学生尚未实现/部署。
