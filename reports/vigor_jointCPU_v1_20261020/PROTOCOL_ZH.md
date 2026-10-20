# JOINT_ENDPOINT_CPU_Q1：合成接口资格冻结

独立CPU资格，不是38组数据/旧S续跑或VLM实验。只用合成3输出×4输入无偏置线性模型和完整softmax CE，CPU FP32求导/FP16实际写回。真实模型/QA/图像/旧权重/GPU/Judge0。不得将CPU通过称为原VLM双成功端点、原F0效果或医学保护。

W0为FP16全零3×4矩阵。源X=[e1,e2]、y=[1,2]，基座argmax均0；null X=[e3]、y=[0]。该可分离合成例故意具有null不变方向，用于验证接口，不代表真实null耦合。源W0梯度G做FP64 thin SVD，固定rank2，s>1e-10*smax准入；每列U绝对值最大元素规范为正，同步翻转V。只由当前合成源求基，不读null选择方向。检查正交与重建残差≤1e-8。

D_j=U A_j V^T，j=1,2。两个独立FP32 2×2系数矩阵，固定初始化A1=.05 E11、A2=-.05 E11；同W0新过程，两臂同初始和基，native写回才核不同。完整矩阵不通过.data安装来算梯度；函数式F.linear(X,W0.float()+D_j)维持A→logits图。

固定目标为两个源CE之和，加rho=10外罚：每端点source无改善relu(Lv-Lv0+1e-5)^2、null relu(abs(Ln-Ln0)-.01)^2、合成cost relu(.5*1e-4*||D||²-.001)^2。合成1e-4对角仅资格fixture，不是原F0或新校准。联合臂再加10*relu(.25²-||A1-A2||²)^2；对照去掉此项，其余相同。外罚不是精确约束求解，最终native不合格即本配方失败。

Adam lr=.02/betas(.9,.999)/eps1e-8/wd0，两个独立参数状态；每臂恰64步，不能首过null即停，不扫率/秩/步/多样性/初值。每步逐端点visual和null各自forward/backward释放图，再独立backward解析cost/diversity，计API backward与模型梯度路径区别。最终固定第64步端点，无历史最佳选择或第三候选。两臂同512个训练模型batch前向/梯度路径总预算；解析耦合开销另记。

最终FP16每端点源两个argmax都等于y、visual CE严格低于W0、null漂移abs≤.01、合成cost≤.001；两端点torch.equal为False。联合臂还要求native ||W1-W2||F≥.25；不能以连续系数距离代替。对照完整保留，即使失败也不改它，CPU资格不要求联合臂行为优于对照。通过只说明该合成可分离例与接口资格。

运行前CPU自检：FP64系数autograd对中心差分eps1e-6相对误差≤1e-5；非零系数梯度；Adam两个state无别名且64步；native强制重复拒绝；1e-8连续差在FP16基值1处折叠实例保留；实际loss不同的null门拒绝；正常/注入异常安装native后完整恢复权重/dtype/flags/mode/hooks/grad。非有限输入直接拒绝。运行仅两臂和上述固定fixture，不新增压力集救结果。

首次本机freeze起整体1h，单次远端CPU signal45秒/本机60秒，线程1/CUDA_VISIBLE_DEVICES空；模型batch forward≤800、模型梯度路径≤600、backward API≤800；计基构造/差分/最终门/恢复。新本机文件<16MiB，远端新文件0、不下载/安装，既有CPU torch环境复用。异常、预算或自检失败留结果停止不重试；科学不合格完整保存，不调超参。无subagent，无GPU/旧PID查询。结果完整匿名交付后才决定后继，唯一pure-w ACTIVE原位。
