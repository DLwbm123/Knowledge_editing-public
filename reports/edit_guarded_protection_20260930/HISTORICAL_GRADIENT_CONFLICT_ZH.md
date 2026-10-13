# PR8 历史梯度审计

零重训、零新Judge。固定诊断batch按真实beta/lambda系数与记录的Gram内积重构g_plus/g_minus；H/S用原训练曲线的实际联合保护梯度。两种batch不混为同一次更新。

|方法|编辑|冲突比例|norm ratio中位数|实际全程clip比例|
|---|---|---|---|---|
|H|24|0.3958|2.9321|0.0000|
|L|24|0.3646|14.2140|0.0005|
|P_01|24|0.5000|0.5282|0.0000|
|P_1|8|0.5312|0.1787|0.0531|
|S|24|0.3438|66.3914|0.2630|
|SP_01|24|0.3958|10.1779|0.2687|
|SP_1|8|0.4062|2.7175|0.3469|

- S separate KL/residual gradient vectors and norms were not retained; its exact combined protection is available. No invented component split.
- PR8 anchor arms have fixed-batch Gram reconstruction and actual clip indicators, but lack actual sampled-batch CE-anchor/protection cross dots. Do not conflate the two batches.
- All simulations are offline raw-gradient algebra, not new model results or Adam monotonic guarantees.
