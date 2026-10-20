# 零调用评分入口修复

首次本机评分入口导入公共 write helper 时，间接导入 GPU prepare 模块，因本机无 torch 在模块导入阶段退出。该执行尚未进入 main、未建立远端评分预约、未构造 Judge packet、无模型请求/新评分。原启动回执和日志保留，不覆盖。

仅将 write 改为不依赖 torch 的 JSON 文件 helper；评分 rubric、候选、资格、模型、每条一次和总预算不变。按零模型成本入口修复约定，CPU 导入检查后允许第一次实际评分。新工程启动回执单列，原错误成本共享资格原时钟，不重生成六目标。

第二项严格源忠实评分在模型请求前，被既有 helper 的“原语义 prompt 精确绑定”检查拒绝：它仍拿原 source-agreement 常量核对新冻结的 faithfulness prompt。该项 execution_evidence/response 不存在、工作输入尚未复制，无 CLI 子进程；原第一项语义评分已完整执行并保留，绝不重判。该项旧 SUMMARY 的 new_attempts 是预约数量而非实际请求数，最终成本按实际 execution_evidence 计，单列零调用失败。

修复只在本次隔离调用期间将 helper 的 expected PROMPT 绑定为事前冻结的同一 criterion prompt，保留完全相等检查、schema、隔离路径、工具禁用和模型设置；结束恢复 PROMPT/profile/flags，避免同进程跨批包装器叠加。CPU核对原6个匿名 packet/同 rubric/同 schema/无任何旧调用 evidence 后，仅执行此前未发出的 faithfulness 六输入一次；不重生目标、不重评 agreement、不换 rubric，所有原失败文件保留。
