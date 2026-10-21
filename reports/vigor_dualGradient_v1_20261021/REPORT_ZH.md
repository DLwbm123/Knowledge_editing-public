固定初始点诊断完成。原目标与forward有限，保护罚项偏导进入FP16反向后溢出；整体缩放会让源项下溢，已拒绝。后继采用逐NLL单位反向与FP32加权合成，详见相邻dualStable报告。

成本与恢复：{"status": "RESTORED", "forwards": 9, "backward_API": 3, "model_gradient_paths": 9}

运行：{"elapsed_seconds": 17.883240938186646, "peak_allocated_bytes": 25507879936}
远端新增字节：2893437；退出/GPU无占用确认。无更新、生成或Judge。
