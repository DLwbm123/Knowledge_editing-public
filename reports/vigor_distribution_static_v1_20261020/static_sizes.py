"""Integer storage arithmetic only; imports no model, tensor or external package."""
import json
from pathlib import Path

def main():
 a,b=14336,4096;d=a*b
 result=dict(kind='STATIC_INTEGER_ARITHMETIC_NOT_NUMERICAL_EXPERIMENT',input_dimension=a,output_dimension=b,parameters=d,one_FP32_parameter_matrix_bytes=4*d,one_FP64_parameter_matrix_bytes=8*d,dense_Fisher_FP32_bytes=4*d*d,dense_Fisher_FP64_bytes=8*d*d,packed_symmetric_FP32_bytes=4*d*(d+1)//2,Kronecker_full_factors_FP32_bytes=4*(a*a+b*b),Kronecker_full_factors_FP64_bytes=8*(a*a+b*b),diagonal_FP32_bytes=4*d,sketch32_FP32_bytes=32*4*d,server_new_storage_strict_cap=50000000000,known_calibration_attention_positions=19237,target_positions='UNKNOWN_not_equal_to_attention_positions',vocabulary_size=32000,configured_position_max=32768,one_position_full_Jacobian_FP32_bytes=4*d*32000,logits_FP32_bytes_at_configured_position_max=4*32768*32000,expanded_sequence_max='UNKNOWN_generation_cap_not_a_training_sequence_cap',model_calls=0,numerical_experiments=0,judge_calls=0)
 assert d==58720256
 assert result['packed_symmetric_FP32_bytes']>result['server_new_storage_strict_cap']
 assert result['Kronecker_full_factors_FP32_bytes']<1024**3
 (Path(__file__).parent/'SIZES.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
