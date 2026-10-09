# 统一FP32真实编辑约束机制实验

{
  "status": "COMPLETE",
  "decision": "LOCAL_MECHANISM_SIGNAL",
  "source_gate_passed": true,
  "primary_panel": "ORIGINAL_DEPLOYMENT_BASE_CORRECT_63",
  "primary_queries": 63,
  "primary_sources": 29,
  "baseline_semantic_migration": {
    "original_correct": 63,
    "FP32_correct": 62,
    "denominator": 63,
    "new_scoring": false
  },
  "protection": {
    "RAW": 6.244406047895309e-08,
    "BOUNDED": 2.7277611591664253e-08,
    "MATCHED_RAW": 6.244406047895309e-08,
    "expert_CI": [
      -3.935261352562388e-08,
      -3.154199770159148e-08
    ],
    "source_CI": [
      -3.8591936562463254e-08,
      -3.13545912904449e-08
    ]
  },
  "edit_progress": {
    "RAW": 0.006137537572136307,
    "BOUNDED": 0.005524845475912469,
    "MATCHED_RAW": 0.006137537572136307
  },
  "response_lost_tokens": {
    "RAW": 0,
    "BOUNDED": 0,
    "MATCHED_RAW": 0
  },
  "edit_lost_tokens": 0,
  "source_fit_diagnostic": {
    "RAW": {
      "KL": 6.490120285086631e-08,
      "lost_correct_tokens": 3
    },
    "BOUNDED": {
      "KL": 2.2871484079303396e-08,
      "lost_correct_tokens": 1
    },
    "MATCHED_RAW": {
      "KL": 6.490120285086631e-08,
      "lost_correct_tokens": 3
    }
  },
  "end_to_end_from_original_Base": {
    "RAW": {
      "KL": 1.904218873576778e-06,
      "lost_correct_tokens": 24
    },
    "BOUNDED": {
      "KL": 1.8712408733826767e-06,
      "lost_correct_tokens": 24
    },
    "MATCHED_RAW": {
      "KL": 1.904218873576778e-06,
      "lost_correct_tokens": 24
    }
  },
  "baseline_numerical_migration": {
    "KL": 1.8418112357528603e-06,
    "lost_correct_tokens": 24
  },
  "all96_secondary": {
    "RAW": {
      "KL": 6.528266201845126e-08,
      "lost_correct_tokens": 0
    },
    "BOUNDED": {
      "KL": 3.100265220708314e-08,
      "lost_correct_tokens": 0
    },
    "MATCHED_RAW": {
      "KL": 6.528266201845126e-08,
      "lost_correct_tokens": 0
    }
  },
  "per_expert": [
    {
      "expert_order": 1,
      "edit_progress": {
        "RAW": 0.006208441463738012,
        "BOUNDED": 0.005587739815853281,
        "MATCHED_RAW": 0.006208441463738012
      },
      "actual_edit_retention": 0.900022952377002,
      "source_gate_passed": true,
      "edit_lost_tokens": {
        "RAW": 0,
        "BOUNDED": 0,
        "MATCHED_RAW": 0
      },
      "geometry": {
        "norm": 1.0,
        "gain": 0.8997604457251918,
        "required": 0.8997604457251918,
        "norm_multiplier": 0.02487650652668357,
        "KKT_residual": 1.7430662711673162e-16,
        "metric_condition": 520.3950150109864,
        "rank": 512,
        "objective": 3.2697015126891246e-08,
        "algorithm": "BOUNDED_FULL_VOCABULARY_FISHER_SKETCH",
        "raw_linear_edit_progress": 0.006200620510041874,
        "actual_linear_edit_progress": 0.005580558466293534,
        "RAW_map_norm": 0.022627019260093605,
        "BOUNDED_map_norm": 0.022609401901305986,
        "objective_RAW": 1.610181902539585e-07,
        "objective_BOUNDED": 3.2697015174937344e-08,
        "source_questions": 61,
        "source_tokens": 1514,
        "proxy_is_full_KL": false,
        "proxy_is_KL_curvature_estimate": true,
        "probes_per_question": 32,
        "initial_actual_edit_gain": 0.005582719667107549,
        "final_actual_edit_gain": 0.005587739815853281,
        "RAW_actual_edit_gain": 0.006208441463738012,
        "restoration_alpha": 0.0077972412109375,
        "restoration_trace": [
          {
            "alpha": 0.0,
            "gain": 0.005582719667107549
          },
          {
            "alpha": 1.0,
            "gain": 0.006208441463738012
          },
          {
            "alpha": 0.5,
            "gain": 0.005896404859473673
          },
          {
            "alpha": 0.25,
            "gain": 0.005741597953836817
          },
          {
            "alpha": 0.125,
            "gain": 0.0056629272339961765
          },
          {
            "alpha": 0.0625,
            "gain": 0.005622593407119741
          },
          {
            "alpha": 0.03125,
            "gain": 0.005602400832059101
          },
          {
            "alpha": 0.015625,
            "gain": 0.005592215284005281
          },
          {
            "alpha": 0.0078125,
            "gain": 0.005588135706678101
          },
          {
            "alpha": 0.00390625,
            "gain": 0.005585418545482071
          },
          {
            "alpha": 0.005859375,
            "gain": 0.005585798172342871
          },
          {
            "alpha": 0.0068359375,
            "gain": 0.005586378832743297
          },
          {
            "alpha": 0.00732421875,
            "gain": 0.005586694312610415
          },
          {
            "alpha": 0.007568359375,
            "gain": 0.005586683322117716
          },
          {
            "alpha": 0.0076904296875,
            "gain": 0.005587128389642316
          },
          {
            "alpha": 0.00775146484375,
            "gain": 0.005586904045375342
          },
          {
            "alpha": 0.007781982421875,
            "gain": 0.005587216144397356
          },
          {
            "alpha": 0.0077972412109375,
            "gain": 0.005587739815853281
          }
        ],
        "restoration_evaluations": 16,
        "objective_BOUNDED_final": 3.298168046216014e-08
      },
      "matching": {
        "scale": 1.0,
        "target": 0.022609401901305986,
        "actual": 0.022627019260093605,
        "relative_error": 0.0007792049902302396,
        "evaluations": 1
      },
      "basis_audit": {
        "rows": 1952,
        "questions": 61,
        "tokens": 1514,
        "probes_per_question": 32,
        "probe_seed_rule": "20261009+original_BASIS_question_index",
        "probe_resampling": false,
        "maximum_centering_error": 9.387731552124023e-07,
        "mean_gradient_checks": 0,
        "full_vocabulary": true,
        "metric": "FINITE_FISHER_SKETCH_NOT_EXACT_KL",
        "zero_rows": 0
      },
      "heldout_KL": {
        "RAW": 7.416073436867695e-08,
        "BOUNDED": 2.7065510449540398e-08,
        "MATCHED_RAW": 7.416073436867695e-08
      }
    },
    {
      "expert_order": 2,
      "edit_progress": {
        "RAW": 0.006606933957993619,
        "BOUNDED": 0.005947355540644901,
        "MATCHED_RAW": 0.006606933957993619
      },
      "actual_edit_retention": 0.900168758830909,
      "source_gate_passed": true,
      "edit_lost_tokens": {
        "RAW": 0,
        "BOUNDED": 0,
        "MATCHED_RAW": 0
      },
      "geometry": {
        "norm": 1.0,
        "gain": 0.8995500702249923,
        "required": 0.8995500702249926,
        "norm_multiplier": 0.022004400743630796,
        "KKT_residual": 1.5822367506755303e-16,
        "metric_condition": 524.9475606484995,
        "rank": 512,
        "objective": 3.002610742917958e-08,
        "algorithm": "BOUNDED_FULL_VOCABULARY_FISHER_SKETCH",
        "raw_linear_edit_progress": 0.006607677157875762,
        "actual_linear_edit_progress": 0.005946909463813309,
        "RAW_map_norm": 0.022627285230327466,
        "BOUNDED_map_norm": 0.02262728529844544,
        "objective_RAW": 1.1719289761783118e-07,
        "objective_BOUNDED": 3.002610772284402e-08,
        "source_questions": 61,
        "source_tokens": 1514,
        "proxy_is_full_KL": false,
        "proxy_is_KL_curvature_estimate": true,
        "probes_per_question": 32,
        "initial_actual_edit_gain": 0.005947355540644901,
        "final_actual_edit_gain": 0.005947355540644901,
        "RAW_actual_edit_gain": 0.006606933957993619,
        "restoration_alpha": 0.0,
        "restoration_trace": [
          {
            "alpha": 0.0,
            "gain": 0.005947355540644901
          },
          {
            "alpha": 1.0,
            "gain": 0.006606933957993619
          }
        ],
        "restoration_evaluations": 0,
        "objective_BOUNDED_final": 3.002610772284402e-08
      },
      "matching": {
        "scale": 1.0,
        "target": 0.02262728529844544,
        "actual": 0.022627285230327466,
        "relative_error": 3.0104350640947667e-09,
        "evaluations": 1
      },
      "basis_audit": {
        "rows": 1952,
        "questions": 61,
        "tokens": 1514,
        "probes_per_question": 32,
        "probe_seed_rule": "20261009+original_BASIS_question_index",
        "probe_resampling": false,
        "maximum_centering_error": 9.387731552124023e-07,
        "mean_gradient_checks": 0,
        "full_vocabulary": true,
        "metric": "FINITE_FISHER_SKETCH_NOT_EXACT_KL",
        "zero_rows": 0
      },
      "heldout_KL": {
        "RAW": 5.9371891579060005e-08,
        "BOUNDED": 2.72394155962282e-08,
        "MATCHED_RAW": 5.9371891579060005e-08
      }
    },
    {
      "expert_order": 3,
      "edit_progress": {
        "RAW": 0.003951277135675288,
        "BOUNDED": 0.003557622233744593,
        "MATCHED_RAW": 0.003951277135675288
      },
      "actual_edit_retention": 0.9003727431881545,
      "source_gate_passed": true,
      "edit_lost_tokens": {
        "RAW": 0,
        "BOUNDED": 0,
        "MATCHED_RAW": 0
      },
      "geometry": {
        "norm": 1.0,
        "gain": 0.8987434570812365,
        "required": 0.8987434570812365,
        "norm_multiplier": 0.028237003798668777,
        "KKT_residual": 2.248476639409753e-16,
        "metric_condition": 389.53166581227504,
        "rank": 512,
        "objective": 2.9478161291811045e-08,
        "algorithm": "BOUNDED_FULL_VOCABULARY_FISHER_SKETCH",
        "raw_linear_edit_progress": 0.003951314857955426,
        "actual_linear_edit_progress": 0.003556183358679907,
        "RAW_map_norm": 0.022627282982497342,
        "BOUNDED_map_norm": 0.02262728289500369,
        "objective_RAW": 9.933808160942239e-08,
        "objective_BOUNDED": 2.9478161075571693e-08,
        "source_questions": 61,
        "source_tokens": 1514,
        "proxy_is_full_KL": false,
        "proxy_is_KL_curvature_estimate": true,
        "probes_per_question": 32,
        "initial_actual_edit_gain": 0.003557622233744593,
        "final_actual_edit_gain": 0.003557622233744593,
        "RAW_actual_edit_gain": 0.003951277135675288,
        "restoration_alpha": 0.0,
        "restoration_trace": [
          {
            "alpha": 0.0,
            "gain": 0.003557622233744593
          },
          {
            "alpha": 1.0,
            "gain": 0.003951277135675288
          }
        ],
        "restoration_evaluations": 0,
        "objective_BOUNDED_final": 2.9478161075571693e-08
      },
      "matching": {
        "scale": 1.0,
        "target": 0.02262728289500369,
        "actual": 0.022627282982497342,
        "relative_error": 3.866732546676467e-09,
        "evaluations": 1
      },
      "basis_audit": {
        "rows": 1952,
        "questions": 61,
        "tokens": 1514,
        "probes_per_question": 32,
        "probe_seed_rule": "20261009+original_BASIS_question_index",
        "probe_resampling": false,
        "maximum_centering_error": 9.387731552124023e-07,
        "mean_gradient_checks": 0,
        "full_vocabulary": true,
        "metric": "FINITE_FISHER_SKETCH_NOT_EXACT_KL",
        "zero_rows": 0
      },
      "heldout_KL": {
        "RAW": 5.189484264064514e-08,
        "BOUNDED": 2.4507322883568674e-08,
        "MATCHED_RAW": 5.189484264064514e-08
      }
    },
    {
      "expert_order": 4,
      "edit_progress": {
        "RAW": 0.008228229415253805,
        "BOUNDED": 0.0074063175079099475,
        "MATCHED_RAW": 0.008228229415253805
      },
      "actual_edit_retention": 0.9001107205615626,
      "source_gate_passed": true,
      "edit_lost_tokens": {
        "RAW": 0,
        "BOUNDED": 0,
        "MATCHED_RAW": 0
      },
      "geometry": {
        "norm": 1.0,
        "gain": 0.8993082713227822,
        "required": 0.8993082713227825,
        "norm_multiplier": 0.02102268434602598,
        "KKT_residual": 1.7147472983629198e-16,
        "metric_condition": 515.6818689513277,
        "rank": 512,
        "objective": 2.8581362886963046e-08,
        "algorithm": "BOUNDED_FULL_VOCABULARY_FISHER_SKETCH",
        "raw_linear_edit_progress": 0.008223961321324679,
        "actual_linear_edit_progress": 0.007401565173539676,
        "RAW_map_norm": 0.022627333019163438,
        "BOUNDED_map_norm": 0.022622716311783066,
        "objective_RAW": 1.3883952377849465e-07,
        "objective_BOUNDED": 2.858136284476428e-08,
        "source_questions": 61,
        "source_tokens": 1514,
        "proxy_is_full_KL": false,
        "proxy_is_KL_curvature_estimate": true,
        "probes_per_question": 32,
        "initial_actual_edit_gain": 0.007403754345090951,
        "final_actual_edit_gain": 0.0074063175079099475,
        "RAW_actual_edit_gain": 0.008228229415253805,
        "restoration_alpha": 0.0020294189453125,
        "restoration_trace": [
          {
            "alpha": 0.0,
            "gain": 0.007403754345090951
          },
          {
            "alpha": 1.0,
            "gain": 0.008228229415253805
          },
          {
            "alpha": 0.5,
            "gain": 0.00781815354038773
          },
          {
            "alpha": 0.25,
            "gain": 0.007607427036133114
          },
          {
            "alpha": 0.125,
            "gain": 0.007503145827396998
          },
          {
            "alpha": 0.0625,
            "gain": 0.007453140965795662
          },
          {
            "alpha": 0.03125,
            "gain": 0.007427916466672974
          },
          {
            "alpha": 0.015625,
            "gain": 0.007415089815258375
          },
          {
            "alpha": 0.0078125,
            "gain": 0.0074122193986582
          },
          {
            "alpha": 0.00390625,
            "gain": 0.0074077956846521865
          },
          {
            "alpha": 0.001953125,
            "gain": 0.007404733057964475
          },
          {
            "alpha": 0.0029296875,
            "gain": 0.007407234988072964
          },
          {
            "alpha": 0.00244140625,
            "gain": 0.007406546744906083
          },
          {
            "alpha": 0.002197265625,
            "gain": 0.007405782046893569
          },
          {
            "alpha": 0.0020751953125,
            "gain": 0.00740615818301534
          },
          {
            "alpha": 0.00201416015625,
            "gain": 0.007405135047376011
          },
          {
            "alpha": 0.002044677734375,
            "gain": 0.007406160290598151
          },
          {
            "alpha": 0.0020294189453125,
            "gain": 0.0074063175079099475
          }
        ],
        "restoration_evaluations": 16,
        "objective_BOUNDED_final": 2.8643177466753734e-08
      },
      "matching": {
        "scale": 1.0,
        "target": 0.022622716311783066,
        "actual": 0.022627333019163438,
        "relative_error": 0.00020407396338906325,
        "evaluations": 1
      },
      "basis_audit": {
        "rows": 1952,
        "questions": 61,
        "tokens": 1514,
        "probes_per_question": 32,
        "probe_seed_rule": "20261009+original_BASIS_question_index",
        "probe_resampling": false,
        "maximum_centering_error": 9.387731552124023e-07,
        "mean_gradient_checks": 0,
        "full_vocabulary": true,
        "metric": "FINITE_FISHER_SKETCH_NOT_EXACT_KL",
        "zero_rows": 0
      },
      "heldout_KL": {
        "RAW": 6.653661522966303e-08,
        "BOUNDED": 2.6966380125498562e-08,
        "MATCHED_RAW": 6.653661522966303e-08
      }
    },
    {
      "expert_order": 5,
      "edit_progress": {
        "RAW": 0.0032651443564819726,
        "BOUNDED": 0.002939826380864326,
        "MATCHED_RAW": 0.0032651443564819726
      },
      "actual_edit_retention": 0.9003664340378016,
      "source_gate_passed": true,
      "edit_lost_tokens": {
        "RAW": 0,
        "BOUNDED": 0,
        "MATCHED_RAW": 0
      },
      "geometry": {
        "norm": 1.0,
        "gain": 0.8987222291183652,
        "required": 0.8987222291183654,
        "norm_multiplier": 0.02234056527260645,
        "KKT_residual": 1.7043176628101983e-16,
        "metric_condition": 537.8340565968307,
        "rank": 512,
        "objective": 3.146699857129345e-08,
        "algorithm": "BOUNDED_FULL_VOCABULARY_FISHER_SKETCH",
        "raw_linear_edit_progress": 0.003271782184664994,
        "actual_linear_edit_progress": 0.0029446039760080006,
        "RAW_map_norm": 0.02262695324655926,
        "BOUNDED_map_norm": 0.022626953330626476,
        "objective_RAW": 1.1626770817745656e-07,
        "objective_BOUNDED": 3.146699874066789e-08,
        "source_questions": 61,
        "source_tokens": 1514,
        "proxy_is_full_KL": false,
        "proxy_is_KL_curvature_estimate": true,
        "probes_per_question": 32,
        "initial_actual_edit_gain": 0.002939826380864326,
        "final_actual_edit_gain": 0.002939826380864326,
        "RAW_actual_edit_gain": 0.0032651443564819726,
        "restoration_alpha": 0.0,
        "restoration_trace": [
          {
            "alpha": 0.0,
            "gain": 0.002939826380864326
          },
          {
            "alpha": 1.0,
            "gain": 0.0032651443564819726
          }
        ],
        "restoration_evaluations": 0,
        "objective_BOUNDED_final": 3.146699874066789e-08
      },
      "matching": {
        "scale": 1.0,
        "target": 0.022626953330626476,
        "actual": 0.02262695324655926,
        "relative_error": 3.7153572452499797e-09,
        "evaluations": 1
      },
      "basis_audit": {
        "rows": 1952,
        "questions": 61,
        "tokens": 1514,
        "probes_per_question": 32,
        "probe_seed_rule": "20261009+original_BASIS_question_index",
        "probe_resampling": false,
        "maximum_centering_error": 9.387731552124023e-07,
        "mean_gradient_checks": 0,
        "full_vocabulary": true,
        "metric": "FINITE_FISHER_SKETCH_NOT_EXACT_KL",
        "zero_rows": 0
      },
      "heldout_KL": {
        "RAW": 5.437686071534371e-08,
        "BOUNDED": 2.3702791441512698e-08,
        "MATCHED_RAW": 5.437686071534371e-08
      }
    },
    {
      "expert_order": 6,
      "edit_progress": {
        "RAW": 0.004739030202710306,
        "BOUNDED": 0.004267716298459263,
        "MATCHED_RAW": 0.004739030202710306
      },
      "actual_edit_retention": 0.9005463387885789,
      "source_gate_passed": true,
      "edit_lost_tokens": {
        "RAW": 0,
        "BOUNDED": 0,
        "MATCHED_RAW": 0
      },
      "geometry": {
        "norm": 1.0,
        "gain": 0.8996551727780002,
        "required": 0.8996551727780002,
        "norm_multiplier": 0.028577851741033323,
        "KKT_residual": 2.064829837324778e-16,
        "metric_condition": 470.33219994206013,
        "rank": 512,
        "objective": 3.4458461356777995e-08,
        "algorithm": "BOUNDED_FULL_VOCABULARY_FISHER_SKETCH",
        "raw_linear_edit_progress": 0.004749664951751735,
        "actual_linear_edit_progress": 0.004274698463269205,
        "RAW_map_norm": 0.022627279075574387,
        "BOUNDED_map_norm": 0.022627279099715455,
        "objective_RAW": 1.4281882630677996e-07,
        "objective_BOUNDED": 3.4458461533440004e-08,
        "source_questions": 61,
        "source_tokens": 1514,
        "proxy_is_full_KL": false,
        "proxy_is_KL_curvature_estimate": true,
        "probes_per_question": 32,
        "initial_actual_edit_gain": 0.004267716298459263,
        "final_actual_edit_gain": 0.004267716298459263,
        "RAW_actual_edit_gain": 0.004739030202710306,
        "restoration_alpha": 0.0,
        "restoration_trace": [
          {
            "alpha": 0.0,
            "gain": 0.004267716298459263
          },
          {
            "alpha": 1.0,
            "gain": 0.004739030202710306
          }
        ],
        "restoration_evaluations": 0,
        "objective_BOUNDED_final": 3.4458461533440004e-08
      },
      "matching": {
        "scale": 1.0,
        "target": 0.022627279099715455,
        "actual": 0.022627279075574387,
        "relative_error": 1.0669010405882495e-09,
        "evaluations": 1
      },
      "basis_audit": {
        "rows": 1952,
        "questions": 61,
        "tokens": 1514,
        "probes_per_question": 32,
        "probe_seed_rule": "20261009+original_BASIS_question_index",
        "probe_resampling": false,
        "maximum_centering_error": 9.387731552124023e-07,
        "mean_gradient_checks": 0,
        "full_vocabulary": true,
        "metric": "FINITE_FISHER_SKETCH_NOT_EXACT_KL",
        "zero_rows": 0
      },
      "heldout_KL": {
        "RAW": 6.734632593169907e-08,
        "BOUNDED": 3.041563252821991e-08,
        "MATCHED_RAW": 6.734632593169907e-08
      }
    },
    {
      "expert_order": 7,
      "edit_progress": {
        "RAW": 0.011041856759218517,
        "BOUNDED": 0.009938085700398605,
        "MATCHED_RAW": 0.011041856759218517
      },
      "actual_edit_retention": 0.9000375495816493,
      "source_gate_passed": true,
      "edit_lost_tokens": {
        "RAW": 0,
        "BOUNDED": 0,
        "MATCHED_RAW": 0
      },
      "geometry": {
        "norm": 1.0,
        "gain": 0.8997908210180974,
        "required": 0.8997908210180975,
        "norm_multiplier": 0.02239126127301605,
        "KKT_residual": 1.7472394985616434e-16,
        "metric_condition": 480.482045458795,
        "rank": 512,
        "objective": 3.507082029699833e-08,
        "algorithm": "BOUNDED_FULL_VOCABULARY_FISHER_SKETCH",
        "raw_linear_edit_progress": 0.011069881268660568,
        "actual_linear_edit_progress": 0.009962893131141883,
        "RAW_map_norm": 0.022627311275179973,
        "BOUNDED_map_norm": 0.022625431417039744,
        "objective_RAW": 1.3047698691451991e-07,
        "objective_BOUNDED": 3.5070820213504004e-08,
        "source_questions": 61,
        "source_tokens": 1514,
        "proxy_is_full_KL": false,
        "proxy_is_KL_curvature_estimate": true,
        "probes_per_question": 32,
        "initial_actual_edit_gain": 0.009936884048378223,
        "final_actual_edit_gain": 0.009938085700398605,
        "RAW_actual_edit_gain": 0.011041856759218517,
        "restoration_alpha": 0.000823974609375,
        "restoration_trace": [
          {
            "alpha": 0.0,
            "gain": 0.009936884048378223
          },
          {
            "alpha": 1.0,
            "gain": 0.011041856759218517
          },
          {
            "alpha": 0.5,
            "gain": 0.010489478814302528
          },
          {
            "alpha": 0.25,
            "gain": 0.010211878238843296
          },
          {
            "alpha": 0.125,
            "gain": 0.010073534506725344
          },
          {
            "alpha": 0.0625,
            "gain": 0.010006229978990035
          },
          {
            "alpha": 0.03125,
            "gain": 0.009971863552691038
          },
          {
            "alpha": 0.015625,
            "gain": 0.009953998392861133
          },
          {
            "alpha": 0.0078125,
            "gain": 0.009945348526249717
          },
          {
            "alpha": 0.00390625,
            "gain": 0.009940669182647516
          },
          {
            "alpha": 0.001953125,
            "gain": 0.009938707842465202
          },
          {
            "alpha": 0.0009765625,
            "gain": 0.0099380411041468
          },
          {
            "alpha": 0.00048828125,
            "gain": 0.009937201529439152
          },
          {
            "alpha": 0.000732421875,
            "gain": 0.009937535689047412
          },
          {
            "alpha": 0.0008544921875,
            "gain": 0.009937685821781577
          },
          {
            "alpha": 0.00079345703125,
            "gain": 0.009937015401754876
          },
          {
            "alpha": 0.000823974609375,
            "gain": 0.009938085700398605
          },
          {
            "alpha": 0.0008087158203125,
            "gain": 0.009937305546307373
          }
        ],
        "restoration_evaluations": 16,
        "objective_BOUNDED_final": 3.509735437805633e-08
      },
      "matching": {
        "scale": 1.0,
        "target": 0.022625431417039744,
        "actual": 0.022627311275179973,
        "relative_error": 8.308606830864757e-05,
        "evaluations": 1
      },
      "basis_audit": {
        "rows": 1952,
        "questions": 61,
        "tokens": 1514,
        "probes_per_question": 32,
        "probe_seed_rule": "20261009+original_BASIS_question_index",
        "probe_resampling": false,
        "maximum_centering_error": 9.387731552124023e-07,
        "mean_gradient_checks": 0,
        "full_vocabulary": true,
        "metric": "FINITE_FISHER_SKETCH_NOT_EXACT_KL",
        "zero_rows": 0
      },
      "heldout_KL": {
        "RAW": 6.522101542656865e-08,
        "BOUNDED": 3.130178226106963e-08,
        "MATCHED_RAW": 6.522101542656865e-08
      }
    },
    {
      "expert_order": 8,
      "edit_progress": {
        "RAW": 0.005059387286018937,
        "BOUNDED": 0.004554100329424831,
        "MATCHED_RAW": 0.005059387286018937
      },
      "actual_edit_retention": 0.9001288242964892,
      "source_gate_passed": true,
      "edit_lost_tokens": {
        "RAW": 0,
        "BOUNDED": 0,
        "MATCHED_RAW": 0
      },
      "geometry": {
        "norm": 0.9999999999999999,
        "gain": 0.8998489007678471,
        "required": 0.8998489007678473,
        "norm_multiplier": 0.0316104214594556,
        "KKT_residual": 2.396829809184875e-16,
        "metric_condition": 386.82105520302525,
        "rank": 512,
        "objective": 2.7358358855274374e-08,
        "algorithm": "BOUNDED_FULL_VOCABULARY_FISHER_SKETCH",
        "raw_linear_edit_progress": 0.005058419730371017,
        "actual_linear_edit_progress": 0.004552577758446132,
        "RAW_map_norm": 0.02262724217373409,
        "BOUNDED_map_norm": 0.022624858124019518,
        "objective_RAW": 1.1708341606997209e-07,
        "objective_BOUNDED": 2.7358358948823213e-08,
        "source_questions": 61,
        "source_tokens": 1514,
        "proxy_is_full_KL": false,
        "proxy_is_KL_curvature_estimate": true,
        "probes_per_question": 32,
        "initial_actual_edit_gain": 0.004553022676640041,
        "final_actual_edit_gain": 0.004554100329424831,
        "RAW_actual_edit_gain": 0.005059387286018937,
        "restoration_alpha": 0.0010528564453125,
        "restoration_trace": [
          {
            "alpha": 0.0,
            "gain": 0.004553022676640041
          },
          {
            "alpha": 1.0,
            "gain": 0.005059387286018937
          },
          {
            "alpha": 0.5,
            "gain": 0.004805589206321213
          },
          {
            "alpha": 0.25,
            "gain": 0.004680377405959696
          },
          {
            "alpha": 0.125,
            "gain": 0.004614906357827997
          },
          {
            "alpha": 0.0625,
            "gain": 0.004583886625779801
          },
          {
            "alpha": 0.03125,
            "gain": 0.0045686629104126795
          },
          {
            "alpha": 0.015625,
            "gain": 0.004560514704311985
          },
          {
            "alpha": 0.0078125,
            "gain": 0.004556775290871404
          },
          {
            "alpha": 0.00390625,
            "gain": 0.0045554265576455075
          },
          {
            "alpha": 0.001953125,
            "gain": 0.004554103012065951
          },
          {
            "alpha": 0.0009765625,
            "gain": 0.004553207553331463
          },
          {
            "alpha": 0.00146484375,
            "gain": 0.0045538811921475636
          },
          {
            "alpha": 0.001220703125,
            "gain": 0.00455432979820559
          },
          {
            "alpha": 0.0010986328125,
            "gain": 0.004553742385819497
          },
          {
            "alpha": 0.00103759765625,
            "gain": 0.004553388925093202
          },
          {
            "alpha": 0.001068115234375,
            "gain": 0.004553686323080269
          },
          {
            "alpha": 0.0010528564453125,
            "gain": 0.004554100329424831
          }
        ],
        "restoration_evaluations": 16,
        "objective_BOUNDED_final": 2.738873026950534e-08
      },
      "matching": {
        "scale": 1.0,
        "target": 0.022624858124019518,
        "actual": 0.02262724217373409,
        "relative_error": 0.00010537302384404192,
        "evaluations": 1
      },
      "basis_audit": {
        "rows": 1952,
        "questions": 61,
        "tokens": 1514,
        "probes_per_question": 32,
        "probe_seed_rule": "20261009+original_BASIS_question_index",
        "probe_resampling": false,
        "maximum_centering_error": 9.387731552124023e-07,
        "mean_gradient_checks": 0,
        "full_vocabulary": true,
        "metric": "FINITE_FISHER_SKETCH_NOT_EXACT_KL",
        "zero_rows": 0
      },
      "heldout_KL": {
        "RAW": 6.06441979399682e-08,
        "BOUNDED": 2.702205744767594e-08,
        "MATCHED_RAW": 6.06441979399682e-08
      }
    }
  ],
  "matching_is_map_norm_not_edit_gain": true,
  "free_generation_evaluated": false,
  "independent_confirmation": false,
  "clinical_protection": false,
  "full_training_success": false,
  "resource": {
    "forward_calls": 7528,
    "backward_calls": 15672,
    "new_GPU_process_hours": 1.394693496359719,
    "cumulative_GPU_process_hours": 43.9942884898848,
    "cumulative_Judge": 12412,
    "temporary_updates": 8,
    "new_generations": 0,
    "new_Judge": 0,
    "temporary_candidate_packages": 8
  }
}

原FP32资格失败保留；原63主面板不改为62。KL使用原固定回答前缀，端到端KL单独实测而非相减；无自由生成保护或临床确认结论。
