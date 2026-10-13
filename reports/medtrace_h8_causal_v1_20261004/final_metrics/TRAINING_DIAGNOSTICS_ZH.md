# 训练诊断

[
  {
    "edit_index": 31,
    "arm": "A_NO_H",
    "seconds": 91.36644291877747,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006384457810781896,
            0.0010228046448901296,
            0.0032958632800728083
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            8.532109495718032e-05,
            8.318803156726062e-05,
            0.00033957051346078515
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.00039508979534730315
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            2.0531761038000695e-05,
            2.7365631467546336e-05,
            0.0001099727742257528
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            5.093814979773015e-05,
            5.6003849749686196e-05,
            0.00047466543037444353
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 31,
    "arm": "B_H1",
    "seconds": 124.85599112510681,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840,
      "extra": 2348
    },
    "steps": 320,
    "clip_steps": 98,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006384457810781896,
            0.0010228046448901296,
            0.0032958632800728083,
            40.49167251586914
          ],
          "weighted_extra_to_common_norm_ratio": 11002.064453125,
          "extra_native_fit_cosine": -0.007482205517590046,
          "extra_U_cosine": 0.07899761199951172,
          "extra_common_cosine": 0.06774142384529114,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            1.2175363302230835,
            1.236405849456787,
            0.062410254031419754,
            9.906898498535156
          ],
          "weighted_extra_to_common_norm_ratio": 4.084690093994141,
          "extra_native_fit_cosine": -0.8398638367652893,
          "extra_U_cosine": 0.21907298266887665,
          "extra_common_cosine": -0.8392938375473022,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.008842354640364647,
            0.0033533605746924877,
            0.004140029661357403,
            5.725458145141602
          ],
          "weighted_extra_to_common_norm_ratio": 455.3600769042969,
          "extra_native_fit_cosine": -0.8356391191482544,
          "extra_U_cosine": 0.029059093445539474,
          "extra_common_cosine": -0.7750517129898071,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0007811515242792666,
            0.0007450394914485514,
            0.0012549498351290822,
            0.03590761870145798
          ],
          "weighted_extra_to_common_norm_ratio": 18.545326232910156,
          "extra_native_fit_cosine": 0.0017018979415297508,
          "extra_U_cosine": 0.014477437362074852,
          "extra_common_cosine": 0.010632427409291267,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00023523597337771207,
            0.0001843587087932974,
            0.00029841886134818196,
            0.007961989380419254
          ],
          "weighted_extra_to_common_norm_ratio": 16.379043579101562,
          "extra_native_fit_cosine": 0.009276705794036388,
          "extra_U_cosine": -0.04108066111803055,
          "extra_common_cosine": -0.017921827733516693,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 31,
    "arm": "C_H025",
    "seconds": 126.70797848701477,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840,
      "extra": 2348
    },
    "steps": 320,
    "clip_steps": 98,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006384457810781896,
            0.0010228046448901296,
            0.0032958632800728083,
            10.125155448913574
          ],
          "weighted_extra_to_common_norm_ratio": 2751.1240234375,
          "extra_native_fit_cosine": -0.007464177906513214,
          "extra_U_cosine": 0.07902960479259491,
          "extra_common_cosine": 0.06777730584144592,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.720051646232605,
            0.6961892247200012,
            0.025629743933677673,
            2.9146921634674072
          ],
          "weighted_extra_to_common_norm_ratio": 2.0773134231567383,
          "extra_native_fit_cosine": -0.9031039476394653,
          "extra_U_cosine": 0.22582101821899414,
          "extra_common_cosine": -0.902504563331604,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.009587977081537247,
            0.0043982090428471565,
            0.0035791778936982155,
            3.59590220451355
          ],
          "weighted_extra_to_common_norm_ratio": 252.89505004882812,
          "extra_native_fit_cosine": -0.8356435298919678,
          "extra_U_cosine": 0.04678308963775635,
          "extra_common_cosine": -0.7999570369720459,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0008325687376782298,
            0.0003569454129319638,
            0.0011544672306627035,
            0.057648010551929474
          ],
          "weighted_extra_to_common_norm_ratio": 36.344547271728516,
          "extra_native_fit_cosine": -0.4659704267978668,
          "extra_U_cosine": 0.019834179431200027,
          "extra_common_cosine": -0.2896130383014679,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00024936016416177154,
            0.00020207105262670666,
            0.00035109606687910855,
            0.013193592429161072
          ],
          "weighted_extra_to_common_norm_ratio": 23.78993034362793,
          "extra_native_fit_cosine": 0.026018040254712105,
          "extra_U_cosine": -0.06379424780607224,
          "extra_common_cosine": -0.02053707279264927,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 31,
    "arm": "D_EXTRA_FIT",
    "seconds": 127.22304320335388,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840,
      "extra": 2560
    },
    "steps": 320,
    "clip_steps": 1,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006384457810781896,
            0.0010228046448901296,
            0.0032958632800728083,
            0.008130895905196667
          ],
          "weighted_extra_to_common_norm_ratio": 2.2092602252960205,
          "extra_native_fit_cosine": -0.19753283262252808,
          "extra_U_cosine": -0.08428649604320526,
          "extra_common_cosine": -0.1547584980726242,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0005169576616026461,
            0.0
          ],
          "weighted_extra_to_common_norm_ratio": 0.0,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0002525373420212418,
            0.00011564853775780648
          ],
          "weighted_extra_to_common_norm_ratio": 0.4579463005065918,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": -0.049752164632081985,
          "extra_common_cosine": -0.049752164632081985,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0001111082237912342,
            3.051593193958979e-05,
            0.0008034601341933012,
            0.0015195193700492382
          ],
          "weighted_extra_to_common_norm_ratio": 1.864512324333191,
          "extra_native_fit_cosine": 0.3134414851665497,
          "extra_U_cosine": -0.029310215264558792,
          "extra_common_cosine": 0.018704354763031006,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.001310422201640904,
            0.0012673254823312163,
            0.0015845602611079812,
            0.0019226238364353776
          ],
          "weighted_extra_to_common_norm_ratio": 0.6324459314346313,
          "extra_native_fit_cosine": 0.2615066170692444,
          "extra_U_cosine": 0.0397111177444458,
          "extra_common_cosine": 0.24194228649139404,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 35,
    "arm": "A_NO_H",
    "seconds": 92.24906969070435,
    "tokens": {
      "native": 4160,
      "fit": 4160,
      "U": 3840
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006279443041421473,
            0.0014000653754919767,
            0.0033961895387619734
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            1.8591230173115036e-06,
            3.5462148844089825e-06,
            0.00027415010845288634
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            4.236671520629898e-05,
            3.569809996406548e-05,
            0.00026063929544761777
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            2.070974005619064e-05,
            8.137551049003378e-05,
            0.0006506720674224198
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00010760746954474598,
            4.159942182013765e-05,
            0.0005243645864538848
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 35,
    "arm": "B_H1",
    "seconds": 127.32155466079712,
    "tokens": {
      "native": 4160,
      "fit": 4160,
      "U": 3840,
      "extra": 1600
    },
    "steps": 320,
    "clip_steps": 19,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006279443041421473,
            0.0014000653754919767,
            0.0033961895387619734,
            31.825647354125977
          ],
          "weighted_extra_to_common_norm_ratio": 8511.39453125,
          "extra_native_fit_cosine": 0.020612113177776337,
          "extra_U_cosine": -0.022641807794570923,
          "extra_common_cosine": -0.011459870263934135,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            1.9257349967956543,
            0.8656776547431946,
            0.005924541503190994,
            1.0296012163162231
          ],
          "weighted_extra_to_common_norm_ratio": 0.3702141344547272,
          "extra_native_fit_cosine": -0.9518067240715027,
          "extra_U_cosine": -0.10776958614587784,
          "extra_common_cosine": -0.9518483281135559,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.00044064823305234313,
            0.00013737709377892315,
            0.0016495762392878532,
            0.0011308956891298294
          ],
          "weighted_extra_to_common_norm_ratio": 0.6636802554130554,
          "extra_native_fit_cosine": 0.07580351084470749,
          "extra_U_cosine": -0.036072030663490295,
          "extra_common_cosine": -0.011765412986278534,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0001639724214328453,
            8.487360901199281e-05,
            0.0005960684502497315,
            0.00020328529353719205
          ],
          "weighted_extra_to_common_norm_ratio": 0.32858097553253174,
          "extra_native_fit_cosine": 0.11198577284812927,
          "extra_U_cosine": -0.08915483951568604,
          "extra_common_cosine": -0.04554861783981323,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00016973492165561765,
            0.00011219951556995511,
            0.0005409710574895144,
            0.00020543027494568378
          ],
          "weighted_extra_to_common_norm_ratio": 0.35664835572242737,
          "extra_native_fit_cosine": 0.09840373694896698,
          "extra_U_cosine": -0.06851563602685928,
          "extra_common_cosine": -0.02203582599759102,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 35,
    "arm": "C_H025",
    "seconds": 128.40450620651245,
    "tokens": {
      "native": 4160,
      "fit": 4160,
      "U": 3840,
      "extra": 1600
    },
    "steps": 320,
    "clip_steps": 18,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006279443041421473,
            0.0014000653754919767,
            0.0033961895387619734,
            7.955042362213135
          ],
          "weighted_extra_to_common_norm_ratio": 2127.482421875,
          "extra_native_fit_cosine": 0.020664386451244354,
          "extra_U_cosine": -0.02270977571606636,
          "extra_common_cosine": -0.011498506180942059,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.8780419826507568,
            0.6757792234420776,
            0.004358760081231594,
            1.0002700090408325
          ],
          "weighted_extra_to_common_norm_ratio": 0.645789623260498,
          "extra_native_fit_cosine": -0.9656729102134705,
          "extra_U_cosine": -0.0920480415225029,
          "extra_common_cosine": -0.9657166600227356,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            6.254678010009229e-05,
            7.004884537309408e-05,
            0.0015477677807211876,
            0.0021671203430742025
          ],
          "weighted_extra_to_common_norm_ratio": 1.4042831659317017,
          "extra_native_fit_cosine": 0.026226475834846497,
          "extra_U_cosine": -0.05253029614686966,
          "extra_common_cosine": -0.05063539370894432,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            4.9891998060047626e-05,
            7.396427099592984e-05,
            0.000388847867725417,
            0.0003777528181672096
          ],
          "weighted_extra_to_common_norm_ratio": 0.9311402440071106,
          "extra_native_fit_cosine": 0.04271375387907028,
          "extra_U_cosine": -0.0538935512304306,
          "extra_common_cosine": -0.03938142955303192,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            5.462864646688104e-05,
            8.282409544335678e-05,
            0.0005191079108044505,
            0.00024467051844112575
          ],
          "weighted_extra_to_common_norm_ratio": 0.4622761309146881,
          "extra_native_fit_cosine": 0.0622548833489418,
          "extra_U_cosine": -0.09690525382757187,
          "extra_common_cosine": -0.08056709915399551,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 35,
    "arm": "D_EXTRA_FIT",
    "seconds": 128.356018781662,
    "tokens": {
      "native": 4160,
      "fit": 4160,
      "U": 3840,
      "extra": 4160
    },
    "steps": 320,
    "clip_steps": 1,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006279443041421473,
            0.0014000653754919767,
            0.0033961895387619734,
            0.004274078644812107
          ],
          "weighted_extra_to_common_norm_ratio": 1.1430519819259644,
          "extra_native_fit_cosine": 0.10216732323169708,
          "extra_U_cosine": 0.003137810155749321,
          "extra_common_cosine": 0.04798056185245514,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            1.7402289813617244e-06,
            0.00038738330476917326,
            1.9249982869951054e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.049687281250953674,
          "extra_native_fit_cosine": -0.1553173065185547,
          "extra_U_cosine": -0.00548296794295311,
          "extra_common_cosine": -0.006180065684020519,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            3.98879592466983e-06,
            4.747299499285873e-06,
            0.00043146463576704264,
            5.628269354929216e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.13033977150917053,
          "extra_native_fit_cosine": 0.33254265785217285,
          "extra_U_cosine": -0.08338839560747147,
          "extra_common_cosine": -0.07702521979808807,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0004717682022601366,
            0.000170853832969442
          ],
          "weighted_extra_to_common_norm_ratio": 0.36215630173683167,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": -0.035038143396377563,
          "extra_common_cosine": -0.035038143396377563,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0003284459817223251,
            8.209572115447372e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.24995197355747223,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": 0.03746725618839264,
          "extra_common_cosine": 0.03746725618839264,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 49,
    "arm": "A_NO_H",
    "seconds": 97.00743293762207,
    "tokens": {
      "native": 2240,
      "fit": 2240,
      "U": 3840
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006072993273846805,
            0.0009772978955879807,
            0.0026283881161361933
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.0002280419139424339,
            0.00019450305262580514,
            0.0002787226694636047
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            1.8293596440344118e-05,
            3.0299246645881794e-05,
            0.00026954541681334376
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            7.378523150691763e-05,
            8.665170753374696e-05,
            0.00039979504072107375
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00011395377805456519,
            0.00011623205500654876,
            0.0006807011086493731
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 49,
    "arm": "B_H1",
    "seconds": 127.16477918624878,
    "tokens": {
      "native": 2240,
      "fit": 2240,
      "U": 3840,
      "extra": 960
    },
    "steps": 320,
    "clip_steps": 7,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006072993273846805,
            0.0009772978955879807,
            0.0026283881161361933,
            47.9374885559082
          ],
          "weighted_extra_to_common_norm_ratio": 16352.54296875,
          "extra_native_fit_cosine": 0.022028185427188873,
          "extra_U_cosine": -0.03023378551006317,
          "extra_common_cosine": -0.01686135306954384,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.01589459553360939,
            0.01215478777885437,
            0.006665455177426338,
            0.003950780723243952
          ],
          "weighted_extra_to_common_norm_ratio": 0.13960930705070496,
          "extra_native_fit_cosine": -0.17665258049964905,
          "extra_U_cosine": 0.19298946857452393,
          "extra_common_cosine": -0.12801137566566467,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0002999267017003149,
            0.00022968986013438553,
            0.0008779103518463671,
            0.0005852367030456662
          ],
          "weighted_extra_to_common_norm_ratio": 0.5573983788490295,
          "extra_native_fit_cosine": 0.10951779782772064,
          "extra_U_cosine": 0.24696114659309387,
          "extra_common_cosine": 0.2578774690628052,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.00019196720677427948,
            0.00015605473890900612,
            0.000399423239286989,
            0.00038736272836104035
          ],
          "weighted_extra_to_common_norm_ratio": 0.7521297335624695,
          "extra_native_fit_cosine": -0.03665599226951599,
          "extra_U_cosine": 0.21608629822731018,
          "extra_common_cosine": 0.1446824073791504,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00015277686179615557,
            0.00015660868666600436,
            0.00022303139849100262,
            0.00024771972675807774
          ],
          "weighted_extra_to_common_norm_ratio": 0.6843324303627014,
          "extra_native_fit_cosine": -0.13314706087112427,
          "extra_U_cosine": 0.08130429685115814,
          "extra_common_cosine": -0.059194594621658325,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 49,
    "arm": "C_H025",
    "seconds": 127.00456023216248,
    "tokens": {
      "native": 2240,
      "fit": 2240,
      "U": 3840,
      "extra": 960
    },
    "steps": 320,
    "clip_steps": 4,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006072993273846805,
            0.0009772978955879807,
            0.0026283881161361933,
            11.98373031616211
          ],
          "weighted_extra_to_common_norm_ratio": 4087.916748046875,
          "extra_native_fit_cosine": 0.021964240819215775,
          "extra_U_cosine": -0.0301900003105402,
          "extra_common_cosine": -0.016851838678121567,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.006010815501213074,
            0.005323747172951698,
            0.008524412289261818,
            0.0035877691116183996
          ],
          "weighted_extra_to_common_norm_ratio": 0.2522394061088562,
          "extra_native_fit_cosine": -0.1428024023771286,
          "extra_U_cosine": 0.28079670667648315,
          "extra_common_cosine": 0.05716710537672043,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.00033243754296563566,
            0.00025236885994672775,
            0.0009138418245129287,
            0.000710042892023921
          ],
          "weighted_extra_to_common_norm_ratio": 0.6088491678237915,
          "extra_native_fit_cosine": 0.1917690932750702,
          "extra_U_cosine": 0.2802491784095764,
          "extra_common_cosine": 0.3115832209587097,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0002048392198048532,
            0.00015980419993866235,
            0.00047850378905422986,
            0.00037041204632259905
          ],
          "weighted_extra_to_common_norm_ratio": 0.6199465394020081,
          "extra_native_fit_cosine": -0.07163199782371521,
          "extra_U_cosine": 0.29352867603302,
          "extra_common_cosine": 0.19389602541923523,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00013539026258513331,
            0.00013593312178272754,
            0.00019348783825989813,
            0.00024069705978035927
          ],
          "weighted_extra_to_common_norm_ratio": 0.7642403244972229,
          "extra_native_fit_cosine": -0.19788889586925507,
          "extra_U_cosine": 0.08411610871553421,
          "extra_common_cosine": -0.10825011879205704,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 49,
    "arm": "D_EXTRA_FIT",
    "seconds": 127.21008682250977,
    "tokens": {
      "native": 2240,
      "fit": 2240,
      "U": 3840,
      "extra": 2240
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006072993273846805,
            0.0009772978955879807,
            0.0026283881161361933,
            0.0029808764811605215
          ],
          "weighted_extra_to_common_norm_ratio": 1.0168431997299194,
          "extra_native_fit_cosine": 0.29971691966056824,
          "extra_U_cosine": 0.0730353593826294,
          "extra_common_cosine": 0.20489554107189178,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            6.446133193094283e-05,
            4.466380050871521e-05,
            0.00027993039111606777,
            0.0009781864937394857
          ],
          "weighted_extra_to_common_norm_ratio": 3.285040855407715,
          "extra_native_fit_cosine": 0.0022661834955215454,
          "extra_U_cosine": 0.0018039809074252844,
          "extra_common_cosine": 0.0024598916061222553,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            7.871416164562106e-05,
            1.7657519492786378e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.2243245542049408,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": -0.06959021091461182,
          "extra_common_cosine": -0.06959021091461182,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0003840235585812479,
            3.4615535696502775e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.09013909101486206,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": -0.2043166607618332,
          "extra_common_cosine": -0.2043166607618332,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            2.2312879082164727e-06,
            0.0,
            0.00035420272615738213,
            0.00012206162500660866
          ],
          "weighted_extra_to_common_norm_ratio": 0.34474536776542664,
          "extra_native_fit_cosine": 0.08157167583703995,
          "extra_U_cosine": -0.09962864965200424,
          "extra_common_cosine": -0.09915386140346527,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 71,
    "arm": "A_NO_H",
    "seconds": 97.70550227165222,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.000725036661606282,
            0.0008203786564990878,
            0.015270430594682693
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0007601127726957202
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            2.5627796276239678e-05,
            0.00011263225314905867
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            3.601564822020009e-05,
            6.437983392970636e-05
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0003582992358133197
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 71,
    "arm": "B_H1",
    "seconds": 128.32410287857056,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840,
      "extra": 2348
    },
    "steps": 320,
    "clip_steps": 122,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.000725036661606282,
            0.0008203786564990878,
            0.015270430594682693,
            15.437907218933105
          ],
          "weighted_extra_to_common_norm_ratio": 1004.9776000976562,
          "extra_native_fit_cosine": 0.002757510170340538,
          "extra_U_cosine": 0.06451372802257538,
          "extra_common_cosine": 0.06436154991388321,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.13772615790367126,
            0.15468904376029968,
            0.019898075610399246,
            6.3205366134643555
          ],
          "weighted_extra_to_common_norm_ratio": 21.59009552001953,
          "extra_native_fit_cosine": -0.5212366580963135,
          "extra_U_cosine": 0.022126469761133194,
          "extra_common_cosine": -0.5173583626747131,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            1.2252483367919922,
            0.5919135212898254,
            0.004989168141037226,
            1.3602066040039062
          ],
          "weighted_extra_to_common_norm_ratio": 0.7572556734085083,
          "extra_native_fit_cosine": -0.9571257829666138,
          "extra_U_cosine": -0.012325156480073929,
          "extra_common_cosine": -0.957133948802948,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0009060711599886417,
            0.0014206365449354053,
            0.002049720846116543,
            0.049877580255270004
          ],
          "weighted_extra_to_common_norm_ratio": 16.133241653442383,
          "extra_native_fit_cosine": 0.02018202468752861,
          "extra_U_cosine": -0.01262160949409008,
          "extra_common_cosine": 0.005705642513930798,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.0001039177441271022,
            0.000409155705710873,
            0.00033177912700921297,
            0.012090284377336502
          ],
          "weighted_extra_to_common_norm_ratio": 21.340673446655273,
          "extra_native_fit_cosine": 0.013163816183805466,
          "extra_U_cosine": -0.02937971241772175,
          "extra_common_cosine": -0.0065574562177062035,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 71,
    "arm": "C_H025",
    "seconds": 128.2666938304901,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840,
      "extra": 2348
    },
    "steps": 320,
    "clip_steps": 92,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.000725036661606282,
            0.0008203786564990878,
            0.015270430594682693,
            3.8611862659454346
          ],
          "weighted_extra_to_common_norm_ratio": 251.35568237304688,
          "extra_native_fit_cosine": 0.002830682322382927,
          "extra_U_cosine": 0.06453091651201248,
          "extra_common_cosine": 0.06438472867012024,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.31552785634994507,
            0.33020246028900146,
            0.10499885678291321,
            1.6487795114517212
          ],
          "weighted_extra_to_common_norm_ratio": 2.506788969039917,
          "extra_native_fit_cosine": -0.5954160690307617,
          "extra_U_cosine": -0.007630460895597935,
          "extra_common_cosine": -0.5849460363388062,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.7670689225196838,
            0.1514233946800232,
            0.0018136976286768913,
            0.44077086448669434
          ],
          "weighted_extra_to_common_norm_ratio": 0.48379194736480713,
          "extra_native_fit_cosine": -0.9527764320373535,
          "extra_U_cosine": 0.009549511596560478,
          "extra_common_cosine": -0.9528094530105591,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.00023162300931289792,
            0.0019257055828347802,
            0.0008871325408108532,
            0.18595030903816223
          ],
          "weighted_extra_to_common_norm_ratio": 84.77127838134766,
          "extra_native_fit_cosine": 0.02080189809203148,
          "extra_U_cosine": 0.003055369947105646,
          "extra_common_cosine": 0.019799482077360153,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            3.4789172786986455e-05,
            3.0180313842720352e-05,
            0.0003104954957962036,
            0.01768529787659645
          ],
          "weighted_extra_to_common_norm_ratio": 57.42523956298828,
          "extra_native_fit_cosine": 0.0026278099976480007,
          "extra_U_cosine": -0.009553266689181328,
          "extra_common_cosine": -0.009093225002288818,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 71,
    "arm": "D_EXTRA_FIT",
    "seconds": 128.1607768535614,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840,
      "extra": 2560
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.000725036661606282,
            0.0008203786564990878,
            0.015270430594682693,
            0.00780330877751112
          ],
          "weighted_extra_to_common_norm_ratio": 0.5079801678657532,
          "extra_native_fit_cosine": -0.04688086360692978,
          "extra_U_cosine": -0.2061012089252472,
          "extra_common_cosine": -0.2087913304567337,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0004127717693336308,
            0.0
          ],
          "weighted_extra_to_common_norm_ratio": 0.0,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            3.315910362289287e-05,
            0.0002344520908081904,
            3.975702929892577e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.1646827608346939,
          "extra_native_fit_cosine": 0.9626212120056152,
          "extra_U_cosine": 0.14997467398643494,
          "extra_common_cosine": 0.2778671383857727,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            1.9666587832034566e-05,
            0.00032934651244431734,
            2.5679493774077855e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.07790966331958771,
          "extra_native_fit_cosine": 0.6887965202331543,
          "extra_U_cosine": 0.011203953064978123,
          "extra_common_cosine": 0.05229352414608002,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            3.0482931833830662e-05,
            1.4267961887526326e-05,
            0.00038321534520946443,
            0.00026849680580198765
          ],
          "weighted_extra_to_common_norm_ratio": 0.6952880024909973,
          "extra_native_fit_cosine": 0.17489664256572723,
          "extra_U_cosine": -0.02501242607831955,
          "extra_common_cosine": -0.010299083776772022,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 87,
    "arm": "A_NO_H",
    "seconds": 95.53031897544861,
    "tokens": {
      "native": 5440,
      "fit": 5440,
      "U": 3840
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0013027231907472014,
            0.0014288900420069695,
            0.006240000016987324
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.00021434055815916508,
            0.00019619159866124392,
            0.00037105081719346344
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            4.460189666133374e-05,
            6.137890159152448e-05,
            0.00029035029001533985
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.00014900647511240095,
            0.000150039151776582,
            0.0006916980491951108
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.006178378127515316,
            0.007842696271836758,
            0.0004722002486232668
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 87,
    "arm": "B_H1",
    "seconds": 125.6982171535492,
    "tokens": {
      "native": 5440,
      "fit": 5440,
      "U": 3840,
      "extra": 1600
    },
    "steps": 320,
    "clip_steps": 14,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0013027231907472014,
            0.0014288900420069695,
            0.006240000016987324,
            22.580862045288086
          ],
          "weighted_extra_to_common_norm_ratio": 3422.79638671875,
          "extra_native_fit_cosine": -0.09413589537143707,
          "extra_U_cosine": 0.16828322410583496,
          "extra_common_cosine": 0.12774527072906494,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.5405399799346924,
            0.4885494112968445,
            0.020285097882151604,
            1.5013210773468018
          ],
          "weighted_extra_to_common_norm_ratio": 1.466414451599121,
          "extra_native_fit_cosine": -0.9810754656791687,
          "extra_U_cosine": 0.1030525490641594,
          "extra_common_cosine": -0.9803103804588318,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0045755719766020775,
            0.0015964440535753965,
            0.0019722788129001856,
            0.0025025857612490654
          ],
          "weighted_extra_to_common_norm_ratio": 0.4366995394229889,
          "extra_native_fit_cosine": -0.666305422782898,
          "extra_U_cosine": -0.062051963061094284,
          "extra_common_cosine": -0.6328623294830322,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0015987786464393139,
            0.0007827361696399748,
            0.0007159058004617691,
            0.0012930174125358462
          ],
          "weighted_extra_to_common_norm_ratio": 0.636654257774353,
          "extra_native_fit_cosine": -0.6294187307357788,
          "extra_U_cosine": -0.04174378141760826,
          "extra_common_cosine": -0.5997041463851929,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.0010245161829516292,
            0.00046473558177240193,
            0.0003107710799667984,
            0.0007304136524908245
          ],
          "weighted_extra_to_common_norm_ratio": 0.5653287172317505,
          "extra_native_fit_cosine": -0.36431318521499634,
          "extra_U_cosine": 0.15091001987457275,
          "extra_common_cosine": -0.3211939334869385,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 87,
    "arm": "C_H025",
    "seconds": 126.99135780334473,
    "tokens": {
      "native": 5440,
      "fit": 5440,
      "U": 3840,
      "extra": 1600
    },
    "steps": 320,
    "clip_steps": 11,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0013027231907472014,
            0.0014288900420069695,
            0.006240000016987324,
            5.6478352546691895
          ],
          "weighted_extra_to_common_norm_ratio": 856.09619140625,
          "extra_native_fit_cosine": -0.09412163496017456,
          "extra_U_cosine": 0.16839057207107544,
          "extra_common_cosine": 0.12785154581069946,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.26083123683929443,
            0.22955547273159027,
            0.012756850570440292,
            0.7782613635063171
          ],
          "weighted_extra_to_common_norm_ratio": 1.5961862802505493,
          "extra_native_fit_cosine": -0.979998767375946,
          "extra_U_cosine": 0.12369941920042038,
          "extra_common_cosine": -0.9788766503334045,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.007551371585577726,
            0.001370816258713603,
            0.0013058388140052557,
            0.004814420826733112
          ],
          "weighted_extra_to_common_norm_ratio": 0.5825266242027283,
          "extra_native_fit_cosine": -0.8721944093704224,
          "extra_U_cosine": -0.07158869504928589,
          "extra_common_cosine": -0.8658561110496521,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0024003698490560055,
            0.0008336943574249744,
            0.0006572307902388275,
            0.0024279849603772163
          ],
          "weighted_extra_to_common_norm_ratio": 0.8963592052459717,
          "extra_native_fit_cosine": -0.7046726942062378,
          "extra_U_cosine": 0.002116317395120859,
          "extra_common_cosine": -0.6915419101715088,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.0009839675622060895,
            0.0004934801836498082,
            0.00023164899903349578,
            0.0009114806889556348
          ],
          "weighted_extra_to_common_norm_ratio": 0.75201416015625,
          "extra_native_fit_cosine": -0.5188592076301575,
          "extra_U_cosine": 0.11946916580200195,
          "extra_common_cosine": -0.492765873670578,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 87,
    "arm": "D_EXTRA_FIT",
    "seconds": 127.42421388626099,
    "tokens": {
      "native": 5440,
      "fit": 5440,
      "U": 3840,
      "extra": 5440
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0013027231907472014,
            0.0014288900420069695,
            0.006240000016987324,
            0.009293685667216778
          ],
          "weighted_extra_to_common_norm_ratio": 1.4087324142456055,
          "extra_native_fit_cosine": -0.07169999927282333,
          "extra_U_cosine": 0.019442761316895485,
          "extra_common_cosine": -0.005546399392187595,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            1.0603576811263338e-05,
            9.473855243413709e-06,
            0.00030105208861641586,
            6.407540786312893e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.21270032227039337,
          "extra_native_fit_cosine": 0.690268337726593,
          "extra_U_cosine": -0.012964521534740925,
          "extra_common_cosine": 0.033026136457920074,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0005647334037348628,
            1.0464570550539065e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.01853010803461075,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": -0.0065230936743319035,
          "extra_common_cosine": -0.0065230936743319035,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            9.693922038422897e-05,
            1.0770584594865795e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.11110657453536987,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": -0.02127877250313759,
          "extra_common_cosine": -0.02127877250313759,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.00029049674049019814,
            2.6123569114133716e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.08992723375558853,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": -0.02968892827630043,
          "extra_common_cosine": -0.02968892827630043,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 125,
    "arm": "A_NO_H",
    "seconds": 98.20612001419067,
    "tokens": {
      "native": 2880,
      "fit": 2880,
      "U": 3840
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0008068764000199735,
            0.0008032173500396311,
            0.012226510792970657
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            8.305599294544663e-06,
            8.733406502869911e-06,
            0.0007925084210000932
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            1.6354204490198754e-05,
            0.0002895630896091461
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            4.506639470491791e-06,
            0.00016309191414620727
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.0006327051087282598,
            0.0002957965189125389,
            0.0005603870959021151
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 125,
    "arm": "B_H1",
    "seconds": 128.40070033073425,
    "tokens": {
      "native": 2880,
      "fit": 2880,
      "U": 3840,
      "extra": 960
    },
    "steps": 320,
    "clip_steps": 6,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0008068764000199735,
            0.0008032173500396311,
            0.012226510792970657,
            66.87093353271484
          ],
          "weighted_extra_to_common_norm_ratio": 5483.22998046875,
          "extra_native_fit_cosine": -0.016643883660435677,
          "extra_U_cosine": 0.1750822365283966,
          "extra_common_cosine": 0.1738515943288803,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.006075907964259386,
            0.012466620653867722,
            0.010176611132919788,
            0.02648361399769783
          ],
          "weighted_extra_to_common_norm_ratio": 1.2694412469863892,
          "extra_native_fit_cosine": -0.029159963130950928,
          "extra_U_cosine": 0.06001145765185356,
          "extra_common_cosine": 0.004018529783934355,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.00038121725083328784,
            0.000494837062433362,
            0.0010434063151478767,
            0.0009516205172985792
          ],
          "weighted_extra_to_common_norm_ratio": 0.7863190174102783,
          "extra_native_fit_cosine": 0.06659576296806335,
          "extra_U_cosine": 0.11279352009296417,
          "extra_common_cosine": 0.13781173527240753,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.00019908517424482852,
            0.0003464278706815094,
            0.0004498286871239543,
            0.0004868124960921705
          ],
          "weighted_extra_to_common_norm_ratio": 0.8202843070030212,
          "extra_native_fit_cosine": -0.04327203333377838,
          "extra_U_cosine": 0.05821537226438522,
          "extra_common_cosine": 0.012100056745111942,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00011299883044557646,
            0.00020857134950347245,
            0.00024993589613586664,
            0.00028282986022531986
          ],
          "weighted_extra_to_common_norm_ratio": 0.7976307272911072,
          "extra_native_fit_cosine": 0.02395058050751686,
          "extra_U_cosine": -0.11065436899662018,
          "extra_common_cosine": -0.059860989451408386,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 125,
    "arm": "C_H025",
    "seconds": 127.2705237865448,
    "tokens": {
      "native": 2880,
      "fit": 2880,
      "U": 3840,
      "extra": 960
    },
    "steps": 320,
    "clip_steps": 4,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0008068764000199735,
            0.0008032173500396311,
            0.012226510792970657,
            16.72088623046875
          ],
          "weighted_extra_to_common_norm_ratio": 1371.0660400390625,
          "extra_native_fit_cosine": -0.016629118472337723,
          "extra_U_cosine": 0.17506667971611023,
          "extra_common_cosine": 0.17383751273155212,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.00436772033572197,
            0.009445419535040855,
            0.009811527095735073,
            0.013448315672576427
          ],
          "weighted_extra_to_common_norm_ratio": 0.867299497127533,
          "extra_native_fit_cosine": -0.1114276722073555,
          "extra_U_cosine": -0.01410437747836113,
          "extra_common_cosine": -0.10487785190343857,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0002894554927479476,
            0.0005890673492103815,
            0.0009886508341878653,
            0.0007471076096408069
          ],
          "weighted_extra_to_common_norm_ratio": 0.6342325210571289,
          "extra_native_fit_cosine": -0.07235066592693329,
          "extra_U_cosine": 0.14543406665325165,
          "extra_common_cosine": 0.07588984072208405,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.00014343477960210294,
            0.00038317195139825344,
            0.00046261062379926443,
            0.0004730995569843799
          ],
          "weighted_extra_to_common_norm_ratio": 0.8054453730583191,
          "extra_native_fit_cosine": -0.1206485778093338,
          "extra_U_cosine": 0.041427575051784515,
          "extra_common_cosine": -0.057669807225465775,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            6.0849015426356345e-05,
            0.00018476316472515464,
            0.0003114375867880881,
            0.00027623301139101386
          ],
          "weighted_extra_to_common_norm_ratio": 0.7395142912864685,
          "extra_native_fit_cosine": -0.035435233265161514,
          "extra_U_cosine": -0.08284251391887665,
          "extra_common_cosine": -0.08950327336788177,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 125,
    "arm": "D_EXTRA_FIT",
    "seconds": 126.89511036872864,
    "tokens": {
      "native": 2880,
      "fit": 2880,
      "U": 3840,
      "extra": 2880
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0008068764000199735,
            0.0008032173500396311,
            0.012226510792970657,
            0.004391184542328119
          ],
          "weighted_extra_to_common_norm_ratio": 0.36006489396095276,
          "extra_native_fit_cosine": 0.14305943250656128,
          "extra_U_cosine": -0.001970120705664158,
          "extra_common_cosine": 0.012424529530107975,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            4.196528152533574e-06,
            1.3418963135336526e-05,
            0.0005880393437109888,
            7.844903302611783e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.13319984078407288,
          "extra_native_fit_cosine": -0.3040454387664795,
          "extra_U_cosine": -0.1538059115409851,
          "extra_common_cosine": -0.1610674262046814,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            3.7628655263688415e-05,
            0.0,
            0.00021037420083303005,
            0.00014722223568242043
          ],
          "weighted_extra_to_common_norm_ratio": 0.6928880214691162,
          "extra_native_fit_cosine": 0.7668631076812744,
          "extra_U_cosine": -0.019053693860769272,
          "extra_common_cosine": 0.11694306880235672,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0002371883310843259,
            0.0
          ],
          "weighted_extra_to_common_norm_ratio": 0.0,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0004415608709678054,
            1.9716830138349906e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.04465257376432419,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": 0.030646931380033493,
          "extra_common_cosine": 0.030646931380033493,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 126,
    "arm": "A_NO_H",
    "seconds": 97.44836330413818,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006078638834878802,
            0.0009736082865856588,
            0.007991422899067402
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.00010411129187559709,
            8.357213664567098e-05,
            0.00036662360071204603
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0001412370038451627,
            0.00015402033750433475,
            7.869968976592645e-05
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0002274646976729855,
            0.00022163540415931493,
            0.0007324431790038943
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00046180779463611543,
            0.0004032530705444515,
            0.0028248897287994623
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 126,
    "arm": "B_H1",
    "seconds": 127.36190485954285,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840,
      "extra": 1600
    },
    "steps": 320,
    "clip_steps": 9,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006078638834878802,
            0.0009736082865856588,
            0.007991422899067402,
            59.41984176635742
          ],
          "weighted_extra_to_common_norm_ratio": 7392.66259765625,
          "extra_native_fit_cosine": -0.1048726886510849,
          "extra_U_cosine": 0.0633322075009346,
          "extra_common_cosine": 0.045602668076753616,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.011410835199058056,
            0.010137946344912052,
            0.008649421855807304,
            0.010622359812259674
          ],
          "weighted_extra_to_common_norm_ratio": 0.46574005484580994,
          "extra_native_fit_cosine": 0.058353014290332794,
          "extra_U_cosine": -0.00492372689768672,
          "extra_common_cosine": 0.05283340439200401,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.00016517411859240383,
            0.00031824997859075665,
            0.0015203995862975717,
            0.0008603762253187597
          ],
          "weighted_extra_to_common_norm_ratio": 0.5341089963912964,
          "extra_native_fit_cosine": -0.0019005509093403816,
          "extra_U_cosine": -0.04301256686449051,
          "extra_common_cosine": -0.04106667637825012,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.00014757261669728905,
            0.00027701936778612435,
            0.0006226341938599944,
            0.0004785240744240582
          ],
          "weighted_extra_to_common_norm_ratio": 0.6632406115531921,
          "extra_native_fit_cosine": 0.010487351566553116,
          "extra_U_cosine": -0.003568256739526987,
          "extra_common_cosine": 0.001805645413696766,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            8.728550164960325e-05,
            0.0001015991365420632,
            0.00022416716092266142,
            0.0002201654715463519
          ],
          "weighted_extra_to_common_norm_ratio": 0.8071318864822388,
          "extra_native_fit_cosine": 4.7134701162576675e-05,
          "extra_U_cosine": -0.014629371464252472,
          "extra_common_cosine": -0.011993899941444397,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 126,
    "arm": "C_H025",
    "seconds": 127.43382382392883,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840,
      "extra": 1600
    },
    "steps": 320,
    "clip_steps": 8,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006078638834878802,
            0.0009736082865856588,
            0.007991422899067402,
            14.854129791259766
          ],
          "weighted_extra_to_common_norm_ratio": 1848.0623779296875,
          "extra_native_fit_cosine": -0.10488110780715942,
          "extra_U_cosine": 0.06333322823047638,
          "extra_common_cosine": 0.04560229182243347,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.011800861917436123,
            0.011129374615848064,
            0.008518029004335403,
            0.022373786196112633
          ],
          "weighted_extra_to_common_norm_ratio": 0.9240016937255859,
          "extra_native_fit_cosine": 0.2121790051460266,
          "extra_U_cosine": 0.008847488090395927,
          "extra_common_cosine": 0.2027469277381897,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0002531436912249774,
            0.00027700045029632747,
            0.0010333980899304152,
            0.0005085810553282499
          ],
          "weighted_extra_to_common_norm_ratio": 0.4221940338611603,
          "extra_native_fit_cosine": 0.1408073902130127,
          "extra_U_cosine": -0.025971366092562675,
          "extra_common_cosine": 0.03348865360021591,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0001695318496786058,
            0.0003786705492530018,
            0.0006206289399415255,
            0.00032939252560026944
          ],
          "weighted_extra_to_common_norm_ratio": 0.43310096859931946,
          "extra_native_fit_cosine": -0.005901048891246319,
          "extra_U_cosine": -0.0006857137195765972,
          "extra_common_cosine": -0.004284076392650604,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00010756924166344106,
            0.000126787053886801,
            0.00016193500778172165,
            0.00024636596208438277
          ],
          "weighted_extra_to_common_norm_ratio": 1.0002113580703735,
          "extra_native_fit_cosine": -0.0008995514363050461,
          "extra_U_cosine": -0.03036150522530079,
          "extra_common_cosine": -0.02073284611105919,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 126,
    "arm": "D_EXTRA_FIT",
    "seconds": 127.69813513755798,
    "tokens": {
      "native": 2560,
      "fit": 2560,
      "U": 3840,
      "extra": 2560
    },
    "steps": 320,
    "clip_steps": 0,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.0006078638834878802,
            0.0009736082865856588,
            0.007991422899067402,
            0.00565706379711628
          ],
          "weighted_extra_to_common_norm_ratio": 0.703818142414093,
          "extra_native_fit_cosine": 0.16520796716213226,
          "extra_U_cosine": -0.017294898629188538,
          "extra_common_cosine": 0.010160165838897228,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            1.5631287624273682e-06,
            1.5249461284838617e-06,
            0.0005221710307523608,
            1.4421391824726015e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.027620095759630203,
          "extra_native_fit_cosine": 0.5278033018112183,
          "extra_U_cosine": 0.026195047423243523,
          "extra_common_cosine": 0.029287129640579224,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            4.385243300930597e-06,
            4.3645131881930865e-06,
            0.0003573096764739603,
            2.4053957531577908e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.06726254522800446,
          "extra_native_fit_cosine": 0.8005754947662354,
          "extra_U_cosine": 0.022290252149105072,
          "extra_common_cosine": 0.041674233973026276,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.0004351318930275738,
            2.4861563360900618e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.0571356937289238,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": -0.17681071162223816,
          "extra_common_cosine": -0.17681071162223816,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            1.1684039236570243e-05,
            4.1901294025592506e-05,
            0.00023669151414651424,
            0.0001738187565933913
          ],
          "weighted_extra_to_common_norm_ratio": 0.7125018835067749,
          "extra_native_fit_cosine": 0.12922212481498718,
          "extra_U_cosine": 0.028607169166207314,
          "extra_common_cosine": 0.05499442666769028,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 146,
    "arm": "A_NO_H",
    "seconds": 98.23351407051086,
    "tokens": {
      "native": 2880,
      "fit": 2880,
      "U": 3840
    },
    "steps": 320,
    "clip_steps": 2,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.001015775604173541,
            0.0030382052063941956,
            0.030835622921586037
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.00010813029803102836,
            0.0001077069973689504,
            0.0009551116381771863
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            1.694763523119036e-05,
            0.0002981356519740075
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            7.485330115741817e-06,
            7.50657454773318e-06,
            6.416841642931104e-05
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00010369539086241275,
            0.00016692711506038904,
            0.0006284129922278225
          ],
          "weighted_extra_to_common_norm_ratio": null,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": null,
          "extra_common_cosine": null,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 146,
    "arm": "B_H1",
    "seconds": 128.71176052093506,
    "tokens": {
      "native": 2880,
      "fit": 2880,
      "U": 3840,
      "extra": 1920
    },
    "steps": 320,
    "clip_steps": 7,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.001015775604173541,
            0.0030382052063941956,
            0.030835622921586037,
            20.27495574951172
          ],
          "weighted_extra_to_common_norm_ratio": 646.277587890625,
          "extra_native_fit_cosine": -0.09092435985803604,
          "extra_U_cosine": -0.024873685091733932,
          "extra_common_cosine": -0.03384709358215332,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.03323522210121155,
            0.024697108194231987,
            0.01760054938495159,
            0.003924960736185312
          ],
          "weighted_extra_to_common_norm_ratio": 0.06112951040267944,
          "extra_native_fit_cosine": -0.4978231191635132,
          "extra_U_cosine": -0.1428242325782776,
          "extra_common_cosine": -0.4865992069244385,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0002377333294134587,
            0.00025233038468286395,
            0.0009921459713950753,
            0.0008853481849655509
          ],
          "weighted_extra_to_common_norm_ratio": 0.8365294337272644,
          "extra_native_fit_cosine": 0.045820895582437515,
          "extra_U_cosine": -0.13276776671409607,
          "extra_common_cosine": -0.10521644353866577,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.00020478978694882244,
            0.00019119765784125775,
            0.0004665989545173943,
            0.0003369792830199003
          ],
          "weighted_extra_to_common_norm_ratio": 0.5864810943603516,
          "extra_native_fit_cosine": -0.15812739729881287,
          "extra_U_cosine": 0.043291762471199036,
          "extra_common_cosine": -0.06545309722423553,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.00014186899352353066,
            0.00016559391224291176,
            0.0002907021844293922,
            0.00024190789554268122
          ],
          "weighted_extra_to_common_norm_ratio": 0.6299676895141602,
          "extra_native_fit_cosine": -0.10216327756643295,
          "extra_U_cosine": 0.0031163692474365234,
          "extra_common_cosine": -0.0716012716293335,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 146,
    "arm": "C_H025",
    "seconds": 128.52915930747986,
    "tokens": {
      "native": 2880,
      "fit": 2880,
      "U": 3840,
      "extra": 1920
    },
    "steps": 320,
    "clip_steps": 6,
    "H_nonzero_steps": 320,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.001015775604173541,
            0.0030382052063941956,
            0.030835622921586037,
            5.069250583648682
          ],
          "weighted_extra_to_common_norm_ratio": 161.58570861816406,
          "extra_native_fit_cosine": -0.09090197086334229,
          "extra_U_cosine": -0.024829164147377014,
          "extra_common_cosine": -0.03380101919174194,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.042312417179346085,
            0.032423727214336395,
            0.013990557752549648,
            0.002300021005794406
          ],
          "weighted_extra_to_common_norm_ratio": 0.029117591679096222,
          "extra_native_fit_cosine": -0.4398648142814636,
          "extra_U_cosine": -0.09022758901119232,
          "extra_common_cosine": -0.4307119846343994,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0003474969125818461,
            0.00033816619543358684,
            0.0011536648962646723,
            0.0009001700091175735
          ],
          "weighted_extra_to_common_norm_ratio": 0.7033174633979797,
          "extra_native_fit_cosine": -0.08263920247554779,
          "extra_U_cosine": -0.14809903502464294,
          "extra_common_cosine": -0.17411328852176666,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            0.00027656491147354245,
            0.0002455201174598187,
            0.00045704207150265574,
            0.0004499605856835842
          ],
          "weighted_extra_to_common_norm_ratio": 0.6977958083152771,
          "extra_native_fit_cosine": -0.22149375081062317,
          "extra_U_cosine": -0.04971982538700104,
          "extra_common_cosine": -0.20416361093521118,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.0001741857558954507,
            0.0001709590869722888,
            0.00023345705994870514,
            0.00024346407735720277
          ],
          "weighted_extra_to_common_norm_ratio": 0.5944720506668091,
          "extra_native_fit_cosine": -0.4004583954811096,
          "extra_U_cosine": -0.00867123156785965,
          "extra_common_cosine": -0.32716578245162964,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  },
  {
    "edit_index": 146,
    "arm": "D_EXTRA_FIT",
    "seconds": 129.03535890579224,
    "tokens": {
      "native": 2880,
      "fit": 2880,
      "U": 3840,
      "extra": 2880
    },
    "steps": 320,
    "clip_steps": 2,
    "H_nonzero_steps": null,
    "diagnostic_steps": [
      {
        "step": 1,
        "gradient": {
          "weighted_term_norms": [
            0.001015775604173541,
            0.0030382052063941956,
            0.030835622921586037,
            0.004296386614441872
          ],
          "weighted_extra_to_common_norm_ratio": 0.13695016503334045,
          "extra_native_fit_cosine": 0.49469488859176636,
          "extra_U_cosine": 0.13426321744918823,
          "extra_common_cosine": 0.18310332298278809,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 20,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            6.647679583693389e-06,
            0.000944540137425065,
            0.0001264431921299547
          ],
          "weighted_extra_to_common_norm_ratio": 0.13392126560211182,
          "extra_native_fit_cosine": -0.37156420946121216,
          "extra_U_cosine": 0.09207332879304886,
          "extra_common_cosine": 0.08949419856071472,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 80,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.00013610078894998878,
            2.456889887980651e-05
          ],
          "weighted_extra_to_common_norm_ratio": 0.18051987886428833,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": 0.043470121920108795,
          "extra_common_cosine": 0.043470121920108795,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 160,
        "gradient": {
          "weighted_term_norms": [
            1.076922762877075e-05,
            1.071603583113756e-05,
            0.00013191445032134652,
            0.00021469886996783316
          ],
          "weighted_extra_to_common_norm_ratio": 1.6283864974975586,
          "extra_native_fit_cosine": 0.17475834488868713,
          "extra_U_cosine": -0.0653444305062294,
          "extra_common_cosine": -0.03690086305141449,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      },
      {
        "step": 320,
        "gradient": {
          "weighted_term_norms": [
            0.0,
            0.0,
            0.000346931628882885,
            0.00020826613763347268
          ],
          "weighted_extra_to_common_norm_ratio": 0.600308895111084,
          "extra_native_fit_cosine": null,
          "extra_U_cosine": 0.059490617364645004,
          "extra_common_cosine": 0.059490617364645004,
          "extraction": "actual successive backward gradient increments; no additional forward/RNG draw"
        }
      }
    ]
  }
]

H缺失臂记NA；梯度冲突仅作机制线索；D不严格等算力。
