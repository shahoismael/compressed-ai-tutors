# Table 1. Accuracy by model family, difficulty tier and compression level

| family | tier | level | n_items | accuracy | accuracy_ci | delta_pp | delta_ci_pp | mcnemar_b | mcnemar_c | n_discordant | p_holm | significant_05 | parse_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:3b-instruct | school | fp16 | 1000 | 0.8 | [0.775, 0.824] | 0 | [0, 0] | 0 | 0 | 0 | — | no | 1 |
| qwen2.5:3b-instruct | school | q8_0 | 1000 | 0.806 | [0.781, 0.83] | 0.6 | [0.1, 1.2] | 1 | 7 | 8 | 0.4219 | no | 1 |
| qwen2.5:3b-instruct | school | q4_K_M | 1000 | 0.8 | [0.775, 0.825] | 0 | [-1.4, 1.5] | 28 | 28 | 56 | 1 | no | 1 |
| qwen2.5:3b-instruct | college | fp16 | 922 | 0.5401 | [0.5076, 0.5727] | 0 | [0, 0] | 0 | 0 | 0 | — | no | 1 |
| qwen2.5:3b-instruct | college | q8_0 | 922 | 0.5391 | [0.5076, 0.5716] | -0.108 | [-0.976, 0.759] | 9 | 8 | 17 | 1 | no | 1 |
| qwen2.5:3b-instruct | college | q4_K_M | 922 | 0.5119 | [0.4794, 0.5445] | -2.82 | [-5.206, -0.542] | 75 | 49 | 124 | 0.1462 | no | 1 |
| qwen2.5:3b-instruct | expert | fp16 | 1000 | 0.534 | [0.503, 0.565] | 0 | [0, 0] | 0 | 0 | 0 | — | no | 1 |
| qwen2.5:3b-instruct | expert | q8_0 | 1000 | 0.532 | [0.501, 0.563] | -0.2 | [-0.9, 0.5] | 8 | 6 | 14 | 1 | no | 1 |
| qwen2.5:3b-instruct | expert | q4_K_M | 1000 | 0.512 | [0.481, 0.544] | -2.2 | [-4.4, 0] | 71 | 49 | 120 | 0.3287 | no | 1 |
| llama3.2:3b-instruct | school | fp16 | 1000 | 0.735 | [0.707, 0.761] | 0 | [0, 0] | 0 | 0 | 0 | — | no | 1 |
| llama3.2:3b-instruct | school | q8_0 | 1000 | 0.737 | [0.709, 0.763] | 0.2 | [-0.3, 0.8] | 3 | 5 | 8 | 1 | no | 1 |
| llama3.2:3b-instruct | school | q4_K_M | 1000 | 0.726 | [0.698, 0.753] | -0.9 | [-2.4, 0.6] | 33 | 24 | 57 | 1 | no | 1 |
| llama3.2:3b-instruct | college | fp16 | 922 | 0.4761 | [0.4447, 0.5087] | 0 | [0, 0] | 0 | 0 | 0 | — | no | 1 |
| llama3.2:3b-instruct | college | q8_0 | 922 | 0.4805 | [0.449, 0.513] | 0.434 | [-0.325, 1.193] | 4 | 8 | 12 | 1 | no | 1 |
| llama3.2:3b-instruct | college | q4_K_M | 922 | 0.4664 | [0.4338, 0.4989] | -0.976 | [-2.82, 0.868] | 43 | 34 | 77 | 1 | no | 1 |
| llama3.2:3b-instruct | expert | fp16 | 1000 | 0.485 | [0.454, 0.516] | 0 | [0, 0] | 0 | 0 | 0 | — | no | 1 |
| llama3.2:3b-instruct | expert | q8_0 | 1000 | 0.488 | [0.457, 0.519] | 0.3 | [-0.1, 0.8] | 1 | 4 | 5 | 1 | no | 1 |
| llama3.2:3b-instruct | expert | q4_K_M | 1000 | 0.479 | [0.449, 0.51] | -0.6 | [-2.4, 1.2] | 44 | 38 | 82 | 1 | no | 1 |
| gemma2:2b-instruct | school | fp16 | 1000 | 0.759 | [0.732, 0.785] | 0 | [0, 0] | 0 | 0 | 0 | — | no | 1 |
| gemma2:2b-instruct | school | q8_0 | 1000 | 0.758 | [0.731, 0.784] | -0.1 | [-0.6, 0.4] | 4 | 3 | 7 | 1 | no | 1 |
| gemma2:2b-instruct | school | q4_K_M | 1000 | 0.749 | [0.722, 0.776] | -1 | [-2.4, 0.3] | 29 | 19 | 48 | 0.9671 | no | 1 |
| gemma2:2b-instruct | college | fp16 | 922 | 0.461 | [0.4295, 0.4924] | 0 | [0, 0] | 0 | 0 | 0 | — | no | 1 |
| gemma2:2b-instruct | college | q8_0 | 922 | 0.4653 | [0.4328, 0.4968] | 0.434 | [-0.108, 1.085] | 2 | 6 | 8 | 1 | no | 1 |
| gemma2:2b-instruct | college | q4_K_M | 922 | 0.4436 | [0.4111, 0.475] | -1.735 | [-3.362, -0.108] | 38 | 22 | 60 | 0.2595 | no | 1 |
| gemma2:2b-instruct | expert | fp16 | 1000 | 0.485 | [0.454, 0.516] | 0 | [0, 0] | 0 | 0 | 0 | — | no | 1 |
| gemma2:2b-instruct | expert | q8_0 | 1000 | 0.487 | [0.455, 0.518] | 0.2 | [-0.2, 0.6] | 1 | 3 | 4 | 1 | no | 1 |
| gemma2:2b-instruct | expert | q4_K_M | 1000 | 0.481 | [0.45, 0.512] | -0.4 | [-2.1, 1.2] | 36 | 32 | 68 | 1 | no | 1 |
