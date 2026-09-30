# Table 3. Confident error and abstention on the probe set

| family | tier | level | n_items | hallucination_rate | hallucination_ci | abstention_rate | delta_pp | delta_ci_pp | p_holm | criterion2_pass |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:3b-instruct | school | fp16 | 200 | 0.625 | [0.56, 0.69] | 0.365 | 0 | [0, 0] | — | — |
| qwen2.5:3b-instruct | school | q8_0 | 200 | 0.59 | [0.52, 0.66] | 0.4 | -3.5 | [-6, -1] | 0.0625 | no |
| qwen2.5:3b-instruct | school | q4_K_M | 200 | 0.535 | [0.465, 0.605] | 0.46 | -9 | [-14.5, -4] | 0.007874 | yes |
| qwen2.5:3b-instruct | college | fp16 | 200 | 0.805 | [0.745, 0.86] | 0.18 | 0 | [0, 0] | — | — |
| qwen2.5:3b-instruct | college | q8_0 | 200 | 0.815 | [0.76, 0.87] | 0.175 | 1 | [0, 2.5] | 1 | no |
| qwen2.5:3b-instruct | college | q4_K_M | 200 | 0.74 | [0.68, 0.8] | 0.25 | -6.5 | [-12.5, -1] | 0.2458 | yes |
| qwen2.5:3b-instruct | expert | fp16 | 200 | 0.79 | [0.73, 0.845] | 0.205 | 0 | [0, 0] | — | — |
| qwen2.5:3b-instruct | expert | q8_0 | 200 | 0.785 | [0.725, 0.84] | 0.21 | -0.5 | [-3.5, 2.5] | 1 | no |
| qwen2.5:3b-instruct | expert | q4_K_M | 200 | 0.66 | [0.595, 0.725] | 0.34 | -13 | [-19.5, -7] | 0.000636 | yes |
| llama3.2:3b-instruct | school | fp16 | 200 | 0.86 | [0.81, 0.905] | 0.135 | 0 | [0, 0] | — | — |
| llama3.2:3b-instruct | school | q8_0 | 200 | 0.87 | [0.82, 0.915] | 0.125 | 1 | [0, 2.5] | 1 | no |
| llama3.2:3b-instruct | school | q4_K_M | 200 | 0.93 | [0.895, 0.965] | 0.07 | 7 | [3, 11] | 0.007874 | no |
| llama3.2:3b-instruct | college | fp16 | 200 | 0.95 | [0.92, 0.98] | 0.05 | 0 | [0, 0] | — | — |
| llama3.2:3b-instruct | college | q8_0 | 200 | 0.955 | [0.925, 0.98] | 0.045 | 0.5 | [0, 1.5] | 1 | no |
| llama3.2:3b-instruct | college | q4_K_M | 200 | 0.975 | [0.95, 0.995] | 0.025 | 2.5 | [0, 5.5] | 0.625 | no |
| llama3.2:3b-instruct | expert | fp16 | 200 | 0.935 | [0.9, 0.97] | 0.065 | 0 | [0, 0] | — | — |
| llama3.2:3b-instruct | expert | q8_0 | 200 | 0.94 | [0.905, 0.97] | 0.06 | 0.5 | [0, 1.5] | 1 | no |
| llama3.2:3b-instruct | expert | q4_K_M | 200 | 0.965 | [0.935, 0.99] | 0.03 | 3 | [1, 5.5] | 0.1562 | no |
| gemma2:2b-instruct | school | fp16 | 200 | 0.985 | [0.965, 1] | 0.015 | 0 | [0, 0] | — | — |
| gemma2:2b-instruct | school | q8_0 | 200 | 0.985 | [0.965, 1] | 0.015 | 0 | [0, 0] | 1 | no |
| gemma2:2b-instruct | school | q4_K_M | 200 | 0.955 | [0.925, 0.98] | 0.045 | -3 | [-6, -0.5] | 0.2109 | yes |
| gemma2:2b-instruct | college | fp16 | 200 | 0.99 | [0.975, 1] | 0.01 | 0 | [0, 0] | — | — |
| gemma2:2b-instruct | college | q8_0 | 200 | 0.99 | [0.975, 1] | 0.01 | 0 | [0, 0] | 1 | no |
| gemma2:2b-instruct | college | q4_K_M | 200 | 0.985 | [0.965, 1] | 0.015 | -0.5 | [-1.5, 0] | 1 | no |
| gemma2:2b-instruct | expert | fp16 | 200 | 0.94 | [0.905, 0.97] | 0.06 | 0 | [0, 0] | — | — |
| gemma2:2b-instruct | expert | q8_0 | 200 | 0.945 | [0.91, 0.975] | 0.055 | 0.5 | [0, 1.5] | 1 | no |
| gemma2:2b-instruct | expert | q4_K_M | 200 | 0.935 | [0.895, 0.965] | 0.065 | -0.5 | [-2.5, 1] | 1 | no |
