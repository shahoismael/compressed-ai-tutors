# Table 4. Net energy and the pre-registered deployability rule

| family | tier | level | j_per_item_mcq | j_per_item_open | ratio_mcq | ratio_open | energy_difference_established | criterion1_accuracy | criterion2_hallucination | criterion3_energy | criterion4_aurac | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5:3b-instruct | school | fp16 | 39.24 | 275.2 | 1 | 1 | no | yes | — | no | yes | baseline |
| qwen2.5:3b-instruct | school | q8_0 | 38.54 | 167.7 | 0.9823 | 0.6095 | no | yes | no | no | yes | FAIL |
| qwen2.5:3b-instruct | school | q4_K_M | 25.03 | 112.6 | 0.6379 | 0.4092 | yes | yes | yes | no | yes | FAIL |
| qwen2.5:3b-instruct | college | fp16 | 42.61 | 288.2 | 1 | 1 | no | yes | — | no | yes | baseline |
| qwen2.5:3b-instruct | college | q8_0 | 42.24 | 161 | 0.9914 | 0.5586 | no | yes | no | no | no | FAIL |
| qwen2.5:3b-instruct | college | q4_K_M | 25.49 | 107.4 | 0.5982 | 0.3725 | yes | yes | yes | yes | no | FAIL |
| qwen2.5:3b-instruct | expert | fp16 | 51.78 | 362.5 | 1 | 1 | no | yes | — | no | yes | baseline |
| qwen2.5:3b-instruct | expert | q8_0 | 51.93 | 200.2 | 1.003 | 0.5523 | no | yes | no | no | no | FAIL |
| qwen2.5:3b-instruct | expert | q4_K_M | 31.08 | 140.4 | 0.6002 | 0.3872 | yes | yes | yes | no | yes | FAIL |
| llama3.2:3b-instruct | school | fp16 | 42.32 | 413.5 | 1 | 1 | no | yes | — | no | yes | baseline |
| llama3.2:3b-instruct | school | q8_0 | 40 | 226.5 | 0.9451 | 0.5477 | yes | yes | no | no | yes | FAIL |
| llama3.2:3b-instruct | school | q4_K_M | 23.64 | 151 | 0.5587 | 0.3651 | yes | yes | no | yes | no | FAIL |
| llama3.2:3b-instruct | college | fp16 | 41.77 | 417.2 | 1 | 1 | no | yes | — | no | yes | baseline |
| llama3.2:3b-instruct | college | q8_0 | 42.42 | 251.7 | 1.016 | 0.6034 | no | yes | no | no | no | FAIL |
| llama3.2:3b-instruct | college | q4_K_M | 25.17 | 153.3 | 0.6027 | 0.3674 | yes | yes | no | no | yes | FAIL |
| llama3.2:3b-instruct | expert | fp16 | 51.78 | 557.4 | 1 | 1 | no | yes | — | no | yes | baseline |
| llama3.2:3b-instruct | expert | q8_0 | 51.88 | 303.7 | 1.002 | 0.5448 | no | yes | no | no | no | FAIL |
| llama3.2:3b-instruct | expert | q4_K_M | 31.1 | 194.7 | 0.6005 | 0.3493 | yes | yes | no | no | no | FAIL |
| gemma2:2b-instruct | school | fp16 | 48.58 | 258.9 | 1 | 1 | no | yes | — | no | yes | baseline |
| gemma2:2b-instruct | school | q8_0 | 44.3 | 169.1 | 0.9119 | 0.6531 | yes | yes | no | no | yes | FAIL |
| gemma2:2b-instruct | school | q4_K_M | 27.18 | 121.6 | 0.5596 | 0.4699 | yes | yes | yes | yes | no | FAIL |
| gemma2:2b-instruct | college | fp16 | 49.72 | 266.7 | 1 | 1 | no | yes | — | no | yes | baseline |
| gemma2:2b-instruct | college | q8_0 | 46.41 | 176.3 | 0.9335 | 0.6611 | yes | yes | no | no | no | FAIL |
| gemma2:2b-instruct | college | q4_K_M | 28.89 | 117.2 | 0.581 | 0.4394 | yes | yes | no | yes | yes | FAIL |
| gemma2:2b-instruct | expert | fp16 | 58.39 | 279.8 | 1 | 1 | no | yes | — | no | yes | baseline |
| gemma2:2b-instruct | expert | q8_0 | 53.35 | 172 | 0.9137 | 0.6149 | yes | yes | no | no | yes | FAIL |
| gemma2:2b-instruct | expert | q4_K_M | 32.92 | 120.4 | 0.5637 | 0.4303 | yes | yes | no | yes | yes | FAIL |
