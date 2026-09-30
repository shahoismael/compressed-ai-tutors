# Energy alignment report
**ET&S study — measured inference energy, all 81 cells**
**Aligned 2026-09-28 from HWiNFO CPU Package Power, idle-subtracted**

---

## 1. Provenance

| Item | Value |
|---|---|
| Instrument | HWiNFO64 8.50-6020, sensor log, 1 s interval |
| Sensor | `CPU Package Power [W]` — Intel RAPL package domain |
| Machine | Lenovo ThinkPad E15 Gen 4, Intel Core i7-1255U, Windows 11 |
| Logs | `ets-energy-night1.csv` (14,358 samples, 8.01 h) + `ets-energy-night2.csv` (22,508 samples, 12.55 h) |
| Merged series | 36,866 samples |
| Idle baseline | **3.218 W**, mean of 4 idle windows (2 per night, 120 s each) |
| Idle spread | 1.221 W across the four windows |
| Cells aligned | **81 / 81** |
| Cells uncovered | **0** |

`IGPU Power` was present in the log and deliberately excluded: on this
processor the integrated-graphics rail sits inside the package domain, so
counting it again would inflate every figure. No discrete GPU is present, so
the package figure is the whole inference power.

Net energy per cell = integrated package joules over the cell's recorded
wall-clock window, minus (3.218 W x integrated seconds). Gross and net are both
stored in each cell's JSON. Model load and cache fill happened during the
discarded warm-up, outside the measured window.

---

## 2. Net joules per assessed item

### 2.1 Forced choice (mcq) — 8 generated tokens per item

| Model | Tier | FP16 | q8_0 | q4_K_M | q8/FP16 | q4/FP16 |
|---|---|---|---|---|---|---|
| gemma2:2b | school | 48.578 | 44.300 | 27.182 | 91.2% | 56.0% |
| gemma2:2b | college | 49.720 | 46.414 | 28.888 | 93.4% | 58.1% |
| gemma2:2b | expert | 58.389 | 53.348 | 32.917 | 91.4% | 56.4% |
| llama3.2:3b | school | 42.319 | 39.996 | 23.642 | 94.5% | 55.9% |
| llama3.2:3b | college | 41.767 | 42.417 | 25.174 | 101.6% | 60.3% |
| llama3.2:3b | expert | 51.780 | 51.880 | 31.096 | 100.2% | 60.1% |
| qwen2.5:3b | school | 39.235 | 38.539 | 25.030 | 98.2% | 63.8% |
| qwen2.5:3b | college | 42.609 | 42.243 | 25.488 | 99.1% | 59.8% |
| qwen2.5:3b | expert | 51.778 | 51.933 | 31.076 | 100.3% | 60.0% |

### 2.2 Abstention probe — 8 generated tokens per item

| Model | Tier | FP16 | q8_0 | q4_K_M | q8/FP16 | q4/FP16 |
|---|---|---|---|---|---|---|
| gemma2:2b | school | 52.343 | 49.853 | 30.356 | 95.2% | 58.0% |
| gemma2:2b | college | 55.469 | 51.645 | 32.473 | 93.1% | 58.5% |
| gemma2:2b | expert | 62.962 | 59.853 | 36.767 | 95.1% | 58.4% |
| llama3.2:3b | school | 36.525 | 35.870 | 21.913 | 98.2% | 60.0% |
| llama3.2:3b | college | 40.383 | 38.496 | 23.818 | 95.3% | 59.0% |
| llama3.2:3b | expert | 54.647 | 50.368 | 33.164 | 92.2% | 60.7% |
| qwen2.5:3b | school | 36.640 | 36.217 | 21.744 | 98.8% | 59.3% |
| qwen2.5:3b | college | 41.000 | 39.332 | 24.627 | 95.9% | 60.1% |
| qwen2.5:3b | expert | 50.961 | 49.373 | 30.386 | 96.9% | 59.6% |

### 2.3 Open-ended — 5 samples x up to 96 tokens per item

| Model | Tier | FP16 | q8_0 | q4_K_M | q8/FP16 | q4/FP16 |
|---|---|---|---|---|---|---|
| gemma2:2b | school | 258.889 | 169.082 | 121.649 | 65.3% | 47.0% |
| gemma2:2b | college | 266.708 | 176.328 | 117.191 | 66.1% | 43.9% |
| gemma2:2b | expert | 279.822 | 172.049 | 120.411 | 61.5% | 43.0% |
| llama3.2:3b | school | 413.490 | 226.451 | 150.960 | 54.8% | 36.5% |
| llama3.2:3b | college | 417.207 | 251.750 | 153.289 | 60.3% | 36.7% |
| llama3.2:3b | expert | 557.356 | 303.662 | 194.700 | 54.5% | 34.9% |
| qwen2.5:3b | school | 275.165 | 167.719 | 112.597 | 60.9% | 40.9% |
| qwen2.5:3b | college | 288.204 | 161.003 | 107.356 | 55.9% | 37.3% |
| qwen2.5:3b | expert | 362.531 | 200.225 | 140.379 | 55.2% | 38.7% |

---

## 3. What the numbers show

**3.1 q8_0 buys almost nothing on short-answer assessment.** Across the
eighteen forced-choice and probe conditions, q8_0 sits between 91% and 102% of
the FP16 figure, and in four conditions it is marginally *higher*. Against the
pre-registered 60% energy criterion, q8_0 fails on every tier of every model
for these two modes.

**3.2 q4_K_M sits on the threshold, not comfortably under it.** On forced
choice it ranges 55.9%–63.8% of baseline; the pre-registered rule is ≤60%.
Four of nine conditions exceed it (llama college 60.3%, llama expert 60.1%,
qwen school 63.8%, qwen expert 60.0% at the boundary). On the probe the range
is 58.0%–60.7%, again straddling the line. Whether a configuration passes
criterion 3 therefore has to be decided per tier, not per model — which is what
the rule was written to allow.

**3.3 The saving grows with output length.** In the open-ended mode, where each
item generates five samples of up to 96 tokens instead of a single letter, q8_0
falls to 54.5%–66.1% and q4_K_M to 34.9%–47.0%. The same compression that saved
nothing on a one-letter answer saves roughly half to two-thirds on generated
prose.

This is a reportable association, not a demonstrated mechanism, and the paper
states it that way. The design does not isolate prefill from decode, so the
explanation is left to the Discussion as a hypothesis consistent with the data:
a forced-choice item is dominated by prompt processing, which is compute-bound,
while generation is dominated by weight streaming, which is memory-bound and is
where reduced precision pays.

**3.4 Practical reading for a department.** If the tutor's job is
multiple-choice assessment, quantisation below FP16 is not an energy argument —
it is a memory-footprint argument. The energy case for compression appears only
when the tutor writes answers.

---

## 4. Integrity notes

- Every figure here comes from an executed run with a measured power trace. No
  TDP estimate, no modelled value, no extrapolation.
- Coverage was 100%: no cell window fell outside the logged interval, so no
  figure was reconstructed.
- The idle baseline varied by 1.221 W across the four windows. That spread
  propagates into the net figures and is reported in Limitations rather than
  smoothed away.
- Energy differences below 5% between configurations are treated as not
  established, per the analysis plan. The q8_0-vs-FP16 comparisons on mcq and
  probe fall inside that band in several conditions and are reported as
  **no established difference**, not as a saving.
- Idle subtraction removed a substantial share of the gross reading; gross and
  net are both retained in each cell's JSON so the subtraction can be audited.

---

## 5. Files

| Path | Contents |
|---|---|
| `06-results/energy/<cell>.json` | per-cell record, `energy` block holds gross, idle, net, J/item, J/token |
| `06-results/energy/raw/<cell>.jsonl` | per-item responses from the energy pass |
| `06-results/energy/session__*.json` | idle windows and session timing, one per night |
| `05-codes/run_energy.py` | the energy pass |
| `05-codes/hwinfo_align.py` | the alignment step that produced this report |
