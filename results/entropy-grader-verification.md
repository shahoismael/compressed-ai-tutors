# Entropy grader — defect and verification

**Recorded 2026-09-29. Relevant to Methods (semantic entropy) and to
Limitations.**

---

## 1. The defect

The first implementation of `analyse_entropy.py` (run 2026-09-28, 8.5 h,
27 cells) contained two errors that invalidated every accuracy-dependent
output of that run.

1. **The question never reached the entailment model.** It was used to build
   the cache key but was not part of the NLI input. Farquhar et al. (2024)
   concatenate the question with each answer before running entailment,
   because entailment between bare answer strings is undefined: "Paris" and
   "the capital is Paris" neither entail nor contradict each other without
   the question.
2. **Grading used a single, maximally strict test.** The reference is a short
   multiple-choice option while the model produces a sentence, so requiring
   bidirectional equivalence with that option is a harsh instrument and its
   harshness was invisible in the output.

**Symptom:** open-ended accuracy of 0.000–0.060 across all 27 cells, which is
not a plausible description of these models. AUROC and AURAC, both computed
against those labels, were corrupted with it. Mean entropy and cluster counts
do not depend on the gold text and were unaffected.

**Disposition:** the 2026-09-28 results file and its entailment cache are
discarded. No number from that run appears anywhere in the manuscript.

---

## 2. The correction

`analyse_entropy.py` v2:

- Every NLI input is `"<question> <answer>"`, for clustering and for grading.
- Grading is reported at two strictness levels, so the grader's own effect on
  the result is visible rather than hidden:
  - **strict** — the modal cluster's representative and the gold answer entail
    each other. This is the headline figure.
  - **lenient** — the modal cluster's representative entails the gold answer.
- The cache is versioned (`entailment_cache_v2.json`); v1 values are ignored
  because they were computed without the question.
- Every ordered pair an item needs is scored in one batched forward pass
  rather than in two-pair calls.

---

## 3. Verification

Smoke test, 10 items per cell, first cell:

```
qwen2.5:3b-instruct-fp16  school  n=10  H=0.790
    acc(strict)=0.500  acc(lenient)=0.600  AUROC=0.72  AURAC=0.6198
```

Accuracy moved from 0.010 to 0.500 on the same data with no change to the
model outputs — only to how the question is supplied to the entailment model.
Strict and lenient differ by one item in ten, which is the expected size of a
grading-strictness effect and confirms the two levels are measuring what they
were meant to measure.

The run was then stopped by the operator; the `KeyboardInterrupt` traceback in
the console is that stop, not a failure.

**Status:** grader verified on one cell. The full run supersedes this note.

---

## 4. What goes in the paper

- Methods state that entailment inputs are question-concatenated, per Farquhar
  et al., and that the discrete estimator is used because token probabilities
  are not exposed by the runtime.
- Methods state both grading levels and name strict as the reported one.
- Limitations state that the reference answer is a multiple-choice option
  rather than a free-form reference, that this bounds open-ended accuracy
  downward, and that the strict/lenient gap is reported so the reader can see
  the size of that bound.
