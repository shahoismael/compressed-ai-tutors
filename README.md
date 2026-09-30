# compressed-ai-tutors

Do compressed classroom LLMs keep their answers, not just their scores?

Three open-weight model families were quantised to 8-bit and 4-bit and run on
2,922 fixed items across a school-to-expert difficulty gradient, on one CPU.
Alongside accuracy the campaign recorded answer fidelity to the uncompressed
model, confident error under an abstention probe, semantic entropy on
open-ended generations, and inference energy measured on the machine.
53,298 inferences across 135 cells.

## Headline

| | |
|---|---|
| Accuracy change under compression | −2.8 to +0.6 pp, none significant after Holm correction |
| Individual answers changed at 4-bit | 5.7 % to 20.3 % |
| Confident error at 4-bit | −13 pp to +7 pp, direction depends on the model family |
| Energy at 8-bit | 0.91 to 1.02 of baseline |
| Energy at 4-bit | 0.56 to 0.64 (forced choice), 0.35 to 0.47 (generation) |
| Semantic entropy AUROC | 0.69 to 0.99 |
| Configurations passing the pre-registered deployability rule | 0 of 18 |

Accuracy held everywhere. Up to one answer in five changed underneath it.

## Layout

```
code/        the campaign, the analysis and the figure scripts
data/        item_manifest.csv (identifiers only) and raw/ model outputs
results/     core_results.json, entropy_results.json, per-cell energy
tables/      Tables 1-6 as produced by analyse_core.py, .md and .csv
figures/     Figures 1-4, vector PDF and 600 dpi PNG, with captions
```

## Reproducing

```bash
pip install -r requirements.txt

python code/build_items.py      # rebuilds the item set from the source benchmarks
python code/run_all.py          # the accuracy campaign      (~40 h on one CPU)
python code/run_energy.py       # the energy pass            (~11 h, machine idle)
python code/hwinfo_align.py     # aligns power logs to cells
python code/analyse_core.py     # Tables 1-5 and core_results.json
python code/analyse_entropy.py  # Table 6 and entropy_results.json (~13 h)
python code/fig1.py ; python code/fig2.py ; python code/fig3.py ; python code/fig4.py
```

Inference is served locally by [Ollama](https://ollama.com). The long runs are
resumable and take a `--max-hours` budget, so they can be split across nights.
Every script writes per-cell output and skips what is already complete.

`build_items.py` needs the three source benchmarks; see `LICENSE-DATA` for
where to get them. Every other step runs from what is in this repository.

## What is and is not here

`data/raw/` holds one JSONL per cell: the model's answer, whether it was
correct, whether it abstained, token counts and wall time. It carries no
benchmark question or option text, and the reference answer strings were
removed before publication. `data/item_manifest.csv` carries identifiers only.
Rejoin with the originals using `build_items.py`.

The entailment caches (38 MB) are not included. `analyse_entropy.py` rebuilds
them on first run.

Raw HWiNFO power logs are in the Zenodo archive rather than here.

## A correction worth reading

The first semantic-entropy pass wrote the question into the cache key but not
into the entailment model's input, which made open-ended grading meaningless.
That run was discarded in full and nothing from it appears in the paper.
`results/entropy-grader-verification.md` documents the defect, the fix and the
verification.

## Measurement notes

Energy is CPU package power logged at 1 Hz with HWiNFO64, integrated over each
cell's recorded window, minus idle draw (3.218 W, mean of four 120 s windows,
spread 1.221 W). Coverage is complete: all 81 measured cells aligned to a
logged window, and no value is estimated. Carbon conversion follows Green
Algorithms with PUE = 1, this being a personal laptop.

Decoding is greedy at temperature 0 with a fixed seed for correctness,
fidelity and probe measures; the open-ended measure samples five generations
per item at temperature 1.0. Greedy decoding reproduced exactly in 53 of 54
repeated cells; the exception agreed on 498 of 500 items.

## Hardware

Lenovo ThinkPad E15 Gen 4, Intel Core i7-1255U, Windows 11. One machine, one
CPU, no GPU, throughout.

## Licence

Code is MIT. Data, results, tables and figures are CC BY 4.0. See `LICENSE`
and `LICENSE-DATA`; the source benchmarks keep their own terms.

## Citation

See `CITATION.cff`. The archived release carries a DOI; the manuscript is
under review.
