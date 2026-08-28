# Kaggle notebooks

These are the notebooks as they ran on Kaggle, downloaded with their output cells
intact. They are the execution record behind the numbers in the report. The clean
library versions of the same code live in [`../src/`](../src); these files show
the runs.

Every notebook ran on the Kaggle free tier, one Tesla T4, 15.64 GB, compute
capability 7.5, float16, vLLM.

| File | Original Kaggle name | What it actually ran |
|:--|:--|:--|
| `phase7to9_variants_analysis_paired.ipynb` | `vibethinker-phase7-forcing-variants` | Phases 7, 8 and 9: the three forcing phrasings at 8K and 16K, deep trace analysis, and the paired McNemar test |
| `phase11_upper_bounds_and_ceiling.ipynb` | `vibethinker-phase10-32k` | Phases 11 and 11b: the pass@4 oracle comparison and the extraction ceiling, both CPU only |
| `live_demo.ipynb` | `cse465` | The interactive Gradio demo, launched on a T4 and driven through 36 generation calls |
| `phase0_baseline.ipynb` | `phase0-baseline` | Earliest feasibility probe, August 8 |
| `phase0_baseline_v0.ipynb` | `phase0-baseline-v0` | First version of the same probe |

Two of the files were renamed because the Kaggle title did not match the code
inside. Both original names are recorded above so the notebooks can still be
matched to the Kaggle history.

## Things a reader should know before trusting the output cells

**`phase11_upper_bounds_and_ceiling.ipynb` is not a 32K experiment.** The Kaggle
notebook is titled `vibethinker-phase10-32k` because it started life as the
planned 32K extension, which was never run. The code that survives in it computes
the upper bounds and the extraction ceiling from stored traces. Phase 10 remains
unrun and the report says so.

That notebook also contains the retraction described in the report limitations.
Cell 4 prints the first ceiling estimate, which reported 64.7 percent captured at
8K and 64.3 at 16K. That calculation used a denominator which turned out to
coincide with the already correct set, so its numerator and denominator were
disjoint sets and the ratio meant nothing. Cell 5 is the corrected version and
prints 78.6 percent at 8K and 81.8 at 16K, which are the figures the report uses.
Both cells are kept deliberately. The error and its correction are part of the
record.

**The Phase 7 cell in `phase7to9_variants_analysis_paired.ipynb` shows an
error.** The saved version carries a protobuf `VersionError` from a Kaggle image
update, raised when that cell was re-executed after the results already existed.
The Phase 7 results themselves came from an earlier successful run and are in
[`../results/phase7_results.json`](../results/phase7_results.json) and
[`../results/phase7_traces.json`](../results/phase7_traces.json). The Phase 8 and
Phase 9 cells below it ran cleanly in this same session and their output is
genuine.

**`live_demo.ipynb` is the proof that the demo works.** It loads
VibeThinker-3B under vLLM on a T4, serves the Gradio app on a public URL, and
runs 36 generation calls with no traceback and no out of memory error. The
alternating pattern in the output is the method itself: a long generation at
roughly 60 tokens per second, then an instant call with very high input
throughput and about five output tokens, which is the forcing step re reading the
stored prefix and being made to commit.

## What is missing

Notebooks for phases 5 and 6, the 32K baseline and the first forcing measurement,
were not recovered from Kaggle. Those two runs cost roughly 11 GPU hours. Every
number derived from them survives in [`../RESULTS.md`](../RESULTS.md), in
[`../PROGRESS.md`](../PROGRESS.md) section 3 and in the report, and the code that
produced them is in [`../src/phase5_baseline.py`](../src/phase5_baseline.py) and
[`../src/phase6_budget_forcing.py`](../src/phase6_budget_forcing.py).
