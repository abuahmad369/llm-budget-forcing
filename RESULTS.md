# Results

Every number here was measured. Nothing is estimated or projected unless
explicitly labelled. See `PROGRESS.md` for the full lab notebook.

---

## Main matrix

| Budget | No forcing | + Forcing | Gain | 95% CI (bootstrap) | McNemar *p* |
|---|---|---|---|---|---|
| 4k  | 27.5% / 30.0% | 40.8% | +13.3 | [+7.5, +20.0] | — |
| 8k  | 46.7% | 55.8% | +9.1 | [+5.0, +15.8] | **0.0026** |
| 16k | 63.3% | 70.8% | +7.5 | [+3.3, +12.5] | **0.0077** |
| 32k | 80.8% | not measured | — | — | — |

4k is listed twice because it was measured two independent ways — a genuine 4k
generation (Phase 6, 27.5%) and truncation of 16k traces (Phase 8, 30.0%). The
2.5-point difference is 3 samples, within sampling noise at temperature 1.0.

**Report McNemar, not the bootstrap.** The bootstrap treats the gain as unpaired;
the design is paired (same 120 samples scored twice). It was conservative, so the
conclusion is unchanged, but McNemar is the correct test.

---

## Fine-grained no-forcing curve

Derived from saved traces at zero GPU cost.

| Budget | Pass@1 | Truncated | | Budget | Pass@1 | Truncated |
|---|---|---|---|---|---|---|
| 1k | 3.3% | 100.0% | | 8k  | 46.7% | 67.5% |
| 2k | 13.3% | 98.3% | | 10k | 51.7% | 63.3% |
| 3k | 23.3% | 92.5% | | 12k | 56.7% | 59.2% |
| 4k | 30.0% | 87.5% | | 14k | 58.3% | 53.3% |
| 6k | 38.3% | 77.5% | | 16k | 63.3% | 48.3% |

Accuracy exceeds the *finish* rate by 12–18 points at every budget, because the
model writes intermediate `\boxed{}` answers mid-reasoning and some are already
correct. The no-forcing baseline picks these up.

---

## Phase 5 baseline (30 problems × K=4 at 32k, 8.87 GPU-hours)

```
Pass@1                      80.8%   (97/120)
Paper reports for AIME25    91.4%

solved by ALL 4 samples     21/30
solved by SOME samples       6/30
solved by NONE               3/30
samples disagreed            1/30   <- this ruled out CLR and ensembling
finished naturally          95/120  (79%)
median trace length         15,961 tokens
```

### The central observation

Per problem, `correct` equals `finished` almost everywhere:

| Problem | correct | finished |
|---|---|---|
| 8 | 2/4 | 2/4 |
| 13 | 0/4 | 0/4 |
| 14 | 0/4 | 0/4 |
| 19 | 2/4 | 2/4 |
| 26 | 3/4 | 3/4 |
| 27 | 1/4 | 1/4 |
| 29 | 0/4 | 0/4 |

All 21 perfect problems had `fin=4/4` (84 samples). Total correct 97, total
finished 95 → **among the 95 samples that finished, essentially all 95 were
correct**. Of 25 truncated samples, 23 scored zero.

---

## Paired analysis (Phase 9)

| Budget | Truncated | Rescued (b) | Damaged (c) | right→right | McNemar χ² | *p* |
|---|---|---|---|---|---|---|
| 8k  | 81 | 11 | **0** | 17 | 9.09 | 0.0026 |
| 16k | 58 |  9 | **0** | 14 | 7.11 | 0.0077 |

**Zero damage across 139 forced samples.** 31 truncated traces already held a
correct `\boxed{}`; forcing overwrote all 31 and re-derived the same correct
answer each time. A hybrid rule (keep the existing answer, force only when
absent) scores identically — so it is unnecessary.

---

## Forcing-variant comparison (Phase 7)

| Budget | V1 bare | V3 deadline | V2 summarize |
|---|---|---|---|
| 8k  | 55.8% | 55.8% | 56.7% |
| 16k | 70.8% | 70.8% | 70.8% |

Identical at 16k. The 0.9-point spread at 8k is one sample out of 120.

**The prompt does not matter.** And V2 costs more for nothing:

| Budget | V1 cost | V2 cost |
|---|---|---|
| 8k  | 21.2 min | 47.8 min |
| 16k | 54.9 min | 117.3 min |

2.1× the compute, no gain.

---

## Recovery rate, conditioned on rescuable samples

| Budget | Truncated | Already correct | Rescuable | Rescued | Rate |
|---|---|---|---|---|---|
| 8k  | 81 | 17 | 64 | 11 | **17.2%** |
| 16k | 58 | 14 | 44 |  9 | **20.5%** |

Report this, not the raw 15% figure — the raw version wrongly divides by all
truncated samples including ones that were already correct.

---

## Upper bounds (Phase 11)

### Forcing vs. the selection oracle

`pass@4` is the hard ceiling for any method that *selects* among existing samples —
CLR, majority voting, self-consistency all live under it.

| Budget | Unforced pass@4 (oracle) | Forced pass@1 |
|---|---|---|
| 4k  | 40.0% | **40.8%** |
| 8k  | 53.3% | **55.8%** |
| 16k | **73.3%** | 70.8% |

At 4k and 8k, **one forced sample beats the best of four unforced samples.** Forcing
exceeds the selection ceiling because it does not choose among existing answers — it
generates a new one conditioned on partial reasoning. At 16k the oracle wins, which
fits the mechanism: more samples finish naturally, so selection has more to work with.

This quantifies why the CLR/ensembling direction was abandoned. Even a *perfect*
selector over 4 samples scores 53.3% at 8k; forcing scores 55.8%.

**Caveat:** this compares forced pass@1 against unforced pass@**4**. Forced pass@4 was
not computed — it requires per-sample forced predictions, which are in
`phase9_paired.json` rather than the traces. The like-for-like comparison is missing.

### How much of the extraction ceiling does forcing capture?

Restricted to the population forcing can actually act on: truncated **and wrong**.

| Budget | Truncated | Already correct | Rescuable | Answer in tail | FP | Net ceiling | Rescued | Captured | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| 8k  | 81 | 17 | 64 | 16 | 2 | 14 | 11 | **78.6%** | [52%, 92%] |
| 16k | 58 | 14 | 44 | 11 | 0 | 11 |  9 | **81.8%** | [52%, 95%] |

**Forcing captures ~80% of the recoverable signal.** The ceiling population is small
(n=11–14) so the interval is wide, but the lower bound (~52%) is well clear of the
40% threshold at which better extraction would be worth pursuing.

**The 4k row is excluded.** It mixes arms — `forced=40.8%` came from a genuine 4k run
(baseline 27.5%) while the ceiling is trace-derived (baseline 30.0%). Its apparent
100% is coincidence.

### The remaining headroom is not in extraction

Only **16/64** rescuable traces at 8k and **11/44** at 16k contain the correct answer
anywhere in their final 20%. **Roughly 75% of failures never derived the answer at
all** — no extraction method can recover those.

False-positive control (searching for a *different* problem's answer) ran at 0–4.9%
against 32–41% signal, so text matching is 8–12× above noise and the ceiling estimate
is trustworthy.

> The bottleneck is generating the reasoning, not reading it out.

### Retracted metric

An earlier version of Phase 11 reported a `captured` figure of ~62–65% computed
against `ceil_box` (traces where the answer appears in *any* `\boxed{}`). That was
wrong: `ceil_box` turned out to equal the *already-correct* set exactly — 14 traces
at 16k, matching `right→right = 14` from Phase 9 — so it measured samples the
baseline already scored, not ones forcing could rescue. Numerator and denominator
were disjoint sets. Superseded by the table above.

---

## Mechanism

**Problem 1, truth 588, at 16k — 3 of 4 forced samples HIT.** The summary contains
real derived work:

> *"AD=4, DE=16, EB=8 (so AB=28), AF=13, FG=52, GC=26 (so AC=91)… F = (13c_x/91, 6)"* → **588** ✅

**Problem 13, truth 60 — both forced samples miss.** The summary is still setting
up the problem:

> *"The geometric median is the unique point where the sum…"* → **124**, **184** ❌

> Forcing recovers answers already latent in the trace. It cannot manufacture
> reasoning that has not happened.

---

## Compute

| Hours | Phase |
|---|---|
| ~1.5 | 0–4 setup, speed tests, truncation diagnosis |
| 8.87 | 5 baseline |
| 2.06 | 6 budget forcing (4k, 8k) |
| 8.89 | 7 forcing variants |
| 0.00 | 8 analysis (CPU) |
| 1.28 | 9 paired analysis |
| **22.6** | **total** |

Engine throughput on the T4, measured:

| Setting | Throughput |
|---|---|
| HuggingFace `transformers`, K=1 | 15.2 tok/s |
| vLLM, K=1 | 31.4 tok/s |
| vLLM, K=8 | 145.9 tok/s |
| vLLM, K=32 | **246.4 tok/s** |

Concurrency = `kv_cache_tokens / max_model_len` = `192,272 / max_model_len`, which
is why a smaller context is dramatically faster.

---

## Not measured

- 32k + forcing. The trend (+13.3 → +9.1 → +7.5) extrapolates to roughly +5, but
  this is an extrapolation.
- Upper bounds (`src/phase11_upper_bounds.py` is written but was not run). Without
  it, the recovery rate has no ceiling to be compared against.
- Run-to-run variance. Single seed throughout.
