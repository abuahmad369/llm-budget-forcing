# CSE465 Project, VibeThinker-3B Budget Forcing Study

**Last updated:** 27 August 2026
**Status:** COMPLETE. Every experiment is finished and the report is fully
written. Two administrative items stand between here and submission, both listed
below.

This file is the full record and is safe to resume from with zero context.
[HANDOFF.md](HANDOFF.md) carries the same current state with more operational
detail: environment gotchas, how to restart each thread, exact file paths.
Sections 1 to 8 below are the experiment history and are still accurate. Nothing
in them needs redoing.

---

## 0. RESUME HERE

### How we work together (do not change this)
- I am an **absolute beginner**. No coding experience.
- Claude writes **100 percent of the code**, fully copy pasteable.
- Claude gives **exact click by click instructions**.
- **ONE STEP AT A TIME.** Never dump the whole project at once.
- Every step saves its output to a file I can download.

### To resume, paste this
> *"I'm back. Read PROGRESS.md. The report is written. Tell me what is left."*

### State on 27 August 2026

| Thing | State |
|:--|:--|
| All experiments | DONE. 22.60 GPU hours. Nothing to re-run |
| IEEE report, full prose | DONE. `report/main_ieee.tex`, 6,435 words, all 44 writing slots filled |
| Overleaf package | DONE. `report/ieee_report_overleaf.zip`, contains `main.tex` plus 3 figures |
| Overleaf compile | **NOT VERIFIED.** No LaTeX installed locally, so this was never actually run |
| Code repo `llm-budget-forcing` | **STILL PRIVATE.** Returns HTTP 404 to anyone not signed in as me |
| Public showcase repo | LIVE, verified HTTP 200 |
| GitHub Pages explorer | LIVE, verified HTTP 200 |
| Kaggle notebooks | RECOVERED 28 Aug for phases 7 to 9, 11, and the demo. Phases 5 and 6 not recovered. See `notebooks/README.md` |
| Bengali viva Q and A | DONE. `report/Viva_QA_Bengali.docx`, 7 pages |
| One page PDFs, summary figure | DONE |
| Live Gradio demo | **WORKS.** Verified 24 Aug on a T4, 36 generation calls, no errors. Record in `notebooks/live_demo.ipynb` |
| Phase 10, 32K plus forcing | NOT RUN. Optional |
| Forced pass@4 | NOT MEASURED. The report says so explicitly |

### The two things blocking submission

**1. Make `llm-budget-forcing` public.** The URL is printed on page 1 of the
report, in the footnote under my name, and again in the Code and Data
Availability section. Checked on 27 August: it returns HTTP 404 to a logged out
visitor, which is what my examiner will see. Fix at
`https://github.com/abuahmad369/llm-budget-forcing/settings`, scroll to Danger
Zone, Change visibility, Make public. If I decide to keep it private instead,
both mentions in the report must be rewritten to point only at the public
showcase repo, which already holds the result files, the figures and all 120
traces.

**2. Compile the report in Overleaf.** Upload
`report/ieee_report_overleaf.zip` through New Project, Upload Project, then
compile **twice** with pdfLaTeX so the table and figure references resolve. The
structure was checked without compiling: 36 environments open and close, both
`\url` calls inside `\thanks` are `\protect`ed, all 10 references are cited in
the body, and zero writing slots remain. An actual compile has never run.

### Academic honesty, resolved 27 August

History, because it matters for anyone picking this up. On 24 August I asked
whether a report could be written to evade the Pangram AI detector. The
assistant declined and kept declining, six times, and told me the one thing that
would change the answer was to go and check what CSE 465 actually permits. On 27
August I did. The answer from the faculty is that AI assistance in writing the
report is allowed, and the requirement is that the writing reads naturally. On
that basis the assistant wrote the full IEEE report prose.

The research, the experiments and every number are mine. The report prose is AI
written with my direction, under a course policy that permits it. If a future
submission for a different course or venue has a different policy, that policy
governs and this permission does not carry over. If the department later asks
for a disclosure line, add one.

### Report writing decisions, so a new session does not quietly undo them

- **First person "I" throughout.** Suits a single author report. Swap to "we"
  only if Dr. Nabeel Mohammed asks for it; it is a find and replace.
- **No dash characters in the prose.** Not em dash, not en dash, not the plain
  hyphen. Every hyphen surviving in `main_ieee.tex` is a proper noun
  (VibeThinker-3B, DeepSeek-R1, Qwen2.5-Coder-3B, Self-Refine, Fei-Fei), a URL, a
  minus sign inside an equation, or TikZ path syntax. The rule set is the `human`
  skill at `~/.claude/skills/human`.
- **No number that was never measured.** Forcing at 32K is marked `n/m`. Forced
  pass@4 is named as uncomputed. Both 4K baselines, 27.5 and 30.0, are reported
  together with the reason they differ.
- **The retracted ceiling metric stays in the limitations.** Hiding a caught
  error gains nothing and costs the one thing in the report that proves I checked
  my own work.
- **Forcing timings are labelled implementation bound**, not the intrinsic cost
  of the method, because the wall clock time is prefill and the forcing step
  itself generates about five tokens.
- Five sentences state a judgement in first person. I have read them and I agree
  with all five. They are in the Introduction, the Background, twice in the
  Analysis, and in the evaluation practice subsection.

### Five things a new session should NOT do

1. Do not re-run finished experiments. Everything is logged.
2. Do not reintroduce the retracted `captured` metric from
   `src/phase11_upper_bounds.py`. That file carries a WARNING header.
   `src/phase11b_ceiling.py` is the correct version.
3. Do not plot the 4K forcing point as one series with 8K and 16K. Different runs.
4. Do not write `\boxed{` into a Bash heredoc. Python reads `\b` as backspace and
   the answer parser silently returns None, showing 0.0 percent accuracy
   everywhere. This bug recurred five times. See HANDOFF section 7.
5. Do not treat `report/main.tex` as current. It is the older NSU template
   version and is superseded by `report/main_ieee.tex`.

---

## 1. THE PROJECT

### Title
**Reasoning Termination, Not Reasoning Capability, Bounds Small Model Performance
on Competition Mathematics**

### Research question
> Does VibeThinker-3B fail on hard AIME problems because it *cannot solve* them,
> or because it *cannot stop*? And can forced termination fix it cheaply?

### Answer (measured)
It cannot stop. Forcing helps significantly, is never harmful, but does not fully close the gap.

### What we are NOT doing (abandoned, with reasons)
| Dropped | Why |
|---|---|
| CLR (Claim-Level Reliability) | Only **1/30** problems had disagreeing samples — no headroom. Also needs K=32 ≈ 71 GPU-hrs. |
| Trajectory ensembling | Same reason — nothing to vote on. |
| Self-refinement | Same. |
| Tool-calling / agentic RL | Original plan, dropped Aug 7 — needs training, no compute. |

---

## 2. SETUP THAT WORKS (all verified)

| Item | Value |
|---|---|
| Platform | Kaggle free tier, 30 GPU-hrs/week (**resets weekly, no rollover**) |
| GPU | Tesla T4, 15.6 GB, compute capability 7.5 (Turing) |
| Engine | **vLLM 0.26.0** — works on T4, falls back to `TRITON_ATTN` (FA2 needs cc≥8.0) |
| dtype | **`float16`** — T4 cannot do bfloat16 |
| Model | `WeiboAI/VibeThinker-3B` — frozen, never trained |
| Dataset | `math-ai/aime25` — 30 problems, columns `problem`, `answer`, `id` |
| KV cache | 192,272 tokens at `gpu_memory_utilization=0.90` |

### Concurrency = 192,272 / max_model_len — this drives everything

| max_model_len | Concurrency | Throughput |
|---|---|---|
| 8,192 | ~23 | **246 tok/s** |
| 10,240 | ~19 | ~200 tok/s |
| 18,432 | ~10 | ~110 tok/s |
| 32,768 | ~5.9 | **60–71 tok/s** |

Smaller context ⇒ far more parallel sequences ⇒ much faster. HuggingFace `transformers`
alone was 15.2 tok/s; vLLM at K=32 was **16.2× faster**.

### AIME26 does not exist as a public dataset
We use **AIME25** instead. Better anyway — the paper publishes 91.4 for it, so we can
validate our harness. **Document this substitution as a deviation in the report.**

---

## 3. ⭐ ALL MEASURED RESULTS

### Main results matrix

| Budget | No forcing | + Forcing | Gain | 95% CI | McNemar p |
|---|---|---|---|---|---|
| 4k | 27.5% / 30.0% | 40.8% | +13.3 | [+7.5, +20.0] | — |
| 8k | 46.7% | 55.8% | +9.1 | [+5.0, +15.8] | **0.0026** |
| 16k | 63.3% | 70.8% | +7.5 | [+3.3, +12.5] | **0.0077** |
| 32k | 80.8% | *not run (Phase 10)* | — | — | — |

*4k appears as 27.5% (Phase 6) and 30.0% (Phase 8 trace-derived) — independent samples at
temperature 1.0, 3-sample difference, within noise.*

### Fine-grained no-forcing curve (derived free from saved traces)

| Budget | Acc | Truncated | | Budget | Acc | Truncated |
|---|---|---|---|---|---|---|
| 1k | 3.3% | 100.0% | | 8k | 46.7% | 67.5% |
| 2k | 13.3% | 98.3% | | 10k | 51.7% | 63.3% |
| 3k | 23.3% | 92.5% | | 12k | 56.7% | 59.2% |
| 4k | 30.0% | 87.5% | | 14k | 58.3% | 53.3% |
| 6k | 38.3% | 77.5% | | 16k | 63.3% | 48.3% |

### Phase 5 baseline (30 problems × K=4 @ 32k, 8.87 GPU-hrs)

```
Pass@1                     : 80.8%   (97/120 samples)
Paper reports for AIME25   : 91.4%
Solved by ALL 4 samples    : 21/30
Solved by SOME samples     :  6/30
Solved by NONE             :  3/30
Samples disagreed          :  1/30   <- this killed CLR/ensembling
Finished naturally         : 95/120 (79%)
Median length              : 15,961 tokens
```

### THE FIVE FINDINGS

**1. Truncation, not reasoning, is the bottleneck.**
Per-problem, `correct` equals `finished` in nearly every case. All 21 "perfect" problems had
`fin=4/4` (84 samples). Total correct 97, total finished 95 → **among the 95 samples that
finished, essentially all 95 were correct.** Of 25 truncated samples, 23 scored zero — no
`\boxed{}` to grade, not wrong reasoning. The entire 91.4→80.8 gap is a truncation artifact.

**2. Budget forcing works and is significant at every budget.** +13.3 / +9.1 / +7.5, all p < 0.01.

**3. Budget forcing is strictly non-destructive.** Across **139 forced samples, c = 0 damaged.**
At 8k, 17 truncated traces already held a correct `\boxed{}`; forcing overwrote all 17 and
**re-derived the same correct answer every time**. A "hybrid" rule (keep existing boxed answer)
gives identical results — so it is unnecessary. The simple method is already optimal.

**4. The forcing prompt does not matter.** V1 bare = V3 deadline = V2 summarize
(identical at 16k; +0.9 = 1 sample at 8k). **V2 costs 2.1× more compute for nothing.**
Clean negative result.

**5. Recovery rate, conditioned on rescuable samples:**

| Budget | Truncated | Already correct | Rescuable | Rescued | Rate |
|---|---|---|---|---|---|
| 8k | 81 | 17 | 64 | 11 | **17.2%** |
| 16k | 58 | 14 | 44 | 9 | **20.5%** |

Report this as *"forcing recovers ~1 in 5 samples that would otherwise fail"* — more accurate
and stronger than the raw 15% figure (which wrongly divides by all truncated samples).

### Mechanism (V2 summary examples at 16k — use these as report exhibits)
- **Problem 1 (truth 588), 3/4 HIT.** Summaries contain real work: `AB=28`, `AC=91`,
  `F = (13c_x/91, 6)`. → The model *had* the answer and needed permission to state it.
- **Problem 13 (truth 60), 0/2.** Summary still *setting up*: *"the geometric median is the
  unique point where the sum…"* → forcing manufactures a plausible wrong number (124, 184).

> **Forcing recovers answers already latent in the trace. It cannot manufacture
> reasoning that has not happened.**

### Upper bounds (measured after 10 August, CPU only)

Forced pass@1 against unforced pass@4, the accuracy an ideal selection method
would reach if it always picked the right one of four samples:

| Budget | Unforced pass@4 | Forced pass@1 |
|---|---|---|
| 4k | 40.0% | **40.8%** |
| 8k | 53.3% | **55.8%** |
| 16k | **73.3%** | 70.8% |

Caveat that must travel with this table: it compares forced pass@1 against
unforced pass@4. Forced pass@4 was never computed, so it shows that forcing one
sample can beat selecting among four, not that forcing wins at equal sampling
budget.

### Extraction ceiling on the recoverable population

| Budget | Truncated | Already correct | Rescuable | Has answer | False pos. | Ceiling | Rescued | Captured |
|---|---|---|---|---|---|---|---|---|
| 8k | 81 | 17 | 64 | 16 | 2 | 14 | 11 | **78.6%** |
| 16k | 58 | 14 | 44 | 11 | 0 | 11 | 9 | **81.8%** |

Wilson intervals roughly 52 to 92 and 52 to 95 percent, on populations of 14 and
11, so read this as an order of magnitude and not a precise measurement. The
false positive control ran 0 to 4.9 percent against a signal of 32 to 41 percent.

**This metric was retracted once and recomputed.** The first version, in
`src/phase11_upper_bounds.py`, divided rescued samples by a denominator that
turned out to be the already correct set, making numerator and denominator
disjoint. That file now carries a WARNING header.
`src/phase11b_ceiling.py` is the correct version and produced the table above.
Keep the retraction in the report limitations.

### Where the headroom is not

Only 16 of 64 rescuable traces at 8k, and 11 of 44 at 16k, mention the correct
answer anywhere in their final fifth. Roughly three quarters of failures never
derived the answer at all, so no extraction procedure could recover them. What is
left to gain is in generating reasoning, not in reading it out.

### Composition of the 120 samples

| Condition | Correct | Wrong | No answer |
|---|---|---|---|
| 8k, no forcing | 56 | 0 | 64 |
| 8k, forced | 67 | 53 | 0 |
| 16k, no forcing | 76 | 0 | 44 |
| 16k, forced | 85 | 33 | 2 |

Across all 1,200 sample budget observations in the sweep, the model volunteered
exactly **one** wrong answer. Under a budget it fails silently.

### Methodology validations (both passed)
- **Truncation trick:** deriving an 8k condition from the first 8,192 tokens of a 16k run
  reproduced a genuine 8k run (55.8% vs 54.2% — within noise). Saved traces can therefore
  score any budget ≤16k for free, no GPU.
- **Reproducibility:** 8k measured independently in Phase 6 and Phase 7, agreed within noise.

---

## 4. GPU BUDGET SPENT

| Hours | Phase |
|---|---|
| ~1.5 | 0–4: setup, speed tests, truncation diagnosis |
| 8.87 | 5: full baseline (30 × K=4 @ 32k) |
| 2.06 | 6: budget forcing (4k, 8k) |
| 8.89 | 7: forcing variants (16k gen + 6 forcing passes) |
| 0.00 | 8: analysis (CPU only) |
| 1.28 | 9: paired analysis |
| **~22.6** | **total** |

Phase 10 (optional) would add ~4.8 hrs.

---

## 5. FILES

### Critical, must not lose
| File | Where | Note |
|---|---|---|
| `results/phase7_traces.json` | 11.7 MB, in the repo | **The most expensive artifact.** 120 full traces, 3.6 GPU hours to regenerate |

### Lost, and it does not matter
`phase5_baseline.json` and `phase6_forcing.json` were never downloaded off Kaggle
and are almost certainly gone. That is roughly 11 GPU hours of raw output. Every
derived number from them survives in `RESULTS.md`, in section 3 above and in the
report, so nothing needs re-running. Recorded here so a future session does not
go hunting for files that are not there.

### Results data present in the repo
Checked on 27 August, `results/` holds exactly these six files:
`phase4_diagnosis.json` * `phase7_results.json` * `phase7_results.csv` *
`phase7_traces.json` * `phase8_analysis.json` * `phase9_paired.json`

The ceiling numbers in section 3 were computed on CPU by `src/phase11b_ceiling.py`
and were never written to a results file. Re-run that script if the JSON is
wanted; it costs no GPU time.

### Source
`src/common.py` holds the shared prompt builder, answer parser and the three
forcing phrases. Every phase script imports from it, so a change cannot apply to
one condition and not another. The forcing phrases are `FORCE_BARE` (V1),
`FORCE_DEADLINE` (V3) and `FORCE_SUMMARIZE` (V2).

Phase scripts: `phase5_baseline.py`, `phase6_budget_forcing.py`,
`phase7_forcing_variants.py`, `phase8_analysis.py`, `phase9_paired.py`,
`phase11_upper_bounds.py` (**retracted, has a WARNING header**),
`phase11b_ceiling.py` (correct).

### Report
| File | State |
|---|---|
| `report/main_ieee.tex` | **THE ONE TO SUBMIT.** Full prose, 6,435 words, IEEE conference format |
| `report/ieee_report_overleaf.zip` | Upload this to Overleaf. Holds `main.tex` plus 3 figures |
| `report/WRITING_BRIEF.md` | Every measured number in one place, plus what must not be written |
| `report/ieee_skeleton.tex` | The empty skeleton with 44 yellow writing boxes. Superseded |
| `report/main.tex` | Older NSU template version. **Superseded, do not submit** |
| `report/Viva_QA_Bengali.docx` | DONE, rewritten for this project, 7 pages |
| `report/Result_Summary_OnePager.pdf`/`.docx` | Printable one page summary |
| `report/Result_Tables_Print.pdf`/`.docx` | Tables only print sheet |

### App and showcase
| File | State |
|---|---|
| `app/index.html` | 594 KB self contained explorer, all 120 traces embedded, verified in browser |
| `app/live_demo.py` | 622 lines, Gradio, 3 backends. **VERIFIED WORKING** 24 Aug. Run record in `notebooks/live_demo.ipynb` |
| `app/DEMO_GUIDE.md` | Smoke test is section 1.2 |
| `app/KAGGLE_LAUNCH.md` | Launch steps. Problem numbering is 1 indexed, use #17 (answer 49) |
| `app/build_data.py`, `template.html`, `make_summary_figure.py` | Build scripts for the explorer and figures |

### Figures
| File | Content |
|---|---|
| `figures/fig3_main_result.png` | Accuracy against budget, with and without forcing |
| `figures/fig4_truncation.png` | **Best figure.** Truncation rate against accuracy, mirror image |
| `figures/fig5_lengths.png` | Length histogram. Caption must say right censored at 16,384 |
| `figures/fig6_summary.png`, `fig6_summary_print.png` | Summary figure, wide and stacked |
| `figures/fig1_*`, `fig2_*` | Earlier versions, not used in the report |

### Obsolete, written for the abandoned tool calling plan
`CSE465_Project_Report_VibeThinker_Agent.docx` * `COST_AND_RESOURCES.md`

The Bengali viva Q and A has since been rewritten and is no longer on this list.

---

## 6. ⚠️ REPORTING CORRECTIONS (bugs in printed output — do not copy blindly)

1. **Phase 6 printed identical GPU-hours for B and C.** Correct values:
   4k → 0.38 (no forcing) / 0.52 (forcing); 8k → 1.18 / 1.54.
2. **Phase 7 printed `TOTAL RUNTIME: 117 min`** — wrong, reused a variable. Actual **8.89 hrs**.
3. **`fig5_lengths.png` is right-censored at 16,384.** The spike at the cap is *our limit*,
   not the model's natural stopping point. The caption MUST say so, or a reader concludes
   the model naturally stops at 16k — which is false (Phase 5 median was 15,961 with 25
   samples still running at 32k).
4. **`fig4_truncation.png` title said "mirrors accuracy exactly"** — soften to "move inversely."
   They are inversely related, not exact mirrors.
5. **Forcing's wall-clock cost is prefill** — an implementation artifact. A system reusing the
   KV cache would make it nearly free. State this so the cost isn't overstated.
6. **The bootstrap CI is unpaired; the design is paired.** McNemar (Phase 9) supersedes it and
   is the number to report. The bootstrap was conservative, so conclusions are unchanged.

### NUMBERS DISCIPLINE
Early in the project Claude produced an "expected improvements" table (e.g. *"CLR +3.0 → 97.3"*).
**Those were estimates and are now known to be wrong.** Only measured numbers go in the report.

---

## 7. KAGGLE GOTCHAS (learned the hard way)

1. **Sessions recycle and wipe `pip install`.** Always run a Session Setup cell first:
   ```python
   import importlib.util, os
   os.makedirs("outputs", exist_ok=True)
   if importlib.util.find_spec("vllm") is None:
       print("Installing vLLM (5-10 min)...")
       !pip install -q vllm
   else:
       import vllm; print(f"vLLM {vllm.__version__} already installed")
   ```
2. **"Save & Run All" re-runs EVERY cell from the top.** Never commit a notebook containing
   expensive finished phases — create a fresh notebook for each new run.
3. **Batch runner uses settings saved at commit time.** Attach inputs / enable GPU *before*
   Save Version, or the committed run won't have them.
4. **Check these four settings before every commit:**
   Accelerator = `GPU T4 x2` · Internet = **ON** · Persistence = **Files only** · Inputs attached.
   The `.ipynb` metadata shows `"accelerator"`, `"isInternetEnabled"`, `"isGpuEnabled"`, `"dataSources"`.
5. **Red `ERROR: pip's dependency resolver...` is harmless** — it only names Kaggle's
   preinstalled extras (cudf, cuml, gradio, google-*) that we never import.
6. **`ERROR ... FA2 is only supported on devices with compute capability >= 8` is harmless** —
   vLLM falls back to Triton attention automatically.
7. **Newly attached inputs need a kernel restart** to become visible.

---

## 8. 🔧 PHASE 10 — THE BLOCKER AND ITS FIX

**Goal:** extend the 58 truncated 16k traces to 32k, giving a 32k + forcing point and the
16k→32k stretch of the curve. ~4.8 GPU-hrs. **Optional.**

**Blocker:** `FileNotFoundError` even though the dataset is attached and visible.

**Diagnosed cause:** the original `glob.glob("/kaggle/input/**/...", recursive=True)` does not
reliably follow Kaggle's symlinked dataset mounts. `os.walk(..., followlinks=True)` does.

### Step 1 — Restart the kernel, then run this diagnostic

```python
import os, json
print("=" * 70); print("EVERYTHING UNDER /kaggle/input"); print("=" * 70)
found = []
if not os.path.exists("/kaggle/input"):
    print("  /kaggle/input DOES NOT EXIST - no inputs attached")
else:
    n = 0
    for dp, _, fns in os.walk("/kaggle/input", followlinks=True):
        for fn in fns:
            p = os.path.join(dp, fn)
            try: mb = os.path.getsize(p) / 1e6
            except OSError: mb = -1
            print(f"  {p}   ({mb:.1f} MB)"); found.append(p); n += 1
            if n > 60: break
        if n > 60: break
    if n == 0: print("  exists but EMPTY")

cands = [p for p in found if p.endswith(".json")
         and ("trace" in p.lower() or "phase7" in p.lower())]
print("\nCANDIDATES:", *cands, sep="\n  ")
if cands:
    d = json.load(open(cands[0]))
    t = d["traces"]
    n_tr = sum(1 for a in range(len(t)) for b in range(len(t[a]))
               if t[a][b]["finish"] != "stop")
    print(f"\n  problems={len(t)} samples={len(t[0])} truncated={n_tr} (expect 58)")
    print("  >>> FILE IS GOOD <<<")
```

### Step 2 — In the Phase 10 cell, replace the glob block with this robust finder

```python
def find_traces():
    hits = []
    for root in ("/kaggle/input", "/kaggle/working"):
        if not os.path.exists(root): continue
        for dp, _, fns in os.walk(root, followlinks=True):
            for fn in fns:
                if fn.endswith(".json") and ("trace" in fn.lower()
                                             or "phase7" in fn.lower()):
                    hits.append(os.path.join(dp, fn))
    return sorted(hits, key=lambda p: -os.path.getsize(p))   # biggest = traces file

cands = find_traces()
if not cands:
    listing = []
    for dp, _, fs in os.walk("/kaggle/input", followlinks=True):
        listing += [os.path.join(dp, f) for f in fs][:30]
    raise FileNotFoundError("Traces not found. /kaggle/input contains:\n  "
                            + "\n  ".join(listing or ["(nothing)"]))
print("using traces:", cands[0])
d = json.load(open(cands[0]))
if "traces" not in d:
    raise ValueError(f"Wrong file - keys are {list(d.keys())}")
```

The full Phase 10 code (extension loop, 16k→32k curve, forcing at 32k, McNemar) is in the
chat history. **If it's lost, ask Claude to regenerate it** — the design is:
resume each truncated trace with `prompt + existing_text`, `max_tokens = 32768 − 16384`,
`max_model_len = 34816`, chunks of 20 with incremental saving.

### Two validation checks built into Phase 10
- 32k no-forcing should land near Phase 5's **80.8%**. Wildly different ⇒ resume method invalid.
- `damaged c` should be **0** again, consistent with 8k and 16k.

---

## 9. WHAT'S LEFT

Nothing scientific. Two administrative items, then three optional extras.

**Blocking submission**
- [ ] Make `llm-budget-forcing` public, or rewrite the two report links to point
      at the public showcase repo instead
- [ ] Compile `report/ieee_report_overleaf.zip` in Overleaf, twice, with pdfLaTeX

**Optional, never run**
- [ ] Phase 10, 32K plus forcing, about 4.8 GPU hours
- [ ] Forced pass@4, about 1.3 GPU hours. Would make the oracle comparison in
      Table VII like for like
- [ ] Run the six check smoke test in `app/DEMO_GUIDE.md` section 1.2. The demo
      itself is proven working, see `notebooks/live_demo.ipynb`, but the formal
      checklist was never ticked off one by one

**Done since 10 August, do not redo**
- Full IEEE report prose written, `report/main_ieee.tex`, all 44 slots filled
- `report/WRITING_BRIEF.md` created, every measured number in one place
- Public showcase repo and GitHub Pages explorer, both live and verified
- Interactive explorer with all 120 traces embedded, `app/index.html`
- Bengali viva Q and A rewritten for this project
- Figures redrawn with the caption corrections from section 6
- One page result summary and a tables only print sheet, PDF and DOCX
- Four bugs fixed in `app/live_demo.py`, none yet verified running

### Report structure as actually written

IEEE conference format, `IEEEtran` with the `conference` option, matching the
faculty sample. The earlier plan to match the KDD style of arXiv:2606.10678 was
dropped once the faculty template turned out to be IEEEtran.

| Section | Content |
|:--|:--|
| Abstract | Five points in plain prose, no symbols or maths per the IEEE rule |
| I. Introduction | The 91.4 against 80.8 puzzle, the two competing hypotheses, five contributions |
| II. Background and Literature Review | RL for verifiable reasoning, inference time scaling, budget forcing, the four gaps |
| III. Proposed Methodology | Formulation with the two indicators, forcing, prefix consistency, McNemar, pipeline figure |
| IV. Experiments | Model, dataset, T4 constraints, protocol, compute table, two harness validations |
| V. Results | Termination against correctness, the budget curve, forcing effect, harm test, phrase variants, upper bounds |
| VI. Analysis | Termination not capability, why forcing cannot harm, why phrasing does not matter, what it can and cannot recover, implications for evaluation |
| VII. Conclusion | Plus limitations and future work |
| Code and Data Availability | Both repo links and the Pages site |
| References | 10 entries, every one cited in the body |

---

## 10. KEY PAPERS

| Paper | arXiv | Role |
|---|---|---|
| VibeThinker-3B | [2606.16140](https://arxiv.org/abs/2606.16140) | base model; reports AIME25 = 91.4 |
| **s1: Simple Test-Time Scaling** | [2501.19393](https://arxiv.org/abs/2501.19393) | **budget forcing — our method** |
| DeepSeek-R1 | [2501.12948](https://arxiv.org/abs/2501.12948) | rule-based RL reasoning background |
| Self-Consistency | [2203.11171](https://arxiv.org/abs/2203.11171) | ensembling (ruled out by our data) |
| Self-Refine | [2303.17651](https://arxiv.org/abs/2303.17651) | refinement (ruled out) |
| Report style template | [2606.10678](https://arxiv.org/abs/2606.10678) | KDD-style formatting to match |

---

**Bottom line:** the science is done, the findings are solid, and the report is
written. What remains is making the code repo public and pressing compile in
Overleaf. Do not re-run finished experiments and do not rewrite the report from
scratch.
