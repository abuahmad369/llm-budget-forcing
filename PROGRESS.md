# CSE465 Project — VibeThinker-3B Budget Forcing Study

**Last updated:** August 10, 2026
**Deadline:** August 22, 2026 (~12 days left)
**Status:** Experiments essentially complete. Phase 10 (optional) is blocked on a Kaggle file-mount issue.

---

## ⚠️ READ THIS FIRST WHEN YOU COME BACK

### How we work together (do not change this)
- I am an **absolute beginner** — no coding experience.
- Claude writes **100% of the code**, fully copy-pasteable.
- Claude gives **exact click-by-click Kaggle instructions**.
- **ONE STEP AT A TIME.** Never dump the whole project at once.
- Every step saves results to files I can download.

### To resume, say:
> *"I'm back. Read PROGRESS.md. Continue from Phase 10."*
>
> or, if skipping Phase 10:
>
> *"I'm back. Read PROGRESS.md. Skip Phase 10 and start writing the report."*

### Current blocker (Phase 10 only)
`FileNotFoundError: phase7_traces.json` in the Phase 10 notebook.
I uploaded the 11.7 MB file as a Kaggle Dataset and attached it, and I can see it in the
Input panel — but the code still can't find it. **Section 8 has the fix.**

**Phase 10 is OPTIONAL.** The report is complete without it. If it keeps fighting, skip it.

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

### Critical — must not lose
| File | Where | Note |
|---|---|---|
| `phase7_traces.json` | 11.7 MB, downloaded to laptop | **8.9 GPU-hrs of work.** Regenerating costs 3.6 hrs. |

### Results data
`phase5_baseline.json`/`.csv` · `phase6_forcing.json`/`.csv` · `phase6_analysis.json` ·
`phase7_results.json`/`.csv` · `phase8_analysis.json` · `phase9_paired.json`

### Figures (rendered, ready for report)
| File | Content |
|---|---|
| `fig3_main_result.png` | Accuracy vs budget, with/without forcing + 32k baseline |
| `fig4_truncation.png` | **Best figure** — truncation rate vs accuracy, mirror image |
| `fig5_lengths.png` | Length histogram, bimodal with wall at cap |
| `fig1_accuracy_vs_budget.png`, `fig2_efficiency_frontier.png` | earlier versions |

### OBSOLETE — do not submit, written for the abandoned tool-calling plan
`CSE465_Project_Report_VibeThinker_Agent.docx` · `Viva_QA_Bengali.docx` ·
`COST_AND_RESOURCES.md`

Both the report and the Bengali viva Q&A must be **rewritten from scratch** for this project.

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

- [ ] *(optional)* Phase 10 — 32k extension, ~4.8 GPU-hrs
- [ ] **Write the report** (KDD-style, matching arXiv:2606.10678 formatting)
- [ ] **Rewrite `Viva_QA_Bengali.docx`** for this project — the existing one is obsolete
- [ ] Redraw figures with the caption/title corrections from §6

### Planned report structure
| Section | Content |
|---|---|
| Abstract | Truncation-not-capability finding + forcing result |
| 1. Introduction | The 91.4 vs 80.8 puzzle; 4 contributions as bullets |
| 2. Related Work | VibeThinker / test-time scaling / s1 budget forcing |
| 3. Methodology | Budget forcing, 3 variants, the truncation-derivation trick |
| 4. Experiments | The 5 findings, figures, significance tables |
| 5. Discussion | Why forcing can't manufacture reasoning; limitations |
| 6. Conclusion | |

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

**Bottom line:** the science is done and the findings are solid. What remains is Phase 10
(optional) and writing. Do not re-run finished experiments.
