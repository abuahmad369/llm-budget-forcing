# HANDOFF: complete project state

**Last updated:** 27 August 2026
**Purpose:** everything needed to resume this project from zero context, on any
Claude account, with this conversation deleted. Nothing here depends on chat
history or assistant memory.

Read this file first. `PROGRESS.md` covers the experiment phases in more depth.
Where the two disagree, this file is newer.

---

## 0. HOW TO WORK WITH ME (do not change this)

Paste this to a new assistant at the start of a session.

> I am an absolute beginner. I do not write code. You write 100 percent of the
> code, fully copy pasteable, and you tell me exactly which buttons to click.
> Give me ONE step at a time and wait for my result before the next step. Every
> step should save its output to a file I can download.

Two standing preferences that matter:

**Prose style.** For anything another person reads, no dash characters at all in
prose. Not em dash, not en dash, not the plain hyphen. Rewrite compounds instead.
Sentence case headings. No AI vocabulary. There is a `human` skill installed at
`~/.claude/skills/human` that encodes the full rule set.

**Academic honesty, resolved 27 August.** History, because it matters for
anyone picking this up. On 24 August I asked whether a report could be written to
evade the Pangram AI detector. The assistant declined and kept declining, six
times, and told me the one thing that would change the answer was to go and check
what CSE 465 actually permits. On 27 August I did. The answer from the faculty is
that AI assistance in writing the report is allowed, and the requirement is that
the writing reads naturally. On that basis the assistant wrote the full IEEE
report prose, which is `report/main_ieee.tex`.

What this means going forward. The research, the experiments and every number are
mine. The report prose is AI written with my direction, under a course policy that
permits it. If a future submission for a different course or venue has a different
policy, that policy governs and this permission does not carry over. If the
department later asks for a disclosure line, add one; it costs nothing and the
work stands on its own either way.

---

## 1. WHAT THE PROJECT IS

**Title:** Reasoning Termination, Not Reasoning Capability, Bounds Small Language
Model Performance on Competition Mathematics.

**One line:** VibeThinker-3B scores 91.4 on AIME 2025 in its paper. I measured
80.8. The gap is not weaker reasoning, it is generations being cut off before the
answer gets written. Forcing the model to commit at the token limit recovers 7 to
13 points and has never destroyed a correct answer.

**Course:** CSE 465, North South University. Abu Ahmad, ID 2121725042.
Advisor: Dr. Nabeel Mohammed.

---

## 2. STATUS OF EVERY DELIVERABLE

| Item | State | Notes |
|:--|:--|:--|
| All experiments | DONE | 22.6 GPU hours, all logged |
| Public showcase repo | DONE, LIVE | See section 3 |
| GitHub Pages explorer | DONE, LIVE, verified HTTP 200 | Serves "The Budget Cut" |
| Interactive explorer HTML | DONE, verified in browser | No console errors, 3 theme states checked |
| Summary figure | DONE | Two layouts, wide and stacked |
| One page PDFs | DONE, verified by rendering | Two versions, with and without the figure |
| Bengali viva Q and A | DONE, verified 7 pages | Local only at `report/Viva_QA_Bengali.docx`, deliberately not in the repo |
| CV fact sheet | DONE | Produced 25 Aug, not saved to a file, regenerate if needed |
| IEEE report, full prose | WRITTEN 27 Aug, compile status [VERIFY] | `report/main_ieee.tex`, 6,435 words of prose, all 44 skeleton slots filled. Upload `report/ieee_report_overleaf.zip` to Overleaf and compile twice |
| Older NSU template report | SUPERSEDED | `report/main.tex`, kept for reference only. The IEEE version is the one to submit |
| Live Gradio demo | **WORKS.** Verified 24 Aug on a T4 | `notebooks/live_demo.ipynb` is the run record. See below |
| Phase 10, 32k plus forcing | NOT RUN | Blocked on a file mount issue, then dropped |
| Forced pass@4 | NOT MEASURED | Makes the oracle comparison not like for like |

### The live demo works, corrected 28 August

This section previously said the demo had never completed a successful run. That
was true when it was written and is now wrong. The Kaggle notebook was recovered
on 28 August and is saved as `notebooks/live_demo.ipynb`. Its output cells show a
full working session on 24 August:

- vLLM loaded VibeThinker-3B on a Tesla T4, float16, TRITON_ATTN backend, 18,432
  token context, 168,832 tokens of KV cache
- AIME 2025 loaded, 30 problems
- Gradio served on a public URL
- **36 generation calls, zero tracebacks, zero out of memory errors**

The alternating pattern in the output is the method working. A long generation at
roughly 60 tokens per second, then an instant call with very high input
throughput and about five output tokens, which is the forcing step re reading the
stored prefix and being made to commit.

The four fixes made by code review (budget slider clamped to engine capacity,
truncation detection on the transformers backend, a lock around concurrent
generation, and failing fast when the GPU is already occupied) were therefore
exercised against a live engine after all.

Still not done formally: the six check smoke test in `app/DEMO_GUIDE.md` section
1.2. What is proven is that the app launches, serves, generates and forces
without crashing. Whether every individual check in that list passes was never
recorded.

---

## 3. WHERE EVERYTHING LIVES

**Private working repo** (code, traces, report)
https://github.com/abuahmad369/llm-budget-forcing
Local: `D:\semester\14. Summer 2026\CSE465\ViveThinker upgration`
14 commits. Currently private.

**Public showcase repo** (no source code)
https://github.com/abuahmad369/reasoning-termination
Local: `D:\semester\14. Summer 2026\CSE465\showcase-public`
13 files, 2 MB. Contains README, figures, result JSON and CSV, two PDFs,
`index.html`.

**Live explorer**
https://abuahmad369.github.io/reasoning-termination/

**Claude artifact copy of the explorer**
https://claude.ai/code/artifact/557455a0-2cd3-4ec5-b793-0cc0ea8f6e94
(private to the account that published it, so the GitHub Pages link is the one to
share)

**Most expensive single file:** `results/phase7_traces.json`, 12 MB, all 120 full
reasoning traces at 16,384 tokens. Cost 3.6 GPU hours to generate and everything
downstream depends on it. Back it up outside the repo.

**Never downloaded from Kaggle and now probably lost:** `phase5_baseline.json`
(8.87 GPU hours) and `phase6_forcing.json` (2.06 GPU hours). The derived numbers
survive in this file and in `RESULTS.md`, so nothing in the report is at risk, but
the per sample data behind the baseline is gone unless a saved Kaggle version
still holds it.

---

## 4. EVERY MEASURED NUMBER

Setup for all of it: Kaggle free tier, one Tesla T4 (15.64 GB, compute capability
7.5), float16, vLLM, `math-ai/aime25`, 30 problems, K equals 4, so 120 samples per
condition. Temperature 1.0, top_p 0.95, seed 0.

### Accuracy against budget, no intervention

| Budget | Pass@1 | Truncated |
|:--|--:|--:|
| 1,024 | 3.3% | 100.0% |
| 2,048 | 13.3% | 98.3% |
| 3,072 | 23.3% | 92.5% |
| 4,096 | 30.0% | 87.5% |
| 6,144 | 38.3% | 77.5% |
| 8,192 | 46.7% | 67.5% |
| 10,240 | 51.7% | 63.3% |
| 12,288 | 56.7% | 59.2% |
| 14,336 | 58.3% | 53.3% |
| 16,384 | 63.3% | 48.3% |
| 32,768 | 80.8% | 20.8% |

Paper reports 91.4 uncapped.

### Effect of budget forcing

| Budget | No forcing | Forced | Gain | McNemar chi2 | p |
|:--|--:|--:|--:|--:|--:|
| 4,096 | 27.5% | 40.8% | +13.3 | not run | not run |
| 8,192 | 46.7% | 55.8% | +9.1 | 9.09 | 0.0026 |
| 16,384 | 63.3% | 70.8% | +7.5 | 7.11 | 0.0077 |

The 4k row mixes arms. Its baseline came from a dedicated 4k run while the curve
above is trace derived (30.0 percent). Do not plot the 4k forcing point beside 8k
and 16k as if it were the same experiment.

### Harm check, paired

| Budget | Truncated | Rescued b | Damaged c | Unchanged correct |
|:--|--:|--:|--:|--:|
| 8,192 | 81 | 11 | 0 | 17 |
| 16,384 | 58 | 9 | 0 | 14 |

Zero damage across 139 forced samples. 31 truncated traces already held a correct
boxed answer; forcing overwrote all 31 and reproduced the same correct value each
time. A hybrid rule that preserves an existing answer therefore scores identically
and is unnecessary. It was built, tested, and discarded.

### Composition of the 120 samples

| Condition | Correct | Wrong | No answer |
|:--|--:|--:|--:|
| 8k, no forcing | 56 | 0 | 64 |
| 8k, forced | 67 | 53 | 0 |
| 16k, no forcing | 76 | 0 | 44 |
| 16k, forced | 85 | 33 | 2 |

Across all 1,200 sample budget observations the model volunteered exactly ONE
wrong answer. It states the correct answer or states nothing. This is the sharpest
result in the project.

### Recovery rate among rescuable samples

| Budget | Truncated | Already correct | Rescuable | Rescued | Rate |
|:--|--:|--:|--:|--:|--:|
| 8,192 | 81 | 17 | 64 | 11 | 17.2% |
| 16,384 | 58 | 14 | 44 | 9 | 20.5% |

Random guessing on AIME (integers 0 to 999) succeeds 0.1 percent of the time, so
forcing is roughly 150 times better than chance.

### Upper bounds

| Budget | Unforced pass@4 | Forced pass@1 |
|:--|--:|--:|
| 4,096 | 40.0% | 40.8% |
| 8,192 | 53.3% | 55.8% |
| 16,384 | 73.3% | 70.8% |

Extraction ceiling captured: 78.6% at 8k, 81.8% at 16k. Wilson 95 percent
intervals roughly 52 to 92 and 52 to 95, because the ceiling population is only 14
and 11 samples. Only about 25 percent of rescuable traces mention the correct
answer anywhere in their final fifth, so roughly three quarters of failures never
derived it.

### Forcing prompt variants

| Budget | V1 bare | V3 deadline | V2 summarise |
|:--|--:|--:|--:|
| 8,192 | 55.8% | 55.8% | 56.7% |
| 16,384 | 70.8% | 70.8% | 70.8% |

V2 costs 2.1 times more compute (47.8 vs 21.2 minutes at 8k, 117.3 vs 54.9 at 16k)
for no measurable gain. The prompt wording does not matter.

### Engine throughput on the same T4

| Configuration | Tokens per second |
|:--|--:|
| HuggingFace transformers, K=1 | 15.2 |
| vLLM, K=1 | 31.4 |
| vLLM, K=8 | 145.9 |
| vLLM, K=32 | 246.4 |

16.2 times faster at K=32.

### Compute spent

| Stage | GPU hours |
|:--|--:|
| Setup, probes, diagnosis | 1.50 |
| Baseline at 32k | 8.87 |
| Forcing at 4k and 8k | 2.06 |
| 16k generation plus 6 forcing passes | 8.89 |
| Analysis and figures | 0.00 (CPU) |
| Paired analysis | 1.28 |
| Upper bounds | 0.00 (CPU) |
| **Total** | **22.60** |

### Other measured facts

Median trace length 15,961 tokens. vLLM KV cache on the T4 at 0.90 utilisation is
192,272 tokens. Methodology validation: an 8k condition measured two independent
ways gave 54.2 and 55.8 percent, a difference of two samples out of 120.

---

## 5. CORRECTIONS AND KNOWN ERRORS

Do not reintroduce any of these.

**A retracted metric.** An early version of `phase11_upper_bounds.py` reported a
capture rate of about 62 to 65 percent computed against `ceil_box`. That was
wrong. `ceil_box` turned out to equal the already correct set exactly, so
numerator and denominator were disjoint and the ratio was meaningless.
`phase11b_ceiling.py` supersedes it with 78.6 and 81.8 percent computed against
truncated AND wrong samples. The old script carries a warning at the top. The
retraction is documented publicly in the showcase README, deliberately.

**Printed output bugs, already corrected in the write ups.** Phase 6 printed
identical GPU hours for the forced and unforced arms; correct values are 0.38 and
0.52 at 4k, 1.18 and 1.54 at 8k. Phase 7 printed `TOTAL RUNTIME: 117 min` from a
reused variable; the real figure is 8.89 hours.

**Figure caption requirements.** `fig5_lengths.png` is right censored at 16,384
tokens. The spike at the cap is our limit, not the model stopping naturally. The
caption must say so. `fig4_truncation.png` originally said truncation "mirrors
accuracy exactly"; it does not, so the wording was softened to "move inversely".

**Cost framing.** Forcing's wall clock cost is prefill, from reprocessing each
stored trace. An implementation reusing the KV cache would make it nearly free.
Do not present the timings as an intrinsic cost of the method.

**Statistics.** Report McNemar, not the bootstrap. The bootstrap treats the gain
as unpaired when the design is paired. It was conservative so conclusions are
unchanged, but McNemar is the correct test.

**Estimates that were never measured.** Early in the project the assistant
produced an "expected improvements" table (for example CLR plus 3.0 to 97.3).
Those were guesses and are now known to be wrong, since CLR cannot work here at
all. Only measured numbers go anywhere.

---

## 6. WHY THE ORIGINAL PLAN WAS ABANDONED

The project started as adding tool calling and agentic loops to VibeThinker-3B,
composing Tool-N1, Search-R1, GiGPO, ARPO and ReTool. It was dropped on 7 August
for two independently fatal reasons, both measured.

Compute: median trace 15,961 tokens, K equals 4 cost 8.87 GPU hours, so K equals
32 (which the planned methods needed) projected to about 71 GPU hours against a 30
hour weekly cap.

No headroom: the baseline found samples disagreeing on only 1 of 30 problems. CLR,
ensembling and self refinement all work by selecting among disagreeing candidates.
There was nothing to select.

The upper bound work later confirmed the call quantitatively. A perfect selector
over four samples tops out at 53.3 percent at 8k; forcing reaches 55.8 percent. The
abandoned approach had a lower ceiling than what replaced it.

Superseded documents from that direction live in `archive/` with a README
explaining why. Do not quote numbers from them.

---

## 7. ENVIRONMENT GOTCHAS

**Kaggle sessions recycle and wipe pip installs.** Always run a session setup cell
first that reinstalls vLLM if missing.

**Save and Run All re runs every cell.** Never commit a notebook containing
finished expensive phases. Make a fresh notebook per run.

**Batch runner uses settings saved at commit time.** Attach inputs and enable GPU
before Save Version.

**Check four settings before every commit:** Accelerator GPU T4 x2, Internet ON,
Persistence Files only, Inputs attached.

**Run the launch cell ONCE.** Running it twice loads a second model while the
first still holds the GPU. `importlib.reload` does not free VRAM, only a kernel
restart does. This caused the one live demo failure.

**Newly attached Kaggle inputs need a kernel restart to be visible.**

**`glob('**')` does not reliably follow Kaggle's symlinked dataset mounts.** Use
`os.walk(..., followlinks=True)`.

**Harmless errors that look alarming:** the pip dependency resolver wall of red
(it only names Kaggle preinstalled extras we never import), and
`FA2 is only supported on devices with compute capability >= 8` (vLLM falls back
to Triton attention automatically on the T4).

**vLLM does not run natively on Windows.** Use WSL2 or the transformers backend.

**Local GPU is a GTX 1650, 4.29 GB.** float16 weights need 6.2 GB, so the model
cannot run locally without 4 bit quantisation, which changes outputs and is not
comparable to any reported number.

**The Bash tool on this machine mangles backslashes in heredocs.** Writing Python
containing `\\boxed{` or `\n` through a heredoc silently corrupts it. Build such
strings with `chr(92)` or use a file write tool instead. This bit three separate
times.

---

## 8. THE TRICK THAT MADE THE STUDY AFFORDABLE

Generation is autoregressive, so the first N tokens of a longer run are
distributionally identical to what a budget N run would have produced. Traces were
generated once at 16,384 tokens and saved; every smaller budget is scored by
truncating them on CPU at zero GPU cost.

This was validated, not assumed: an 8k condition measured both ways agreed to
within two samples of 120.

Consequence: `results/phase7_traces.json` can answer any question about any budget
up to 16k without touching a GPU. Most follow up analysis should start there.

---

## 9. WHAT IS LEFT

Nothing is blocking. In rough priority order:

1. Verify the live demo end to end on Kaggle using the smoke test in
   `app/DEMO_GUIDE.md`. It has three fixes that have never been exercised.
2. Confirm `report/main.tex` compiles cleanly in Overleaf. Upload
   `report/overleaf_upload.zip`, which contains the tex plus figures, and compile
   twice so the table of contents resolves.
3. Rewrite the report prose in my own words, per the decision in section 0.
4. Optional, about 4.8 GPU hours: extend the 16k traces to 32k to complete the
   matrix with a 32k plus forcing point. Predicted gain about plus 5 from the
   trend, but that is extrapolation. Design is in `PROGRESS.md`.
5. Optional, about 1.3 GPU hours: compute forced pass@4 so the oracle comparison
   in the report becomes like for like.

---

## 10. HOW TO RESTART A SPECIFIC THREAD

**Live demo.** `app/live_demo.py`, launch instructions in `app/KAGGLE_LAUNCH.md`,
demo script and smoke test in `app/DEMO_GUIDE.md`. Backends are vllm (Kaggle T4,
the one to demo), hf16 (8 to 12 GB card, Windows friendly) and hf4bit (under 8 GB,
not comparable to the report).

**Explorer rebuild.** `app/build_data.py` builds `data.json` from the traces, then
inject it into `app/template.html` at the `__DATA__` marker. Full instructions in
`app/README.md`. Two hazards documented there: the template must stay pure ASCII
or it renders as mojibake, and a split multi byte character leaves a U+FFFD that
the artifact deploy endpoint rejects.

**Figures.** `python app/make_summary_figure.py` emits both the wide and stacked
versions of `fig6`. `src/phase8_analysis.py` regenerates fig3, fig4 and fig5 from
the traces, CPU only.

**Printable sheets.** `app/build_onepager.js` and `app/build_tables_sheet.js`.
Both need the `docx` npm package, which is installed at
`C:/Users/abuah/node_modules`, so run with
`NODE_PATH="C:/Users/abuah/node_modules" node <script>`.

**Analysis phases.** `src/phase5_baseline.py` through `src/phase11b_ceiling.py`.
`src/common.py` holds the shared prompt builder, answer parser and forcing phrase.
If you change the parser or the forcing phrase there, change the duplicate copy in
`app/live_demo.py` too, or the demo stops measuring what the report measured.

---

## 11. KEY PAPERS

| Paper | arXiv | Role |
|:--|:--|:--|
| VibeThinker-3B | 2606.16140 | The model. Source of the 91.4 figure. |
| s1: Simple Test Time Scaling | 2501.19393 | Where budget forcing comes from |
| DeepSeek-R1 | 2501.12948 | Background on RL for reasoning |
| Self-Consistency | 2203.11171 | Ensembling, ruled out by our data |
| Self-Refine | 2303.17651 | Refinement, ruled out |
| Report style template | 2606.10678 | Formatting the LaTeX report follows |

---

**Bottom line:** the science is finished and the numbers are solid. What remains
is verification of the demo, a clean compile, and writing the report prose in my
own words.
