# Demo guide

Everything needed to test the live app and present it. Read the whole thing once
before you touch the app in front of anyone.

---

## Part 1 - Test it works (do this now, not on the day)

### 1.1 Warm the model up first

**The first generation after loading is much slower than the rest.** vLLM
compiles Triton attention kernels on first use - your own Kaggle logs show
`Triton kernel JIT compilation during inference ... causes a latency spike`.
If your first click in front of an examiner is also the model's first
generation, you will stare at a progress bar for a minute with no explanation.

So: open the app, run **any** problem at budget 1024, K=1, and throw the result
away. Now the kernels are compiled and everything afterwards runs at full speed.

### 1.2 Smoke test - six checks, about five minutes

Work down this list. If any check fails, stop and fix it before going further.

| # | Do this | Expected | If it fails |
|---|---|---|---|
| 1 | Open the `gradio.live` link | Page loads, header says `Backend: vLLM / float16` and `comparable to report: yes` | Link expired or the Kaggle cell stopped. Re-run Cell 2. |
| 2 | Open the **Problem** dropdown | 31 entries: `Custom problem` then `AIME25 #01` ... `#30` | AIME fetch failed - check Internet is ON in Kaggle settings |
| 3 | Pick `AIME25 #17`. | Statement box fills, **Known answer** shows `49` | Dropdown handler broken - report it |
| 4 | Budget **2048**, K **2**, click Run | Finishes in ~20-40 s. Result table appears with 2 rows | See "If it breaks" below |
| 5 | Open **Reasoning traces** | Shows the tail of each trace, and for cut samples a line saying what forcing replied | - |
| 6 | Switch to **Sweep budgets**, run `1024, 2048` with K=1 | A plot appears with two lines | matplotlib missing - `pip install matplotlib` |

### 1.3 What a healthy result looks like

For **#17 at budget 2048, K=2**, expect roughly:

```
| sample | tokens | stopped  | no forcing | with forcing | verdict |
|      1 |   2048 | cut off  |     -      |      49      | RESCUED |
|      2 |   2048 | cut off  |     -      |      49      | RESCUED |
```

Accuracy **0% without forcing, 100% with**. That is the entire thesis in one
screen: the model had solved it and simply had not written the answer down.

Because the app samples at temperature 1.0, it generates **new** traces each
time - it is not replaying anything. Lengths and answers will vary slightly run
to run. That is honest randomness, not a bug, and it is worth saying out loud.

### 1.4 Timings on a T4 (measure your own and write them here)

| Configuration | Roughly |
|---|---|
| K=2, budget 2048 | 20-40 s |
| K=2, budget 4096 | 40-80 s |
| Sweep of 4 budgets, K=2 | 3-5 min |
| Reproduce, 5 problems, K=2, 4096 | 4-8 min |

**Run the sweep and the reproduce tab before the viva starts** and leave the
results on screen. Nobody wants to watch a progress bar for six minutes.

---

## Part 2 - What each tab does

### Tab 1: Run one problem

The core demonstration. One generation pass, scored twice.

1. Generates K samples at the budget you set.
2. Any sample that hits the budget without finishing gets the commit phrase
   appended, and the model is asked once more for just the answer.
3. Both scores are shown side by side.

**Samples that finished on their own are never touched.** This is why the
damaged count is zero - forcing only ever acts on a sample that had already
failed.

Reading the table:

| Column | Meaning |
|---|---|
| `tokens` | How many tokens that sample generated |
| `stopped` | `finished` = model stopped by itself; `cut off` = hit the budget |
| `no forcing` | Answer parsed from the trace as written. `-` means none was written |
| `with forcing` | Answer after the commit phrase. Same as left if it finished |
| `verdict` | `RESCUED` wrong/absent to correct. `DAMAGED` correct to wrong. `no change` |

**Accuracy is over K samples only.** At K=2 the only possible values are 0%,
50%, 100%. That is coarse by design - the demo shows the mechanism; the report's
tables have the statistics.

### Tab 2: Sweep budgets

Runs the same problem at several budgets and plots accuracy against budget, with
and without forcing. This reproduces the *shape* of Figure 4.2 live, on whatever
problem the examiner picks.

Use `1024, 2048, 4096` with K=2 for a demo. Adding 8192 roughly doubles the wait.

### Tab 3: Reproduce the benchmark

**This is the tab that answers "why should I believe your table."** It runs the
first N problems of AIME 2025 end to end, measures accuracy with and without
forcing, and prints your result directly beneath the reported figures.

Be upfront about the honest caveat before anyone raises it: 5 problems at K=2 is
**10 samples**, and one problem swings the number by 10-20 points. It will not
land exactly on 46.7%. The claim being demonstrated is the *direction* - forcing
moves accuracy up and never moves a correct answer down. The exact value needs
the full 30-problem run in the report.

---

## Part 3 - The presentation, step by step

Total time about 6 minutes. Every problem below is verified against the measured
traces.

### Step 0 - before they arrive

- Model warmed up (Section 1.1)
- Sweep and reproduce results already generated and left on screen
- Backup tab open: the replay explorer (cannot fail, needs no GPU)
- Laptop set to never sleep; Kaggle cell still running

### Step 1 - frame the question (30 s)

> "This model scores 91.4 on AIME 2025 in its paper. I reproduced 80.8. The
> gap is not that it reasons worse on my hardware - it is that it never finishes
> writing the answer. Let me show you."

### Step 2 - the rescue (about 40 s)

**Tab 1. Problem `AIME25 #17`. Budget 2048. K = 2. Run.**

Both samples are cut off. `no forcing` shows `-`. Say:

> "No answer. Scored zero. But look at the trace."

Open **Reasoning traces** and scroll to the end of a sample - it is mid-
calculation, clearly close to done. Then point at `with forcing`: **49**, the
correct answer.

> "Same generation. Same tokens. All I did was tell it to stop and commit."

### Step 3 - prove it is a recovery mechanism, not a crutch (about 60 s)

**Same problem. Budget 4096. Run.**

Now both samples say `finished`, and `no forcing` and `with forcing` both show
49, verdict `no change`.

> "Given enough room it stops on its own, and forcing does nothing. It only ever
> touches a sample that already failed - which is why it can't damage anything."

### Step 4 - volunteer the limitation (about 60 s)

**Problem `AIME25 #14` (answer 60). Budget 4096. Run.**

Cut off, and forcing produces a confident wrong number.

> "Here it never got near the answer, so forcing invents one. It recovers answers
> the model has already worked out - it cannot manufacture reasoning that never
> happened. In my data about three quarters of failures are this kind."

**Do not skip this step.** An examiner who finds the limitation themselves will
weigh it far more heavily than one you hand them.

### Step 5 - the numbers (about 90 s)

**Tab 3, results already on screen.** Walk through your measured row against the
reported row. Then state the statistics from the report, which the demo is too
small to show:

- Gain is significant at every budget: **p = 0.0026** at 8k, **0.0077** at 16k,
  by McNemar's test for paired data
- Across **139** forced samples, **zero** correct answers were destroyed
- Forcing captures about **80%** of the answers actually recoverable from
  truncated traces

### Step 6 - close (30 s)

> "Any benchmark that caps generation on a reasoning model is measuring reasoning
> ability and stopping behaviour together, and reporting it as reasoning ability.
> Forcing is worth 7 to 13 points and costs about five tokens."

---

## Part 4 - Questions they will ask at the screen

**"Is it actually running, or replaying a recording?"**
Running. Temperature is 1.0, so run it twice and the traces differ. Do exactly
that if asked - it is the most convincing thing you can do.

**"Run one I choose."**
Fine. Pick from the dropdown, or paste any problem into the box. If it has no
known answer, leave that field blank - the app will show both answers and say it
cannot score accuracy.

**"Why only 2 samples?"**
Time. K=4 doubles the wait. The report uses K=4 across all 30 problems; this is a
live demo, not the experiment.

**"Your reproduce number doesn't match your report."**
Expected, and say why: 10 samples versus 480. One problem moves a 10-sample
average by 10-20 points. The direction and the zero-damage result are what the
demo shows; the value is in the report.

**"Isn't forcing just guessing?"**
No - and there is a number for it. AIME answers are integers 0-999, so guessing
lands at 0.1%. Forcing recovers about 17-20% of otherwise-failed samples, which
is roughly 150 times chance. It is reading something real out of the trace.

**"Could you just use a bigger budget instead?"**
Yes, and accuracy does climb - that is the red curve. But 32k tokens per sample
costs about six times what 8k does. Forcing gets part of that back for
essentially nothing.

**"Does this work on other models?"**
Unknown, and it is in the limitations. One model, one benchmark, one seed. The
method transfers directly; whether the result does is untested.

---

## Part 5 - If it breaks live

Rehearse this once so you are not improvising.

| Symptom | Do this |
|---|---|
| Page will not load | Kaggle session died. Say "the cloud session timed out, here are the results from the run I did earlier" and switch to the replay explorer. |
| Run hangs past 3 minutes | Budget too high, or the session lost its GPU. Refresh, drop to budget 1024, K=1. |
| `CUDA out of memory` | Re-run Cell 2 to reload the model clean. |
| Wrong or odd answer appears | Say so plainly. Temperature 1.0 means individual samples vary; that is why the report averages 480 of them. |
| Total failure | Fall back to the replay explorer, then the report figures. |

**Always have the replay explorer open in another tab.** It is static HTML with
120 real traces baked in - no GPU, no session, no network. It cannot fail.

And know the three numbers cold, so you can present with no screen at all:
**80.8 without forcing, 70.8 with forcing at 16k, zero answers damaged.**
