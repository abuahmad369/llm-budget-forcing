# Running the live demo on Kaggle

This is the configuration to demo. float16 on a T4 with vLLM is identical to
every number in the report, so what your examiner sees is directly comparable to
the tables.

## Setup

**Notebook settings** (right panel, before you run anything):

| Setting | Value |
|---|---|
| Accelerator | `GPU T4 x2` |
| Internet | **ON** (required - Gradio needs it for the public link) |
| Persistence | Files only |

## Cell 1 - install and fetch

```python
!pip install -q vllm gradio
!curl -sL https://raw.githubusercontent.com/abuahmad369/llm-budget-forcing/main/app/live_demo.py -o live_demo.py
print("ready")
```

Takes 5-10 minutes. The red `ERROR: pip's dependency resolver...` block names
only Kaggle's preinstalled extras and is safe to ignore.

**Then restart the kernel**: `Run` -> `Restart & clear cell outputs`.

## Cell 2 - launch

```python
import sys, importlib
sys.argv = ["live_demo.py", "--backend", "vllm", "--share"]
import live_demo
importlib.reload(live_demo)
live_demo.main()
```

The model loads (2-4 minutes), then Gradio prints two URLs:

```
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxxxxxxxxxx.gradio.live
```

**Open the `gradio.live` link** on your laptop, or share it with your examiner.
It stays alive for 72 hours or until the Kaggle session ends.

Leave Cell 2 running. Stopping it kills the app.

## What to show, in order

Verified against the measured traces. The dropdown is 1-indexed, so `#17` is the
seventeenth problem and its answer is 49.

**1. `AIME25 #17`, budget 2048, K=2.** All samples cut off with no answer, then
forcing recovers **49**. Three of the four stored traces finish by ~2250 tokens,
so 2048 cuts it about a hundred tokens from the finish line - the cleanest
possible rescue. About 40 seconds.

**2. Same problem, budget 4096.** Now it finishes on its own and forcing changes
nothing. Shows forcing is a recovery mechanism, not a crutch.

**3. `AIME25 #14` (answer 60), budget 4096.** Cut off, and forcing produces a
confident *wrong* answer. Do not skip this. It shows forcing cannot invent
reasoning that never happened, and volunteering that is worth more in a viva
than having it found for you.

**4. "Sweep budgets" tab.** `1024, 2048, 4096`, K=2. Draws the accuracy curve
live, same shape as Figure 4.2.

**5. "Reproduce the benchmark" tab.** 5 problems, budget 4096, K=2. Prints your
measured accuracy beside the reported figures.

**Warm the model up first.** The first generation after loading compiles Triton
kernels and is noticeably slower - your Kaggle logs show the JIT latency spike.
Run anything once at budget 1024 and discard it before presenting.

Full script, smoke tests and failure recovery: **[DEMO_GUIDE.md](DEMO_GUIDE.md)**.

## Expected runtimes on a T4

| What | Roughly |
|---|---|
| One problem, K=2, 2048 tokens | 20-40 s |
| One problem, K=2, 8192 tokens | 1.5-3 min |
| Sweep of 4 budgets, K=2 | 3-5 min |
| Reproduce, 5 problems, K=2, 4096 | 4-8 min |

Have the sweep and reproduce runs finished *before* the viva starts, so the
results are on screen and you are not watching a progress bar in front of an
examiner.

## Honest framing for the reproduce tab

A 5-problem subset will not land exactly on the 30-problem figure - with K=2
that is 10 samples, and one problem moves the number by 20 points. Say so before
they ask. The claim is that forcing moves accuracy up, and that it never moves a
correct answer down; the exact value needs the full run in the report.
