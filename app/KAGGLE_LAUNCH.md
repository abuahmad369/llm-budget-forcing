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

**1. "Run one problem" tab.** Pick `AIME25 #02` (answer 16), budget **2048**,
K = 2. It will be cut off with no answer. Forcing recovers it. This is the whole
thesis in about 40 seconds.

**2. Raise the budget to 8192** on the same problem. Now it finishes on its own
and forcing changes nothing. This shows forcing is a recovery mechanism, not a
crutch.

**3. Pick `AIME25 #14`** (answer 60), budget **4096**. Cut off, and forcing
produces a *wrong* answer. Do not skip this one. It shows forcing cannot invent
reasoning that never happened, and volunteering that limitation is worth more in
a viva than hiding it.

**4. "Sweep budgets" tab.** Budgets `1024, 2048, 4096, 8192`, K = 2. Draws the
accuracy-against-budget curve live, the same shape as Figure 4.2.

**5. "Reproduce the benchmark" tab.** 5 problems, budget 4096, K = 2. A few
minutes. Prints your measured accuracy beside the reported figures.

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
