# The Budget Cut - interactive results explorer

Live: https://claude.ai/code/artifact/557455a0-2cd3-4ec5-b793-0cc0ea8f6e94

A single self-contained HTML page for demoing the finding. Drag the token budget
across 120 real reasoning traces, watch accuracy collapse as truncation rises,
then toggle budget forcing and watch part of it come back.

**Nothing here is simulated.** Every value is measured. VibeThinker-3B cannot run
in a browser, so the page replays stored traces rather than performing inference.

## Files

| File | What it is |
|---|---|
| `template.html` | The page. `__DATA__` is the injection point. |
| `build_data.py` | Builds `data.json` from the measured artifacts. |
| `data.json` | Compact payload, ~0.57 MB. Generated - do not hand-edit. |
| `index.html` | `template.html` with `data.json` injected. Generated. |

## Rebuilding

```bash
cd app
pip install tokenizers
curl -sL https://huggingface.co/WeiboAI/VibeThinker-3B/resolve/main/tokenizer.json -o _tok.json
curl -s "https://datasets-server.huggingface.co/rows?dataset=math-ai%2Faime25&config=default&split=test&offset=0&length=30" -o _aime_raw.json
python build_data.py
python -c "import io; BS=chr(92); t=io.open('template.html',encoding='utf-8').read(); d=io.open('data.json',encoding='utf-8').read().replace('</','<'+BS+'/'); io.open('index.html','w',encoding='utf-8').write(t.replace('__DATA__',d))"
```

## Two things that will bite you if you edit this

**Keep `template.html` pure ASCII.** Non-ASCII literals render as mojibake when the
page is served without a charset header. Write them as `&mdash;` in HTML or
`\u2014` in JavaScript.

**Strip U+FFFD from the payload.** Truncating a trace at an arbitrary token can
split a multi-byte character, and the decoder emits a replacement character. The
Artifact deploy endpoint rejects the file if one survives.

## What the page shows

- Budget scrubber over the ten measured budgets. Forcing data exists per-sample
  only at 8k and 16k; at every other stop the toggle disables itself rather than
  implying data we do not have.
- A 30-tile matrix, four cells per tile - one per sampled attempt. Green correct,
  red wrong, amber cut off with no answer written.
- Accuracy against budget, with truncation rate mirrored behind it.
- Detail panel: the problem in serif, the trace tail in mono ending at a labelled
  cut bar, and the unforced and forced verdicts side by side.

The line under the matrix carries the sharpest result. Left alone the model
volunteered exactly **one** wrong answer across all 1,200 sample-budget
observations - it either states the correct answer or states nothing. Forcing
converts that silence into a commitment, which at 8k is right 11 times and wrong
53 times.

---

# Live demo - `live_demo.py`

The explorer above replays stored traces. This one **runs the model**: pick or
paste a problem, set a budget, and watch it get cut off and then rescued.

## Which machine

The weights need **6.2 GB in float16**. That single number decides everything.

| VRAM | OS | Backend | Precision | Comparable to the report |
|---|---|---|---|---|
| 16 GB (Kaggle T4) | Linux | `vllm` | float16 | **yes - the reference** |
| 12 GB+ | Linux / WSL2 | `vllm` | float16 | yes |
| 8-12 GB | Windows | `hf16` | float16 | yes, different engine |
| 8-12 GB | Linux / WSL2 | `vllm` | float16 | yes, but cramped |
| under 8 GB | any | `hf4bit` | 4-bit NF4 | **no - approximation** |

Pass `--backend auto` (the default) and it measures free VRAM, reports what will
fit, and picks for you. It also sizes the context window to the card instead of
assuming, since vLLM preallocates its KV pool and a window too large on a small
card leaves nothing to batch with.

**vLLM does not run natively on Windows.** On Windows either use `hf16`, or
install vLLM inside WSL2. For a demo two days out, `hf16` is the sane choice.

**See [KAGGLE_LAUNCH.md](KAGGLE_LAUNCH.md)** for the two cells to paste and a
demo order that builds to the interesting failure case.

## Running on an 8 GB card

float16 fits, so results stay comparable to the report. On Windows:

```bash
pip install gradio transformers accelerate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # if torch has no CUDA
python live_demo.py --backend hf16
```

It prints a preflight block before loading anything:

```
GPU        : NVIDIA GeForce RTX 4060
VRAM       : 8.00 GB total, 7.21 GB free
Compute cap: 8.9
vLLM       : not installed (Linux/WSL2 only)
float16    : fits, about 14565 tokens of KV cache left
```

If it says `DOES NOT FIT`, close whatever else is using the GPU - a browser with
hardware acceleration on can hold half a gigabyte - and rerun.

**Settings for a live demo: budget 2048, K = 2.** An 8 GB card decodes this
model at roughly 20-35 tokens per second through transformers, so that is about
two to three minutes per run. Budget 4096 doubles it; 8192 is too slow to stand
in front of.

The trade against Kaggle: same weights and same precision, but transformers
generates sequentially where vLLM batches, so it is slower per run. Individual
samples may differ from vLLM because the attention kernels differ - the
distribution is the same model, the specific numbers on one run are not
guaranteed to match.

## Running on a 4 GB card

```bash
pip install gradio transformers accelerate bitsandbytes
python live_demo.py --backend hf4bit
```

4-bit NF4 compresses the weights to about 2 GB, which fits, but quantization
changes what the model writes - so these numbers show the mechanism without
matching the tables, and the interface says `comparable to report: NO` on screen.
Keep the budget at **1024-2048** and **K = 1-2**; a GTX 1650 manages roughly 8-15
tokens per second.

## The three tabs

**Run one problem.** One generation pass, scored twice - as written, and again
after appending the commit phrase to any sample that ran out of budget. Shows
per-sample tokens, whether it terminated, both answers, and whether forcing
rescued or damaged that sample.

**Sweep budgets.** The same problem at several budgets, plotted live. Reproduces
the shape of Figure 4.2 on whatever problem you choose.

**Reproduce the benchmark.** Runs the first N problems end to end and prints the
measured accuracy beside the reported figures. This is the tab for *why should I
believe your table*.

## Keeping it honest

`FORCE_BARE`, `extract_boxed`, and `to_int` in `live_demo.py` are copied verbatim
from `src/common.py`. **If you edit one, edit both** - otherwise the demo stops
measuring the same quantity the report measured, and the comparison is void.

Forcing only ever touches a sample that failed to finish. A sample that stopped
on its own is passed through untouched, which is why the damaged count is zero.
