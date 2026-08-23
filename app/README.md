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

| Backend | Where | float16? | Comparable to the report |
|---|---|---|---|
| `vllm` | Kaggle T4 | yes | **yes - demo this one** |
| `hf4bit` | local 4 GB card | no, NF4 | no, approximation only |

VibeThinker-3B needs **6.2 GB** in float16. A GTX 1650 has **4 GB**, so float16
will not load. 4-bit NF4 compresses the weights to about 2 GB, which fits, but
quantization changes what the model writes - so local numbers show the mechanism
without matching the tables. The interface says so on screen rather than letting
anyone assume otherwise.

**See [KAGGLE_LAUNCH.md](KAGGLE_LAUNCH.md)** for the two cells to paste and a
demo script that builds to the interesting failure case.

## Local run on a 4 GB card

```bash
pip install gradio transformers accelerate bitsandbytes
python live_demo.py --backend hf4bit
```

Keep the budget at **1024-2048** and **K = 1-2**. A GTX 1650 decodes this model
at roughly 8-15 tokens per second, so K=2 at 2048 tokens is about three minutes
and 4096 tokens is over ten. It will not OOM at those settings; it will simply
be slow.

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
