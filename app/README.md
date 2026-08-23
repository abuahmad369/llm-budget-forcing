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
