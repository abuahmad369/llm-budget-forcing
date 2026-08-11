"""Phase 5 - full AIME25 baseline at a 32k budget.

Establishes the reference point every later arm is compared against, and
produces the finding that drove the whole project: correct == finished.

Cost: ~8.9 GPU-hours on a Tesla T4.
Output: outputs/phase5_baseline.json / .csv
"""
import csv
import json
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import (MODEL, DATASET, TEMPERATURE, TOP_P, K, build_prompt,
                    extract_boxed, to_int, outdir)

outdir()
ds = load_dataset(DATASET)["test"]
assert len(ds) == 30, f"expected 30 problems, got {len(ds)}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Try the largest context the GPU will hold; fall back rather than crash.
llm, MAX_CTX = None, None
for ctx in (32768, 24576, 20480):
    try:
        print(f"trying max_model_len={ctx}")
        llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.90,
                  max_model_len=ctx, trust_remote_code=True)
        MAX_CTX = ctx
        break
    except Exception as e:
        print(f"  failed: {str(e)[:150]}")
if llm is None:
    raise RuntimeError("could not load model at any context size")
print(f"loaded at max_model_len={MAX_CTX}")

params = SamplingParams(n=K, temperature=TEMPERATURE, top_p=TOP_P,
                        max_tokens=MAX_CTX - 512)

rows, t0 = [], time.time()

# Chunked so a crash does not lose the whole run.
for start in range(0, 30, 10):
    idxs = list(range(start, min(start + 10, 30)))
    probs = [ds[i] for i in idxs]
    print(f"--- problems {idxs[0]}-{idxs[-1]} ---")

    outs = llm.generate([build_prompt(tokenizer, p["problem"]) for p in probs],
                        params)

    for p, out in zip(probs, outs):
        truth = to_int(str(p["answer"]))
        lens, preds, finished = [], [], 0
        for c in out.outputs:
            lens.append(len(c.token_ids))
            finished += (c.finish_reason == "stop")
            preds.append(to_int(extract_boxed(c.text)))
        valid = [x for x in preds if x is not None]
        rows.append({
            "id": int(p["id"]), "truth": truth, "lengths": lens,
            "finished": finished, "parsed": len(valid), "predictions": preds,
            "correct": sum(1 for x in valid if x == truth),
            "unique_answers": len(set(valid)),
        })
        print(f"  [{p['id']:>2}] truth={truth:>4} correct={rows[-1]['correct']}/{K} "
              f"finished={finished}/{K} maxlen={max(lens):,}")

    with open("outputs/phase5_baseline.json", "w") as f:
        json.dump(rows, f, indent=2)

elapsed = time.time() - t0
n = len(rows) * K
correct = sum(r["correct"] for r in rows)
fin = sum(r["finished"] for r in rows)

print()
print(f"Pass@1                 : {100 * correct / n:.1f}%")
print(f"Paper reports (AIME25) : 91.4%")
print(f"solved by ALL {K}        : {sum(1 for r in rows if r['correct'] == K)}/30")
print(f"solved by SOME         : {sum(1 for r in rows if 0 < r['correct'] < K)}/30")
print(f"samples disagreed      : {sum(1 for r in rows if r['unique_answers'] > 1)}/30")
print(f"finished naturally     : {fin}/{n} ({100 * fin / n:.0f}%)")
print(f"runtime                : {elapsed / 3600:.2f} GPU-hours")
print()
print("KEY CHECK - if `correct` tracks `finished` per problem, the bottleneck")
print("is termination, not reasoning.")

with open("outputs/phase5_baseline.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "truth", "correct", "K", "finished", "parsed",
                "unique_answers", "max_len", "predictions"])
    for r in rows:
        w.writerow([r["id"], r["truth"], r["correct"], K, r["finished"],
                    r["parsed"], r["unique_answers"], max(r["lengths"]),
                    r["predictions"]])
print("saved outputs/phase5_baseline.json and .csv")
