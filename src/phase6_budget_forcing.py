"""Phase 6 - first budget-forcing measurement at 4k and 8k.

One generation per budget, scored twice: once ignoring truncated samples, once
after forcing them to commit. Halves the compute versus running the arms
separately.

Superseded by phase 7, which derives both budgets from a single 16k generation.
Kept because it is the independent replication that validated the truncation
trick (8k measured here at 54.2%, derived in phase 7 at 55.8%).

Cost: ~2.1 GPU-hours on a Tesla T4.
Output: outputs/phase6_forcing.json / .csv
"""
import csv
import json
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import (MODEL, DATASET, TEMPERATURE, TOP_P, K, build_prompt,
                    extract_boxed, to_int, outdir, FORCE_BARE)

BUDGETS = (4096, 8192)
FORCE_TOKENS = 256

outdir()
ds = load_dataset(DATASET)["test"]
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Keep the context small: concurrency = kv_cache_tokens / max_model_len, so a
# tight context gives far more parallel sequences and much higher throughput.
llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.90,
          max_model_len=max(BUDGETS) + FORCE_TOKENS + 1792,
          trust_remote_code=True)

prompts = [build_prompt(tokenizer, ds[i]["problem"]) for i in range(30)]
truths = [to_int(str(ds[i]["answer"])) for i in range(30)]
results, examples = {}, []

for budget in BUDGETS:
    print(f"\n=== budget {budget:,} ===")

    t0 = time.time()
    outs = llm.generate(prompts, SamplingParams(
        n=K, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=budget))
    gen_time = time.time() - t0
    gen_tokens = sum(len(c.token_ids) for o in outs for c in o.outputs)
    print(f"  generation {gen_time / 60:.1f} min, {gen_tokens:,} tokens, "
          f"{gen_tokens / gen_time:.0f} tok/s")

    plain, jobs = [], []
    for pi, out in enumerate(outs):
        row = []
        for ci, c in enumerate(out.outputs):
            row.append(to_int(extract_boxed(c.text)))
            if c.finish_reason != "stop":
                jobs.append((pi, ci, prompts[pi] + c.text + FORCE_BARE))
        plain.append(row)

    acc0 = sum(1 for pi in range(30) for p in plain[pi] if p == truths[pi])
    print(f"  truncated {len(jobs)}/{30 * K}")
    print(f"  no forcing   {100 * acc0 / 120:.1f}%")

    forced = [r[:] for r in plain]
    force_time = force_tokens = 0.0
    if jobs:
        t0 = time.time()
        fo = llm.generate([j[2] for j in jobs],
                          SamplingParams(n=1, temperature=0.0,
                                         max_tokens=FORCE_TOKENS))
        force_time = time.time() - t0
        force_tokens = sum(len(o.outputs[0].token_ids) for o in fo)
        for (pi, ci, _), o in zip(jobs, fo):
            forced[pi][ci] = to_int(o.outputs[0].text)
            if len(examples) < 4:
                examples.append({"problem": pi, "truth": truths[pi],
                                 "continuation": o.outputs[0].text[:300],
                                 "parsed": forced[pi][ci]})

    acc1 = sum(1 for pi in range(30) for p in forced[pi] if p == truths[pi])
    print(f"  with forcing {100 * acc1 / 120:.1f}%  "
          f"({100 * (acc1 - acc0) / 120:+.1f})")
    print(f"  forcing generated only {force_tokens:,} tokens - the wall-clock")
    print(f"  cost is prefill, an artifact of re-reading each trace.")

    results[budget] = {
        "budget": budget, "truncated": len(jobs),
        "no_forcing_pct": round(100 * acc0 / 120, 1),
        "forcing_pct": round(100 * acc1 / 120, 1),
        "gain": round(100 * (acc1 - acc0) / 120, 1),
        "gen_tokens": gen_tokens, "force_tokens": force_tokens,
        "gen_hours": round(gen_time / 3600, 2),
        "total_hours": round((gen_time + force_time) / 3600, 2),
        "plain_preds": plain, "forced_preds": forced,
    }
    with open("outputs/phase6_forcing.json", "w") as f:
        json.dump({"results": results, "examples": examples,
                   "truths": truths}, f, indent=2)

print("\nsample forced continuations - check these are grounded, not guesses")
for e in examples:
    print(f"  problem {e['problem']} truth={e['truth']} parsed={e['parsed']}"
          f"  {'HIT' if e['parsed'] == e['truth'] else 'miss'}")
    print(f"    {e['continuation'][:150]}")

with open("outputs/phase6_forcing.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["budget", "method", "pass_at_1", "truncated", "gpu_hours"])
    for budget, r in results.items():
        w.writerow([budget, "no_forcing", r["no_forcing_pct"], r["truncated"],
                    r["gen_hours"]])
        w.writerow([budget, "forcing", r["forcing_pct"], r["truncated"],
                    r["total_hours"]])
print("saved outputs/phase6_forcing.json and .csv")
