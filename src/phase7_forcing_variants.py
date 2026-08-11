"""Phase 7 - generate 16k traces once, evaluate three forcing variants at two budgets.

Key efficiency trick: generation is autoregressive, so the first 8,192 tokens of a
16k run are exactly what an 8k run would have produced. One generation therefore
yields both budgets. Validated: the derived 8k condition reproduced a genuine 8k
run (55.8% vs 54.2%, within noise).

Saves every trace so any budget <= 16k can be re-scored later with no GPU.

Cost: ~8.9 GPU-hours on a Tesla T4 (3.6 generation + 5.3 forcing prefill).
Output: outputs/phase7_traces.json, outputs/phase7_results.json / .csv
"""
import csv
import json
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import (MODEL, DATASET, TEMPERATURE, TOP_P, K, build_prompt,
                    extract_boxed, to_int, prefix_at, outdir,
                    FORCE_BARE, FORCE_DEADLINE, FORCE_SUMMARIZE,
                    FORCE_SUMMARIZE_ANSWER)

GEN_BUDGET = 16384
BUDGETS = (8192, 16384)

outdir()
ds = load_dataset(DATASET)["test"]
tokenizer = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.90,
          max_model_len=18432, trust_remote_code=True)

prompts = [build_prompt(tokenizer, ds[i]["problem"]) for i in range(30)]
truths = [to_int(str(ds[i]["answer"])) for i in range(30)]

# ---------------------------------------------------------------- generation
print(f"generating 30 x K={K} at {GEN_BUDGET:,} tokens")
t0 = time.time()
outs = llm.generate(prompts, SamplingParams(
    n=K, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=GEN_BUDGET))
gen_time = time.time() - t0
gen_tokens = sum(len(c.token_ids) for o in outs for c in o.outputs)
print(f"  {gen_time / 60:.1f} min, {gen_tokens:,} tokens, "
      f"{gen_tokens / gen_time:.0f} tok/s")

traces = [[{"ids": list(c.token_ids), "text": c.text, "finish": c.finish_reason}
           for c in o.outputs] for o in outs]
with open("outputs/phase7_traces.json", "w") as f:
    json.dump({"truths": truths, "traces": traces}, f)
print("  traces saved (any budget <= 16k can now be scored with no GPU)")

GREEDY = SamplingParams(n=1, temperature=0.0, max_tokens=24)
SUMMARY = SamplingParams(n=1, temperature=0.0, max_tokens=320)
results, force_cost, examples = {}, {}, []

for budget in BUDGETS:
    print(f"\n=== budget {budget:,} ===")
    base, jobs = [], []
    for pi in range(30):
        row = []
        for ci in range(K):
            text, truncated = prefix_at(traces[pi][ci], budget, tokenizer)
            row.append(to_int(extract_boxed(text)))
            if truncated:
                jobs.append((pi, ci, prompts[pi] + text))
        base.append(row)

    acc0 = sum(1 for pi in range(30) for p in base[pi] if p == truths[pi])
    print(f"  truncated {len(jobs)}/{30 * K}   no forcing {100 * acc0 / 120:.1f}%")
    results[f"{budget}_none"] = round(100 * acc0 / 120, 1)

    # single-pass variants
    for tag, phrase in (("V1_bare", FORCE_BARE), ("V3_deadline", FORCE_DEADLINE)):
        t0 = time.time()
        fo = llm.generate([j[2] + phrase for j in jobs], GREEDY)
        cost = time.time() - t0
        preds = [r[:] for r in base]
        for (pi, ci, _), o in zip(jobs, fo):
            preds[pi][ci] = to_int(o.outputs[0].text)
        acc = sum(1 for pi in range(30) for p in preds[pi] if p == truths[pi])
        results[f"{budget}_{tag}"] = round(100 * acc / 120, 1)
        force_cost[f"{budget}_{tag}"] = round(cost / 60, 1)
        print(f"  [{tag:<12}] {100 * acc / 120:5.1f}%  "
              f"({100 * (acc - acc0) / 120:+.1f})  {cost / 60:.1f} min")

    # two-pass: summarize, then answer
    t0 = time.time()
    sums = llm.generate([j[2] + FORCE_SUMMARIZE for j in jobs], SUMMARY)
    fo = llm.generate(
        [j[2] + FORCE_SUMMARIZE + s.outputs[0].text + FORCE_SUMMARIZE_ANSWER
         for j, s in zip(jobs, sums)], GREEDY)
    cost = time.time() - t0
    preds = [r[:] for r in base]
    for (pi, ci, _), s, o in zip(jobs, sums, fo):
        preds[pi][ci] = to_int(o.outputs[0].text)
        if len(examples) < 6 and pi in (1, 13, 27):
            examples.append({"problem": pi, "truth": truths[pi],
                             "summary": s.outputs[0].text[:400],
                             "answer": o.outputs[0].text[:40],
                             "parsed": preds[pi][ci]})
    acc = sum(1 for pi in range(30) for p in preds[pi] if p == truths[pi])
    results[f"{budget}_V2_summarize"] = round(100 * acc / 120, 1)
    force_cost[f"{budget}_V2_summarize"] = round(cost / 60, 1)
    print(f"  [{'V2_summarize':<12}] {100 * acc / 120:5.1f}%  "
          f"({100 * (acc - acc0) / 120:+.1f})  {cost / 60:.1f} min")

    with open("outputs/phase7_results.json", "w") as f:
        json.dump({"results": results, "force_minutes": force_cost,
                   "gen_minutes": round(gen_time / 60, 1),
                   "v2_examples": examples}, f, indent=2)

print("\nfinal")
for budget in BUDGETS:
    b0 = results[f"{budget}_none"]
    for tag in ("none", "V1_bare", "V3_deadline", "V2_summarize"):
        v = results[f"{budget}_{tag}"]
        delta = "" if tag == "none" else f"{v - b0:+.1f}"
        print(f"  {budget // 1024:>3}k  {tag:<14} {v:5.1f}%  {delta}")

with open("outputs/phase7_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["budget", "method", "pass_at_1", "force_minutes"])
    for budget in BUDGETS:
        for tag in ("none", "V1_bare", "V3_deadline", "V2_summarize"):
            w.writerow([budget, tag, results[f"{budget}_{tag}"],
                        force_cost.get(f"{budget}_{tag}", 0)])
print("saved outputs/phase7_results.json and .csv")
