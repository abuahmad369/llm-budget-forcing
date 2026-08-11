"""Phase 9 - paired analysis. Does budget forcing ever DESTROY a correct answer?

A truncated trace can already contain a correct \\boxed{} written mid-reasoning.
Forcing overwrites it. Net gain alone cannot distinguish "+12 rescued" from
"+16 rescued, -4 destroyed", so this measures the contingency table directly
and applies McNemar's test - the correct paired test for this design.

Also evaluates a hybrid rule: keep an existing boxed answer, force only when
none exists.

Cost: ~1.3 GPU-hours on a Tesla T4 (all prefill).
Output: outputs/phase9_paired.json
"""
import json
import math
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import (MODEL, DATASET, K, build_prompt, extract_boxed, to_int,
                    prefix_at, load_traces, outdir, FORCE_BARE)

outdir()
truths, traces = load_traces()
ds = load_dataset(DATASET)["test"]
tokenizer = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="float16", gpu_memory_utilization=0.90,
          max_model_len=18432, trust_remote_code=True)

GREEDY = SamplingParams(n=1, temperature=0.0, max_tokens=24)
report = {}

for budget in (8192, 16384):
    print(f"\n=== budget {budget:,} ===")
    jobs, base = [], []
    for pi in range(30):
        row = []
        for ci in range(K):
            text, truncated = prefix_at(traces[pi][ci], budget, tokenizer)
            pred = to_int(extract_boxed(text))
            row.append({"pred": pred, "had_boxed": pred is not None})
            if truncated:
                jobs.append((pi, ci,
                             build_prompt(tokenizer, ds[pi]["problem"]) + text))
        base.append(row)

    t0 = time.time()
    fo = llm.generate([j[2] + FORCE_BARE for j in jobs], GREEDY)
    print(f"  forcing took {(time.time() - t0) / 60:.1f} min")

    b = c = same_right = same_wrong = 0
    forced_map, records = {}, []
    for (pi, ci, _), o in zip(jobs, fo):
        old = base[pi][ci]["pred"]
        new = to_int(o.outputs[0].text)
        forced_map[(pi, ci)] = new
        old_ok, new_ok = (old == truths[pi]), (new == truths[pi])
        if not old_ok and new_ok:
            b += 1
        elif old_ok and not new_ok:
            c += 1
        elif old_ok:
            same_right += 1
        else:
            same_wrong += 1
        records.append({"problem": pi, "sample": ci, "truth": truths[pi],
                        "had_boxed": base[pi][ci]["had_boxed"],
                        "before": old, "after": new,
                        "before_ok": old_ok, "after_ok": new_ok})

    discordant = b + c
    chi2 = ((abs(b - c) - 1) ** 2) / discordant if discordant else 0.0
    p = math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0

    acc_none = sum(1 for pi in range(30) for ci in range(K)
                   if base[pi][ci]["pred"] == truths[pi])
    acc_force = sum(1 for pi in range(30) for ci in range(K)
                    if forced_map.get((pi, ci), base[pi][ci]["pred"]) == truths[pi])
    acc_hybrid = 0
    for pi in range(30):
        for ci in range(K):
            cell = base[pi][ci]
            v = cell["pred"] if cell["had_boxed"] else forced_map.get((pi, ci))
            acc_hybrid += (v == truths[pi])

    print(f"  truncated {len(jobs)}, of which {sum(r['had_boxed'] for r in records)} "
          f"already had a boxed answer")
    print(f"  wrong -> RIGHT (rescued) b = {b}")
    print(f"  right -> WRONG (damaged) c = {c}")
    print(f"  McNemar chi2 = {chi2:.2f}  p = {p:.4f}")
    print(f"  none {100 * acc_none / 120:.1f}%   force {100 * acc_force / 120:.1f}%   "
          f"hybrid {100 * acc_hybrid / 120:.1f}%")

    report[budget] = {
        "n_truncated": len(jobs), "b_rescued": b, "c_damaged": c,
        "mcnemar_chi2": round(chi2, 2), "p_value": round(p, 4),
        "acc_none": round(100 * acc_none / 120, 1),
        "acc_force": round(100 * acc_force / 120, 1),
        "acc_hybrid": round(100 * acc_hybrid / 120, 1),
        "per_sample": records,
    }
    with open("outputs/phase9_paired.json", "w") as f:
        json.dump(report, f, indent=2)

print("\nIf c == 0 at every budget, forcing is strictly non-destructive and the")
print("hybrid rule is unnecessary.")
print("saved outputs/phase9_paired.json")
