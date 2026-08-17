"""Phase 11 - upper bounds. What ceiling is budget forcing actually chasing?

WARNING: the `captured` figure printed by this script is WRONG. `ceil_box`
turned out to equal the ALREADY-CORRECT set exactly (14 traces at 16k, matching
right->right = 14 in phase 9), because no trace holds the answer in an early box
but not the last one. So `captured` divided rescued-from-wrong by already-correct
- disjoint sets. Use phase11b_ceiling.py instead.

The pass@4 column IS valid and is the useful output here: it bounds any method
that SELECTS among existing samples, and forced pass@1 exceeds it at 4k and 8k.

Measures, per budget:

  pass@4     oracle over the K samples - bounds any SELECTION method
  ceil_box   truncated traces where the true answer appears in SOME \\boxed{}
  ceil_txt   truncated traces where the true answer appears in the last 20%
  FP_ctrl    the same text test using ANOTHER problem's answer

FP_ctrl matters: AIME answers are small integers, so a bare text match produces
false positives. ceil_txt is only informative to the extent it exceeds FP_ctrl.

Cost: CPU only, ~2 minutes, 0 GPU-hours.
Output: outputs/phase11_upper_bounds.json
"""
import json
import re

from transformers import AutoTokenizer

from common import (MODEL, K, extract_all_boxed, to_int, prefix_at,
                    load_traces, outdir)

# Measured in phases 6 and 7.
FORCED = {4096: 40.8, 8192: 55.8, 16384: 70.8}

outdir()
truths, traces = load_traces()
tokenizer = AutoTokenizer.from_pretrained(MODEL)

rows = []
print()
print(f"{'budget':>7}{'pass@1':>9}{'pass@4':>9}{'ceil_box':>10}"
      f"{'ceil_txt':>10}{'FP_ctrl':>9}{'forced':>9}")
print("-" * 65)

for budget in sorted(FORCED):
    n_correct = n_any = 0
    n_trunc = hit_box = hit_txt = hit_fp = 0

    for pi in range(30):
        truth = truths[pi]
        decoy = truths[(pi + 7) % 30]      # false-positive calibration
        solved_any = False
        for ci in range(K):
            text, truncated = prefix_at(traces[pi][ci], budget, tokenizer)
            all_boxed = extract_all_boxed(text)
            last = to_int(all_boxed[-1]) if all_boxed else None
            ok = (last == truth)
            n_correct += ok
            solved_any |= ok
            if truncated:
                n_trunc += 1
                hit_box += any(to_int(x) == truth for x in all_boxed)
                tail = text[int(len(text) * 0.8):]
                hit_txt += bool(re.search(rf"(?<!\d){truth}(?!\d)", tail))
                hit_fp += bool(re.search(rf"(?<!\d){decoy}(?!\d)", tail))
        n_any += solved_any

    pass1 = 100 * n_correct / (30 * K)
    pass4 = 100 * n_any / 30
    cb = 100 * hit_box / n_trunc if n_trunc else 0.0
    ct = 100 * hit_txt / n_trunc if n_trunc else 0.0
    fp = 100 * hit_fp / n_trunc if n_trunc else 0.0

    print(f"{budget // 1024:>6}k{pass1:>8.1f}%{pass4:>8.1f}%{cb:>9.1f}%"
          f"{ct:>9.1f}%{fp:>8.1f}%{FORCED[budget]:>8.1f}%")
    rows.append({"budget": budget, "pass_at_1": round(pass1, 1),
                 "pass_at_4": round(pass4, 1), "ceiling_boxed_pct": round(cb, 1),
                 "ceiling_text_pct": round(ct, 1),
                 "false_positive_ctrl_pct": round(fp, 1),
                 "n_truncated": n_trunc, "forced_pass_at_1": FORCED[budget]})

print()
print("headroom - how much of the extraction ceiling does forcing capture?")
for r in rows:
    n_tr = r["n_truncated"]
    rescued = round((r["forced_pass_at_1"] - r["pass_at_1"]) / 100 * 120)
    ceiling = round(r["ceiling_boxed_pct"] / 100 * n_tr)
    frac = 100 * rescued / ceiling if ceiling else float("nan")
    print(f"  {r['budget'] // 1024:>3}k  truncated={n_tr:>3}  "
          f"answer-in-trace={ceiling:>3}  rescued={rescued:>3}  "
          f"captured={frac:5.1f}%")

print()
print("captured > 70%  -> forcing is near-optimal, method is done")
print("captured < 40%  -> a better extractor has real headroom, report as future work")

with open("outputs/phase11_upper_bounds.json", "w") as f:
    json.dump({"upper_bounds": rows,
               "note": "CPU-derived from saved traces; no GPU used"}, f, indent=2)
print("saved outputs/phase11_upper_bounds.json")
