"""Phase 11b - corrected extraction ceiling, restricted to rescuable samples.

Supersedes the ceiling reported by phase11_upper_bounds.py, which was wrong.

The bug: that script divided rescued-from-wrong by `ceil_box` (truncated traces
where the answer appears in ANY \\boxed{}). `ceil_box` turned out to equal the
ALREADY-CORRECT set exactly - 14 traces at 16k, matching right->right = 14 from
phase 9 - because no trace has the answer in an early box but not the last one.
Numerator and denominator were disjoint sets, so the ratio was meaningless.

This version restricts to truncated AND wrong samples, which is the only
population budget forcing can act on, and subtracts a false-positive control.

Cost: CPU only, ~2 minutes.
Output: outputs/phase11b_ceiling.json
"""
import json
import re

from transformers import AutoTokenizer

from common import (MODEL, K, extract_all_boxed, to_int, prefix_at,
                    load_traces, outdir)

# Rescued counts: phase 9 measured b directly at 8k and 16k.
# 4k is excluded - its forced number came from a genuine 4k run (baseline 27.5%)
# while the ceiling here is trace-derived (baseline 30.0%), so the arms differ.
RESCUED = {8192: 11, 16384: 9}

outdir()
truths, traces = load_traces()
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print(f"{'budget':>7}{'trunc':>8}{'already_ok':>12}{'RESCUABLE':>11}"
      f"{'has_ans':>9}{'FP':>6}{'net_ceil':>10}{'rescued':>9}{'captured':>10}")
print("-" * 84)

rows = []
for budget in sorted(RESCUED):
    n_trunc = n_already_ok = n_rescuable = has_ans = false_pos = 0
    for pi in range(30):
        truth = truths[pi]
        decoy = truths[(pi + 7) % 30]        # false-positive calibration
        for ci in range(K):
            text, truncated = prefix_at(traces[pi][ci], budget, tokenizer)
            if not truncated:
                continue
            n_trunc += 1
            boxes = extract_all_boxed(text)
            if (to_int(boxes[-1]) if boxes else None) == truth:
                n_already_ok += 1            # already correct -> not rescuable
                continue
            n_rescuable += 1
            tail = text[int(len(text) * 0.8):]
            has_ans += bool(re.search(rf"(?<!\d){truth}(?!\d)", tail))
            false_pos += bool(re.search(rf"(?<!\d){decoy}(?!\d)", tail))

    net_ceiling = max(has_ans - false_pos, 0)
    rescued = RESCUED[budget]
    captured = 100 * rescued / net_ceiling if net_ceiling else float("nan")

    print(f"{budget // 1024:>6}k{n_trunc:>8}{n_already_ok:>12}{n_rescuable:>11}"
          f"{has_ans:>9}{false_pos:>6}{net_ceiling:>10}{rescued:>9}{captured:>9.1f}%")
    rows.append({"budget": budget, "truncated": n_trunc,
                 "already_correct": n_already_ok, "rescuable": n_rescuable,
                 "answer_in_tail": has_ans, "false_positives": false_pos,
                 "net_ceiling": net_ceiling, "rescued": rescued,
                 "captured_pct": round(captured, 1)})

print("-" * 84)
print("RESCUABLE = truncated AND wrong; the only pool forcing can act on")
print("net_ceil  = has_ans - FP; wrong traces that still mention the answer")
print("captured  = rescued / net_ceil")
print()
print("The ceiling population is small (n=11-14), so Wilson 95% intervals are")
print("wide: roughly [52%, 92%] at 8k and [52%, 95%] at 16k. The lower bound is")
print("still well above the 40% level at which better extraction would pay off.")
print()
print("Note that only ~25% of rescuable traces mention the answer at all - the")
print("other ~75% never derived it. The bottleneck is generating the reasoning,")
print("not reading it out.")

with open("outputs/phase11b_ceiling.json", "w") as f:
    json.dump({"corrected_ceiling": rows}, f, indent=2)
print("\nsaved outputs/phase11b_ceiling.json")
