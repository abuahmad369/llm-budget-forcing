"""Phase 8 - analysis and figures from saved traces. No GPU.

Because generation is autoregressive, any budget <= the generation cap can be
scored by truncating the saved traces. This produces the fine-grained curve at
zero compute cost.

Cost: CPU only, ~1 minute.
Output: outputs/fig3..fig5 .png, outputs/phase8_analysis.json
"""
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from common import MODEL, K, extract_boxed, to_int, prefix_at, load_traces, outdir

# Measured in phases 5-7.
BASELINE_32K = {"budget": 32768, "acc": 80.8, "hours": 8.87}
FORCED = {4096: 40.8, 8192: 56.7, 16384: 70.8}
GAINS = {4096: (27.5, 40.8, 104), 8192: (46.7, 56.7, 81), 16384: (63.3, 70.8, 58)}
PAPER_AIME25 = 91.4

outdir()
truths, traces = load_traces()
tokenizer = AutoTokenizer.from_pretrained(MODEL)

GRID = [1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
curve, trunc_curve, flat = [], [], []
for budget in GRID:
    ok = tr = 0
    for pi in range(30):
        for ci in range(K):
            text, truncated = prefix_at(traces[pi][ci], budget, tokenizer)
            tr += truncated
            ok += (to_int(extract_boxed(text)) == truths[pi])
    curve.append(100 * ok / 120)
    trunc_curve.append(100 * tr / 120)
    flat.append({"budget": budget, "acc": round(100 * ok / 120, 1),
                 "trunc_pct": round(100 * tr / 120, 1)})
    print(f"  {budget:>6,} -> {100 * ok / 120:5.1f}%   truncated {100 * tr / 120:5.1f}%")


def bootstrap_ci(gain_points, n=120, iters=4000):
    """Approximate CI for the gain. NOTE: unpaired; McNemar (phase 9) supersedes."""
    random.seed(0)
    k = round(gain_points / 100 * n)
    vals = [1] * k + [0] * (n - k)
    means = sorted(100 * sum(random.choice(vals) for _ in range(n)) / n
                   for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


print("\nforcing gain, 95% bootstrap CI (unpaired approximation)")
ci_rows = []
for budget, (base, forced, n_tr) in GAINS.items():
    lo, hi = bootstrap_ci(forced - base)
    print(f"  {budget // 1024:>3}k  {base:5.1f}% -> {forced:5.1f}%  "
          f"gain {forced - base:+5.1f} [{lo:+.1f},{hi:+.1f}]  "
          f"significant={'yes' if lo > 0 else 'NO'}")
    ci_rows.append({"budget": budget, "base": base, "forced": forced,
                    "gain": round(forced - base, 1),
                    "ci_lo": round(lo, 1), "ci_hi": round(hi, 1)})

# --- figure 3: main result --------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
fb = sorted(FORCED)
ax.plot(GRID, curve, "-", color="#B23A3A", lw=2, label="No forcing (fine grid)")
ax.plot(fb, [FORCED[b] for b in fb], "s-", color="#1F6E43", lw=2, ms=9,
        label="Best forcing")
ax.plot([BASELINE_32K["budget"]], [BASELINE_32K["acc"]], "*", color="#2F4B7C",
        ms=18, label="32k unforced (baseline)")
ax.axhline(PAPER_AIME25, ls=":", color="gray", lw=1.4)
ax.text(1100, PAPER_AIME25 + 1.0, f"paper: {PAPER_AIME25}%", fontsize=8, color="gray")
for b in fb:
    ax.annotate(f"{FORCED[b]:.1f}", (b, FORCED[b]), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color="#1F6E43")
ax.set_xscale("log", base=2)
ax.set_xticks(fb + [BASELINE_32K["budget"]])
ax.set_xticklabels([f"{b // 1024}k" for b in fb + [BASELINE_32K["budget"]]])
ax.set_xlabel("Token budget per sample")
ax.set_ylabel("AIME25 Pass@1 (%)")
ax.set_title("Reasoning budget, not reasoning ability, bounds accuracy")
ax.set_ylim(15, 100)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("outputs/fig3_main_result.png", bbox_inches="tight")

# --- figure 4: truncation vs accuracy ---------------------------------------
fig, ax1 = plt.subplots(figsize=(7.2, 4.6), dpi=150)
ax1.plot(GRID, trunc_curve, "-", color="#C46A1F", lw=2)
ax1.set_xscale("log", base=2)
ax1.set_xticks(GRID[::2])
ax1.set_xticklabels([f"{g // 1024}k" for g in GRID[::2]])
ax1.set_xlabel("Token budget")
ax1.set_ylabel("% samples truncated", color="#C46A1F")
ax1.tick_params(axis="y", labelcolor="#C46A1F")
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(GRID, curve, "-", color="#B23A3A", lw=2)
ax2.set_ylabel("Pass@1 (%)", color="#B23A3A")
ax2.tick_params(axis="y", labelcolor="#B23A3A")
ax1.set_title("Truncation rate and accuracy move inversely with budget")
plt.tight_layout()
plt.savefig("outputs/fig4_truncation.png", bbox_inches="tight")

# --- figure 5: length distribution ------------------------------------------
lengths = [len(traces[p][c]["ids"]) for p in range(30) for c in range(K)]
cap = max(lengths)
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
ax.hist(lengths, bins=30, color="#2F4B7C", edgecolor="white")
ax.axvline(cap, color="#B23A3A", ls="--", lw=2)
ax.text(cap * 0.93, ax.get_ylim()[1] * 0.85, f"{cap // 1024}k cap",
        rotation=90, fontsize=8, color="#B23A3A")
ax.set_xlabel("Trace length (tokens)")
ax.set_ylabel("Number of samples")
ax.set_title(f"Reasoning-length distribution (right-censored at {cap:,} tokens)")
plt.tight_layout()
plt.savefig("outputs/fig5_lengths.png", bbox_inches="tight")

with open("outputs/phase8_analysis.json", "w") as f:
    json.dump({"fine_curve": flat, "forcing_ci": ci_rows,
               "lengths": {"median": sorted(lengths)[len(lengths) // 2],
                           "at_cap": sum(1 for x in lengths if x >= cap)}},
              f, indent=2)
print("\nsaved fig3, fig4, fig5 and outputs/phase8_analysis.json")
print("NOTE: fig5 is right-censored - the spike at the cap is the limit, not")
print("      the model's natural stopping point. Say so in the caption.")
