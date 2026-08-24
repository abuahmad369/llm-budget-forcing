"""One figure that carries the whole result, for a slide or the report.

Left  : accuracy against budget, with truncation mirrored on the right axis.
Right : what the 120 samples are actually made of, before and after forcing.

The right panel is the argument. Without forcing the bars hold no red at all -
the model states the correct answer or states nothing. Forcing converts the
amber block into green and red, and the green never shrinks.

Forcing is plotted at 8k and 16k only. A 4k point exists but mixes arms: its
baseline was measured on a dedicated 4k run while the curve here is derived from
16k traces, so the two are not the same experiment. It stays in the table with
that caveat rather than being drawn as if it were comparable.
"""
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OK, BAD, CUT, ACC = "#1D7A4D", "#B3341F", "#C89B3C", "#0E6E80"
GREY = "#5C6675"
PAPER = 91.4

d = json.load(io.open(os.path.join(HERE, "data.json"), encoding="utf-8"))
B = d["meta"]["budgets"]


def compose(budget, forced):
    ok = wrong = none = 0
    for p in d["problems"]:
        for s in p["samples"]:
            e = s["b"][str(budget)]
            a = e.get("fa") if forced else e["a"]
            k = e.get("fok") if forced else e["ok"]
            if a is None:
                none += 1
            elif k:
                ok += 1
            else:
                wrong += 1
    return ok, wrong, none


acc = [c["acc"] for c in d["curve"]]
trunc = [c["trunc"] for c in d["curve"]]
FB, FBASE, FACC = [8192, 16384], [46.7, 63.3], [55.8, 70.8]

plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": "#4A5462",
                     "axes.labelcolor": "#25303C", "text.color": "#25303C"})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.7), dpi=200,
                               gridspec_kw={"width_ratios": [1.1, 0.9]})

# ---------------------------------------------------------------- left panel
ax2 = axL.twinx()
ax2.plot(B, trunc, "--", color=CUT, lw=1.7, alpha=.85, zorder=1)
ax2.set_ylabel("% of samples cut off before answering", color=CUT, fontsize=9)
ax2.tick_params(axis="y", labelcolor=CUT, labelsize=8.5)
ax2.set_ylim(0, 108)
ax2.text(18500, 44, "share cut off\n(right axis)", color=CUT,
         fontsize=8.4, style="italic", ha="center", linespacing=1.3)

axL.plot(B, acc, "o-", color=BAD, lw=2.3, ms=5, label="No forcing", zorder=3)
axL.plot(FB, FACC, "s-", color=OK, lw=2.3, ms=8,
         label="With budget forcing", zorder=4)
axL.plot([32768], [80.8], "*", color=ACC, ms=17, zorder=5,
         label="32k, no forcing")
axL.axhline(PAPER, ls=":", color=GREY, lw=1.3, zorder=2)
axL.text(1080, PAPER + 2.4, "paper reports 91.4 (no budget cap)",
         fontsize=8.4, color=GREY)

for x, y0, y1 in zip(FB, FBASE, FACC):
    axL.annotate("", xy=(x, y1 - 1.2), xytext=(x, y0 + 1.2),
                 arrowprops=dict(arrowstyle="-|>", color=OK, lw=1.7,
                                 shrinkA=0, shrinkB=0))
    axL.text(x * 0.80, (y0 + y1) / 2 - 1.8, "+%.1f" % (y1 - y0),
             fontsize=9.5, color=OK, fontweight="bold", ha="right")

axL.set_xscale("log", base=2)
axL.set_xticks([1024, 2048, 4096, 8192, 16384, 32768])
axL.set_xticklabels(["1k", "2k", "4k", "8k", "16k", "32k"])
axL.set_xlabel("Token budget per sample")
axL.set_ylabel("AIME 2025 Pass@1 (%)")
axL.set_ylim(0, 108)
axL.grid(alpha=.2, zorder=0)
axL.set_axisbelow(True)
axL.legend(loc="lower right", fontsize=8.6, framealpha=.96)
axL.set_title("Accuracy is set by the budget, not by reasoning",
              fontsize=11.5, fontweight="bold", pad=10)

# --------------------------------------------------------------- right panel
labels, groups = [], []
for b in (8192, 16384):
    groups.append(compose(b, False))
    labels.append("8k" if b == 8192 else "16k")
    groups.append(compose(b, True))
    labels.append("forced")

xs, w = [0, 0.92, 2.5, 3.42], 0.8
bot = [0, 0, 0, 0]
for vals, color, name in (([g[0] for g in groups], OK, "Correct"),
                          ([g[1] for g in groups], BAD, "Wrong answer"),
                          ([g[2] for g in groups], CUT, "Cut off, no answer")):
    axR.bar(xs, vals, w, bottom=bot, color=color, edgecolor="white", lw=.9)
    for x, v, bo in zip(xs, vals, bot):
        if v >= 8:
            axR.text(x, bo + v / 2, str(v), ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold")
    bot = [a + b_ for a, b_ in zip(bot, vals)]

for (p_, r_), txt in zip(((0, 1), (2, 3)),
                         ("+11 rescued\n0 destroyed", "+9 rescued\n0 destroyed")):
    axR.annotate("", xy=(xs[r_] - w / 2 - .03, 124),
                 xytext=(xs[p_] + w / 2 + .03, 124),
                 arrowprops=dict(arrowstyle="-|>", color=OK, lw=1.6))
    axR.text((xs[p_] + xs[r_]) / 2, 129, txt, ha="center", va="bottom",
             fontsize=8.8, color=OK, fontweight="bold", linespacing=1.35)

axR.set_xticks(xs)
axR.set_xticklabels(labels, fontsize=9.5)
axR.set_ylabel("Samples (of 120)")
axR.set_ylim(0, 152)
axR.set_yticks([0, 30, 60, 90, 120])
axR.grid(axis="y", alpha=.2)
axR.set_axisbelow(True)
axR.legend(handles=[Patch(facecolor=OK, label="Correct"),
                    Patch(facecolor=BAD, label="Wrong answer"),
                    Patch(facecolor=CUT, label="Cut off, no answer")],
           loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
           fontsize=8.8, frameon=False)
axR.set_title("Left alone it never guesses; forcing makes it commit",
              fontsize=11.5, fontweight="bold", pad=10)

fig.tight_layout(pad=1.4)
out = os.path.join(HERE, "..", "figures", "fig6_summary.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", os.path.abspath(out))
for lab, g in zip(labels, groups):
    print("  %-8s correct %3d  wrong %3d  no answer %3d" % (lab, g[0], g[1], g[2]))
