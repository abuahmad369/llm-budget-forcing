"""Build a compact JSON payload for the results explorer.

Everything here comes from measured artifacts:
  phase7_traces.json  - 120 real reasoning traces at 16k
  phase9_paired.json  - per-sample forced answers at 8k and 16k
  HF datasets-server  - the 30 AIME 2025 problem statements

Nothing is simulated. Budgets where forcing was not measured are marked as such.
"""
import json, os, re
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

BUDGETS = [1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14336, 16384]
FORCED_FULL = {8192, 16384}          # per-sample forced data exists
FORCED_AGG = {4096: 40.8, 8192: 55.8, 16384: 70.8}   # measured aggregates
TAIL_CHARS = 340
HEAD_CHARS = 420

tok = Tokenizer.from_file(os.path.join(HERE, "_tok.json"))

traces_blob = json.load(open(os.path.join(RES, "phase7_traces.json")))
truths, traces = traces_blob["truths"], traces_blob["traces"]
paired = json.load(open(os.path.join(RES, "phase9_paired.json")))
aime = json.load(open(os.path.join(HERE, "_aime_raw.json")))["rows"]
problems = [r["row"]["problem"] for r in aime]
assert len(problems) == 30, len(problems)

# forced[(budget, pi, ci)] -> answer
forced = {}
for bkey, blk in paired.items():
    for rec in blk["per_sample"]:
        forced[(int(bkey), rec["problem"], rec["sample"])] = rec["after"]


BOX = chr(92) + chr(98) + chr(111) + chr(120) + chr(101) + chr(100) + chr(123)

def boxed(t):
    i = t.rfind(BOX)
    if i == -1:
        return None
    j, d, out = i + 7, 1, []
    while j < len(t) and d > 0:
        ch = t[j]
        if ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
            if d == 0:
                break
        out.append(ch)
        j += 1
    return "".join(out).strip()


def as_int(s):
    if s is None:
        return None
    m = re.search(r"-?\d+", s.replace(",", ""))
    return int(m.group()) if m else None


payload_problems = []
for pi in range(30):
    truth = truths[pi]
    samples = []
    for ci in range(4):
        tr = traces[pi][ci]
        ids, full_text = tr["ids"], tr["text"]
        n_tok = len(ids)
        finished_naturally = tr["finish"] == "stop"

        per_budget = {}
        for B in BUDGETS:
            if n_tok <= B:
                text, truncated = full_text, (not finished_naturally)
            else:
                text, truncated = tok.decode(ids[:B]), True
            ans = as_int(boxed(text))
            entry = {
                "t": 1 if truncated else 0,
                "a": ans,
                "ok": 1 if ans == truth else 0,
                "tail": text[-TAIL_CHARS:],
                "len": min(n_tok, B),
            }
            if B in FORCED_FULL:
                fa = forced.get((B, pi, ci))
                if truncated:
                    entry["fa"] = fa
                    entry["fok"] = 1 if fa == truth else 0
                else:
                    entry["fa"] = ans          # untouched when it terminated
                    entry["fok"] = entry["ok"]
            per_budget[str(B)] = entry

        samples.append({
            "n_tok": n_tok,
            "fin": 1 if finished_naturally else 0,
            "head": full_text[:HEAD_CHARS],
            "b": per_budget,
        })

    payload_problems.append({
        "id": pi,
        "text": problems[pi],
        "truth": truth,
        "samples": samples,
    })

# ---- aggregate curve -------------------------------------------------------
curve = []
for B in BUDGETS:
    ok = tr_ct = 0
    fok = None
    for p in payload_problems:
        for s in p["samples"]:
            e = s["b"][str(B)]
            ok += e["ok"]
            tr_ct += e["t"]
    if B in FORCED_FULL:
        fok = sum(s["b"][str(B)]["fok"] for p in payload_problems for s in p["samples"])
    curve.append({
        "budget": B,
        "acc": round(100 * ok / 120, 1),
        "trunc": round(100 * tr_ct / 120, 1),
        "facc": round(100 * fok / 120, 1) if fok is not None else FORCED_AGG.get(B),
        "measured_per_sample": B in FORCED_FULL,
    })

out = {
    "meta": {
        "model": "VibeThinker-3B",
        "dataset": "AIME 2025 (math-ai/aime25)",
        "n_problems": 30,
        "K": 4,
        "budgets": BUDGETS,
        "forced_per_sample_budgets": sorted(FORCED_FULL),
        "paper_score": 91.4,
        "baseline_32k": 80.8,
        "gpu_hours": 22.6,
        "note": "All values measured on a Kaggle Tesla T4. No simulation.",
    },
    "curve": curve,
    "problems": payload_problems,
    "stats": {
        "8192": {k: paired["8192"][k] for k in
                 ("n_truncated", "b_rescued", "c_damaged", "mcnemar_chi2",
                  "p_value", "acc_none", "acc_force")},
        "16384": {k: paired["16384"][k] for k in
                  ("n_truncated", "b_rescued", "c_damaged", "mcnemar_chi2",
                   "p_value", "acc_none", "acc_force")},
    },
}

path = os.path.join(HERE, "data.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
print("wrote", path, round(os.path.getsize(path) / 1e6, 2), "MB")
print("curve:")
for c in curve:
    print(f"  {c['budget']:>6} acc={c['acc']:>5}% trunc={c['trunc']:>5}% "
          f"forced={c['facc']} per_sample={c['measured_per_sample']}")
