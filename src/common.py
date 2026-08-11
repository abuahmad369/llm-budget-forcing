"""Shared helpers for every phase.

Answer extraction and prompt construction MUST be identical across all
experimental arms - see the controls audit in PROGRESS.md. Import from here
rather than redefining, so a change cannot silently apply to only one arm.
"""
import os
import re
import json

MODEL = "WeiboAI/VibeThinker-3B"
DATASET = "math-ai/aime25"

# Frozen decoding protocol. Do not change these per-arm.
TEMPERATURE = 1.0
TOP_P = 0.95
SEED = 0
K = 4

# Forcing phrases evaluated in phase 7.
FORCE_BARE = ("\n\n</think>\n\nI have reasoned enough. Based on the work above, "
              "the final answer is \\boxed{")
FORCE_DEADLINE = ("\n\n[Note: you have very little space left. Stop exploring and "
                  "commit to your best answer now.]\n\n</think>\n\n"
                  "The final answer is \\boxed{")
FORCE_SUMMARIZE = ("\n\n</think>\n\nI am out of reasoning time. Let me summarize "
                   "exactly what I have established so far.\n\n"
                   "Summary of established facts:")
FORCE_SUMMARIZE_ANSWER = "\n\nTherefore, the final answer is \\boxed{"


def build_prompt(tokenizer, question):
    """Apply the model's chat template. Identical across all arms."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_boxed(text):
    """Return the LAST \\boxed{...} content, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i, depth, out = idx + 7, 1, []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip()


def extract_all_boxed(text):
    """Return every \\boxed{...} in the trace, in order."""
    out, i = [], 0
    while True:
        i = text.find("\\boxed{", i)
        if i == -1:
            return out
        j, depth, buf = i + 7, 1, []
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            j += 1
        out.append("".join(buf).strip())
        i = j


def to_int(s):
    """AIME answers are integers 0-999."""
    if s is None:
        return None
    m = re.search(r"-?\d+", s.replace(",", ""))
    return int(m.group()) if m else None


def prefix_at(trace, budget, tokenizer):
    """(text, was_truncated) for a saved trace capped at `budget` tokens.

    Valid because generation is autoregressive: the first N tokens of a longer
    run are exactly what a budget-N run would have produced. Validated in
    phase 7 (8k derived this way reproduced a genuine 8k run within noise).
    """
    if len(trace["ids"]) <= budget:
        return trace["text"], (trace["finish"] != "stop")
    return tokenizer.decode(trace["ids"][:budget]), True


def find_traces(name_hints=("trace", "phase7")):
    """Locate the traces file across Kaggle / local layouts.

    os.walk(followlinks=True) rather than glob('**') - glob does not reliably
    follow Kaggle's symlinked dataset mounts.
    """
    hits = []
    for root in ("/kaggle/input", "/kaggle/working", "results", "."):
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            for fn in filenames:
                low = fn.lower()
                if fn.endswith(".json") and any(h in low for h in name_hints):
                    hits.append(os.path.join(dirpath, fn))
    return sorted(set(hits), key=lambda p: -os.path.getsize(p))


def load_traces():
    """Load and validate the saved traces file."""
    cands = find_traces()
    if not cands:
        listing = []
        for root in ("/kaggle/input", "results", "."):
            if os.path.exists(root):
                for dp, _, fs in os.walk(root, followlinks=True):
                    listing += [os.path.join(dp, f) for f in fs][:20]
        raise FileNotFoundError(
            "traces file not found. Searched /kaggle/input, results/, .\n"
            "Found instead:\n  " + "\n  ".join(listing or ["(nothing)"]))
    path = cands[0]
    with open(path) as f:
        d = json.load(f)
    if "traces" not in d:
        raise ValueError(f"{path} has keys {list(d.keys())}, expected 'traces'")
    assert len(d["traces"]) == 30, f"expected 30 problems, got {len(d['traces'])}"
    assert len(d["traces"][0]) == K, f"expected K={K} samples per problem"
    print(f"loaded traces: {path}")
    return d["truths"], d["traces"]


def outdir(name="outputs"):
    os.makedirs(name, exist_ok=True)
    return name
