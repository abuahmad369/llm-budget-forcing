"""Live budget-forcing demo for VibeThinker-3B.

Runs the model for real, on a problem you choose, and shows what the answer is
with and without budget forcing.

Two backends:

  vllm    float16 on a Kaggle T4. Identical configuration to every number in the
          report, so results are directly comparable. THIS IS THE ONE TO DEMO.

  hf4bit  4-bit NF4 on a local card with too little memory for float16 (a GTX
          1650 has 4 GB; the weights alone need 6.2 GB). Quantization changes the
          model's outputs, so numbers from this backend are an approximation and
          are labelled as such in the interface.

Launch:
    python live_demo.py                # auto-detect backend
    python live_demo.py --backend vllm --share
    python live_demo.py --backend hf4bit

The forcing phrase and the answer parser below are copied verbatim from
src/common.py. If you change one, change both, or the demo stops measuring the
same thing the report measured.
"""
import argparse
import gc
import json
import re
import time
import urllib.request

import gradio as gr
import torch

MODEL = "WeiboAI/VibeThinker-3B"
AIME_URL = ("https://datasets-server.huggingface.co/rows"
            "?dataset=math-ai%2Faime25&config=default&split=test"
            "&offset=0&length=30")

# --- must match src/common.py -------------------------------------------------
TEMPERATURE = 1.0
TOP_P = 0.95
BOX = chr(92) + "boxed{"
FORCE_BARE = ("\n\n</think>\n\nI have reasoned enough. Based on the work above, "
              "the final answer is " + BOX)

# Measured on a T4, float16, for the comparison table in the Reproduce tab.
REPORTED = {4096: (30.0, 40.8), 8192: (46.7, 55.8), 16384: (63.3, 70.8)}
PAPER_AIME25 = 91.4


# ============================================================== answer parsing
def extract_boxed(text):
    i = text.rfind(BOX)
    if i == -1:
        return None
    j, depth, out = i + len(BOX), 1, []
    while j < len(text) and depth > 0:
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        j += 1
    return "".join(out).strip()


def to_int(s):
    if s is None:
        return None
    m = re.search(r"-?\d+", s.replace(",", ""))
    return int(m.group()) if m else None


# ============================================================== engines
class Gen(object):
    __slots__ = ("text", "n_tokens", "truncated")

    def __init__(self, text, n_tokens, truncated):
        self.text = text
        self.n_tokens = n_tokens
        self.truncated = truncated


class VLLMEngine(object):
    name = "vllm"
    label = "vLLM / float16"
    comparable = True

    def __init__(self, max_len):
        from vllm import LLM
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.llm = LLM(model=MODEL, dtype="float16",
                       gpu_memory_utilization=0.90,
                       max_model_len=max_len, trust_remote_code=True)
        self.max_len = max_len

    def prompt(self, q):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)

    def run(self, prompts, max_tokens, n=1, greedy=False):
        from vllm import SamplingParams
        sp = SamplingParams(
            n=n,
            temperature=0.0 if greedy else TEMPERATURE,
            top_p=1.0 if greedy else TOP_P,
            max_tokens=max_tokens,
        )
        outs = self.llm.generate(prompts, sp)
        return [[Gen(c.text, len(c.token_ids), c.finish_reason != "stop")
                 for c in o.outputs] for o in outs]


class HF4BitEngine(object):
    name = "hf4bit"
    label = "HuggingFace / 4-bit NF4"
    comparable = False

    def __init__(self, max_len):
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                  BitsAndBytesConfig)
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=cfg, device_map="auto",
            trust_remote_code=True)
        self.model.eval()
        self.max_len = max_len

    def prompt(self, q):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)

    def run(self, prompts, max_tokens, n=1, greedy=False):
        results = []
        for p in prompts:
            enc = self.tok(p, return_tensors="pt").to(self.model.device)
            n_in = enc.input_ids.shape[1]
            gens = []
            for _ in range(n):
                with torch.no_grad():
                    out = self.model.generate(
                        **enc,
                        max_new_tokens=max_tokens,
                        do_sample=not greedy,
                        temperature=None if greedy else TEMPERATURE,
                        top_p=None if greedy else TOP_P,
                        pad_token_id=self.tok.eos_token_id,
                    )
                seq = out[0][n_in:]
                txt = self.tok.decode(seq, skip_special_tokens=True)
                gens.append(Gen(txt, int(seq.shape[0]),
                                int(seq.shape[0]) >= max_tokens))
                del out
                gc.collect()
                torch.cuda.empty_cache()
            results.append(gens)
        return results


def build_engine(backend, max_len):
    import importlib.util as iu
    if backend == "auto":
        vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available() else 0)
        has_vllm = iu.find_spec("vllm") is not None
        backend = "vllm" if (has_vllm and vram >= 12) else "hf4bit"
        print("auto-selected backend: %s  (vram %.1f GB, vllm=%s)"
              % (backend, vram, has_vllm))
    return (VLLMEngine if backend == "vllm" else HF4BitEngine)(max_len)


# ============================================================== data
def load_problems():
    try:
        with urllib.request.urlopen(AIME_URL, timeout=45) as r:
            rows = json.loads(r.read().decode("utf-8"))["rows"]
        return [{"idx": i,
                 "text": r["row"]["problem"],
                 "answer": to_int(str(r["row"]["answer"]))}
                for i, r in enumerate(rows)]
    except Exception as e:
        print("could not fetch AIME 2025 (%s); custom problems still work" % e)
        return []


# ============================================================== core measurement
def measure(engine, question, truth, budget, k, progress=None):
    """Generate k samples at `budget`, then force the truncated ones.

    Returns (rows, summary). One generation pass, scored two ways - exactly the
    procedure used in the report.
    """
    p = engine.prompt(question)

    t0 = time.time()
    gens = engine.run([p], max_tokens=budget, n=k)[0]
    gen_s = time.time() - t0
    gen_tokens = sum(g.n_tokens for g in gens)

    # force only the samples that ran out of budget
    idx = [i for i, g in enumerate(gens) if g.truncated]
    forced_text = {}
    force_s = 0.0
    if idx:
        if progress:
            progress(0.75, desc="Forcing %d truncated sample(s)" % len(idx))
        t1 = time.time()
        fprompts = [p + gens[i].text + FORCE_BARE for i in idx]
        fouts = engine.run(fprompts, max_tokens=24, n=1, greedy=True)
        force_s = time.time() - t1
        for i, o in zip(idx, fouts):
            forced_text[i] = o[0].text

    rows, n_plain, n_forced = [], 0, 0
    for i, g in enumerate(gens):
        plain = to_int(extract_boxed(g.text))
        if g.truncated:
            forced = to_int(forced_text.get(i))
        else:
            forced = plain
        p_ok = truth is not None and plain == truth
        f_ok = truth is not None and forced == truth
        n_plain += p_ok
        n_forced += f_ok
        rows.append({
            "sample": i + 1,
            "tokens": g.n_tokens,
            "stopped": "cut off" if g.truncated else "finished",
            "no forcing": "-" if plain is None else str(plain),
            "with forcing": "-" if forced is None else str(forced),
            "verdict": ("no change" if p_ok == f_ok else
                        ("RESCUED" if f_ok else "DAMAGED")),
            "_text": g.text,
            "_forced_tail": forced_text.get(i, ""),
            "_cut": g.truncated,
        })

    summary = {
        "k": k,
        "budget": budget,
        "truth": truth,
        "n_truncated": len(idx),
        "acc_plain": 100.0 * n_plain / k,
        "acc_forced": 100.0 * n_forced / k,
        "rescued": sum(1 for r in rows if r["verdict"] == "RESCUED"),
        "damaged": sum(1 for r in rows if r["verdict"] == "DAMAGED"),
        "gen_s": gen_s,
        "force_s": force_s,
        "tok_s": gen_tokens / gen_s if gen_s > 0 else 0,
        "gen_tokens": gen_tokens,
    }
    return rows, summary


# ============================================================== UI
def build_ui(engine, problems):
    choices = ["Custom problem"] + [
        "AIME25 #%02d  (answer %s)" % (p["idx"] + 1, p["answer"])
        for p in problems]

    warn = ""
    if not engine.comparable:
        warn = ("\n\n> **This backend is 4-bit quantized.** The weights are "
                "compressed to fit a 4 GB card, which changes the model's "
                "outputs. Numbers here demonstrate the *mechanism* but will not "
                "match the report, which was measured in float16 on a T4.")

    header = ("### Budget forcing, live\n"
              "Backend: **%s** &nbsp;|&nbsp; comparable to report: **%s**%s"
              % (engine.label, "yes" if engine.comparable else "NO", warn))

    def pick(sel):
        if sel == "Custom problem" or not problems:
            return gr.update(value=""), gr.update(value=None)
        i = choices.index(sel) - 1
        return (gr.update(value=problems[i]["text"]),
                gr.update(value=problems[i]["answer"]))

    def run_single(question, truth, budget, k, progress=gr.Progress()):
        if not question or not question.strip():
            return "Enter a problem first.", None, ""
        truth = int(truth) if truth not in (None, "") else None
        progress(0.05, desc="Generating %d sample(s) at %d tokens" % (k, budget))
        rows, s = measure(engine, question.strip(), truth, int(budget), int(k),
                          progress)
        progress(0.95, desc="Scoring")

        delta = s["acc_forced"] - s["acc_plain"]
        md = ["#### Result",
              "",
              "| | Without forcing | With forcing |",
              "|---|---|---|",
              "| Accuracy over %d sample(s) | **%.0f%%** | **%.0f%%** |"
              % (s["k"], s["acc_plain"], s["acc_forced"]),
              "",
              "- Cut off before answering: **%d of %d**" % (s["n_truncated"], s["k"]),
              "- Rescued by forcing: **%d** &nbsp; Damaged: **%d**"
              % (s["rescued"], s["damaged"]),
              "- Change: **%+.0f points**" % delta,
              "",
              "Generation %.1fs (%.0f tok/s, %d tokens) &nbsp;|&nbsp; forcing %.1fs"
              % (s["gen_s"], s["tok_s"], s["gen_tokens"], s["force_s"])]
        if truth is None:
            md.append("\n> No known answer supplied, so accuracy cannot be "
                      "scored. The answers themselves are still shown below.")

        table = [[r["sample"], r["tokens"], r["stopped"], r["no forcing"],
                  r["with forcing"], r["verdict"]] for r in rows]

        detail = []
        for r in rows:
            tail = r["_text"][-900:]
            detail.append("### Sample %d - %s (%d tokens)\n\n```\n%s\n```"
                          % (r["sample"], r["stopped"], r["tokens"], tail))
            if r["_cut"]:
                detail.append("**Budget exhausted here.** Forcing appended the "
                              "commit phrase and the model replied: `%s`"
                              % (r["_forced_tail"].strip()[:80] or "(nothing)"))
            detail.append("\n---\n")
        return "\n".join(md), table, "\n".join(detail)

    def run_sweep(question, truth, budgets_txt, k, progress=gr.Progress()):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not question or not question.strip():
            return "Enter a problem first.", None
        truth = int(truth) if truth not in (None, "") else None
        try:
            buds = sorted(set(int(x.strip()) for x in budgets_txt.split(",")
                              if x.strip()))
        except ValueError:
            return "Budgets must be comma-separated integers.", None
        if not buds:
            return "Give at least one budget.", None

        xs, plain, forced = [], [], []
        lines = ["| Budget | No forcing | With forcing | Cut off |",
                 "|---|---|---|---|"]
        for n, b in enumerate(buds):
            progress(n / len(buds), desc="Budget %d" % b)
            _, s = measure(engine, question.strip(), truth, b, int(k))
            xs.append(b)
            plain.append(s["acc_plain"])
            forced.append(s["acc_forced"])
            lines.append("| %d | %.0f%% | %.0f%% | %d/%d |"
                         % (b, s["acc_plain"], s["acc_forced"],
                            s["n_truncated"], s["k"]))

        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
        ax.plot(xs, plain, "o--", color="#B3341F", lw=2, label="No forcing")
        ax.plot(xs, forced, "s-", color="#1D7A4D", lw=2, label="With forcing")
        ax.set_xlabel("Token budget")
        ax.set_ylabel("Accuracy over %d samples (%%)" % k)
        ax.set_ylim(-5, 105)
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        return "\n".join(lines), fig

    def run_repro(n_prob, budget, k, progress=gr.Progress()):
        if not problems:
            return "AIME 2025 could not be fetched, so this tab is unavailable."
        n_prob = int(n_prob)
        tot_p = tot_f = 0
        t0 = time.time()
        for i in range(n_prob):
            progress(i / n_prob, desc="Problem %d of %d" % (i + 1, n_prob))
            _, s = measure(engine, problems[i]["text"], problems[i]["answer"],
                           int(budget), int(k))
            tot_p += s["acc_plain"] * s["k"] / 100.0
            tot_f += s["acc_forced"] * s["k"] / 100.0
        n = n_prob * int(k)
        ap, af = 100.0 * tot_p / n, 100.0 * tot_f / n
        el = time.time() - t0

        md = ["#### Measured here, just now",
              "",
              "| | No forcing | With forcing | Gain |",
              "|---|---|---|---|",
              "| This run (%d problems x K=%d) | **%.1f%%** | **%.1f%%** | %+.1f |"
              % (n_prob, k, ap, af, af - ap)]
        if int(budget) in REPORTED:
            rp, rf = REPORTED[int(budget)]
            md.append("| Report (30 problems x K=4) | %.1f%% | %.1f%% | %+.1f |"
                      % (rp, rf, rf - rp))
        md += ["", "Ran in %.1f minutes on %s." % (el / 60.0, engine.label)]
        if not engine.comparable:
            md.append("\n> 4-bit backend - expect a gap to the reported row.")
        elif n_prob < 30:
            md.append("\n> A %d-problem subset will scatter around the 30-problem "
                      "figure. The direction is the claim, not the exact value."
                      % n_prob)
        return "\n".join(md)

    with gr.Blocks(title="Budget forcing, live",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(header)

        with gr.Tab("Run one problem"):
            with gr.Row():
                dd = gr.Dropdown(choices, value=choices[1] if problems else choices[0],
                                 label="Problem")
            q = gr.Textbox(lines=4, label="Problem statement",
                           value=problems[0]["text"] if problems else "")
            with gr.Row():
                ans = gr.Number(label="Known answer (blank if unknown)",
                                value=problems[0]["answer"] if problems else None,
                                precision=0)
                bud = gr.Slider(512, 16384, value=4096, step=512,
                                label="Token budget")
                kk = gr.Slider(1, 4, value=2, step=1, label="Samples (K)")
            go = gr.Button("Run with and without forcing", variant="primary")
            out_md = gr.Markdown()
            out_tb = gr.Dataframe(
                headers=["sample", "tokens", "stopped", "no forcing",
                         "with forcing", "verdict"],
                label="Per-sample outcome", wrap=True)
            with gr.Accordion("Reasoning traces", open=False):
                out_tr = gr.Markdown()
            dd.change(pick, dd, [q, ans])
            go.click(run_single, [q, ans, bud, kk], [out_md, out_tb, out_tr])

        with gr.Tab("Sweep budgets"):
            gr.Markdown("Run the same problem at several budgets and watch "
                        "accuracy track the budget.")
            q2 = gr.Textbox(lines=3, label="Problem statement",
                            value=problems[0]["text"] if problems else "")
            with gr.Row():
                ans2 = gr.Number(label="Known answer",
                                 value=problems[0]["answer"] if problems else None,
                                 precision=0)
                buds = gr.Textbox(value="1024, 2048, 4096, 8192",
                                  label="Budgets (comma separated)")
                kk2 = gr.Slider(1, 4, value=2, step=1, label="Samples (K)")
            go2 = gr.Button("Run sweep", variant="primary")
            sw_md = gr.Markdown()
            sw_pl = gr.Plot()
            go2.click(run_sweep, [q2, ans2, buds, kk2], [sw_md, sw_pl])

        with gr.Tab("Reproduce the benchmark"):
            gr.Markdown("Run the first N problems of AIME 2025 end to end and "
                        "compare against the reported figures. This is the tab "
                        "that answers *why should I believe the table*.")
            with gr.Row():
                np_ = gr.Slider(1, 30, value=5, step=1, label="Problems")
                bud3 = gr.Slider(1024, 16384, value=4096, step=1024,
                                 label="Token budget")
                kk3 = gr.Slider(1, 4, value=2, step=1, label="Samples (K)")
            gr.Markdown("*Rough cost: problems x K x budget / throughput. "
                        "5 problems, K=2, 4096 tokens is a few minutes on a T4 "
                        "and considerably longer on a 4 GB card.*")
            go3 = gr.Button("Run", variant="primary")
            rp_md = gr.Markdown()
            go3.click(run_repro, [np_, bud3, kk3], rp_md)

        gr.Markdown(
            "---\nOne generation pass per configuration, scored twice: once as "
            "written, once after appending the commit phrase to samples that ran "
            "out of budget. Forcing only ever touches a sample that failed to "
            "finish. The paper reports **%.1f%%** on this benchmark under an "
            "unconstrained budget." % PAPER_AIME25)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "vllm", "hf4bit"])
    ap.add_argument("--max-len", type=int, default=None,
                    help="context window; defaults to 18432 (vllm) / 5120 (4-bit)")
    ap.add_argument("--share", action="store_true",
                    help="public URL - required on Kaggle")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("GPU: %s, %.2f GB, cc %d.%d" %
              (p.name, p.total_memory / 1e9, p.major, p.minor))
    else:
        print("WARNING: no CUDA device found; this will be extremely slow.")

    max_len = a.max_len
    if max_len is None:
        max_len = 18432 if a.backend == "vllm" else 5120

    print("loading model, this takes a few minutes ...")
    engine = build_engine(a.backend, max_len)
    print("loading AIME 2025 ...")
    problems = load_problems()
    print("ready: %d problems, backend %s" % (len(problems), engine.label))

    build_ui(engine, problems).launch(share=a.share, server_port=a.port,
                                      server_name="0.0.0.0")


if __name__ == "__main__":
    main()
