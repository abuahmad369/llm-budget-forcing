# Writing brief

Everything the report needs, so you can write the prose without hunting through
old files for numbers.

`report_skeleton.tex` has 52 writing slots. Each renders as a yellow box in the
compiled PDF telling you what that paragraph must establish. Delete the whole
`\writehere{...}{...}` and put your paragraph in its place.

---

## How to use it

Upload `report_skeleton.tex` plus the `figures/` folder to Overleaf. Compile with
pdfLaTeX, twice, so the contents pages resolve. You will get the full document
with yellow boxes where your writing goes.

Write one chapter at a time. When you finish a slot, delete the box and put your
paragraph there. When everything is written, change `\drafttrue` to `\draftfalse`
near the top and recompile to see the real page count.

---

## Style rules that apply here

No dash characters in your prose. Not the em dash, not the en dash, not the plain
hyphen. Rewrite compounds instead: write "long horizon", "test time scaling",
"pp. 46534 to 46594". The exception is a proper noun that genuinely contains one,
such as VibeThinker-3B or DeepSeek-R1. Search the file before you submit and check
that every remaining hyphen is either a proper noun or LaTeX syntax.

Sentence case for headings, which the skeleton already uses.

Vary your sentence length deliberately. A four word sentence next to a forty word
one is what human writing looks like. Machine prose settles into an even middle
length rhythm.

Take positions. A report that only summarises reads as generated even when every
sentence is clean. Say plainly that you abandoned a direction, that you built a
hybrid rule and threw it away, that you retracted a metric. Those are the parts
that read as a person.

Keep the odd specific detail. The value 588 appearing in the model's working
before it gave up. Problem 13 producing 124 and 184. The fact that the first
throughput measurement projected to over a hundred GPU hours and killed the
original plan. Detail like that cannot be generated from a summary.

Where the genre allows first person, use it. Chapter 4 discussion and Chapter 5
limitations both allow it.

Do not write a stock challenges paragraph or a hopeful closing line. End sections
on the last real fact.

---

## Every number, in one place

Setup for all of it: Kaggle free tier, one Tesla T4 with 15.64 GB and compute
capability 7.5, float16, vLLM, `math-ai/aime25`, 30 problems, K equals 4 giving
120 samples per condition, temperature 1.0, top_p 0.95, seed 0.

### Accuracy against budget, no forcing

1k 3.3 percent, truncated 100.0
2k 13.3, truncated 98.3
3k 23.3, truncated 92.5
4k 30.0, truncated 87.5
6k 38.3, truncated 77.5
8k 46.7, truncated 67.5
10k 51.7, truncated 63.3
12k 56.7, truncated 59.2
14k 58.3, truncated 53.3
16k 63.3, truncated 48.3
32k 80.8, truncated 20.8

Paper reports 91.4 uncapped.

### Forcing

4k: 27.5 to 40.8, gain plus 13.3, McNemar not run
8k: 46.7 to 55.8, gain plus 9.1, chi square 9.09, p equals 0.0026
16k: 63.3 to 70.8, gain plus 7.5, chi square 7.11, p equals 0.0077
32k: 80.8, forcing not measured

The 4k row has two baselines because that budget was measured twice, 27.5 from a
dedicated run and 30.0 from truncated traces. Three samples apart. Say so.

### Harm

8k: 81 truncated, 11 rescued, 0 damaged, 17 right to right
16k: 58 truncated, 9 rescued, 0 damaged, 14 right to right
Total 139 forced samples, zero damaged.

### Composition of the 120 samples

8k no forcing: 56 correct, 0 wrong, 64 no answer
8k forced: 67 correct, 53 wrong, 0 no answer
16k no forcing: 76 correct, 0 wrong, 44 no answer
16k forced: 85 correct, 33 wrong, 2 no answer

Across all 1,200 sample budget observations the model volunteered exactly one
wrong answer.

### Recovery rate among rescuable samples

8k: 81 truncated, 17 already correct, 64 rescuable, 11 rescued, 17.2 percent
16k: 58 truncated, 14 already correct, 44 rescuable, 9 rescued, 20.5 percent
Random guessing on AIME, integers 0 to 999, succeeds 0.1 percent of the time.

### Upper bounds

Unforced pass@4 against forced pass@1:
4k 40.0 against 40.8
8k 53.3 against 55.8
16k 73.3 against 70.8

Extraction ceiling captured 78.6 percent at 8k and 81.8 at 16k. Wilson intervals
roughly 52 to 92 and 52 to 95. Populations of 14 and 11.
False positive control ran 0 to 4.9 percent against a signal of 32 to 41 percent.
Only 16 of 64 rescuable traces at 8k and 11 of 44 at 16k mention the answer in
their final fifth.

### Forcing variants

8k: V1 55.8, V3 55.8, V2 56.7
16k: V1 70.8, V3 70.8, V2 70.8
V2 cost 47.8 minutes against 21.2 at 8k, and 117.3 against 54.9 at 16k. That is
2.1 times.

### Throughput on the same T4

HuggingFace transformers K equals 1: 15.2 tokens per second
vLLM K equals 1: 31.4
vLLM K equals 8: 145.9
vLLM K equals 32: 246.4
Speedup 16.2 times.

### Compute

Setup and probes 1.50, baseline 8.87, forcing at 4k and 8k 2.06, forcing variants
8.89, trace analysis 0.00 on CPU, paired analysis 1.28, upper bounds 0.00 on CPU.
Total 22.60 GPU hours.

### Other facts

Median trace length 15,961 tokens.
KV cache 192,272 tokens at 0.90 utilisation.
Validation: 8k measured two ways gave 54.2 and 55.8, two samples out of 120 apart.

---

## The two worked examples

Use these in Chapter 4 discussion. They are the most convincing thing you have.

**Problem 1, answer 588, at 16k.** Three of four summaries from the summarise
variant contain real derived quantities: `AB = 28`, `AC = 91`, coordinates for the
auxiliary point F. Forcing yields 588, which is correct. The model had the answer
and needed permission to state it.

**Problem 13, answer 60.** The summaries are still setting up the problem. One
reads that the geometric median is the unique point where the sum is minimised.
Forcing yields 124 and 184, both wrong. The model never derived an answer, so
forcing invented one.

Those two together are the mechanism: forcing recovers what is already latent and
cannot manufacture what never happened.

---

## Things you must not write

**No number that was never measured.** Forcing at 32k was not run. Forced pass@4
was not computed. Run to run variance was never estimated. If a sentence needs one
of those, write it without a number or say it was not measured.

**Do not present the timings as the cost of the method.** Forcing's wall clock
cost is prefill from reprocessing each trace. The forcing step itself generates
about five tokens. A cache reusing implementation would make it nearly free.

**Figure 4.4 caption must say right censored.** The mass at 16,384 tokens is your
imposed limit, not the model stopping. Without that line a reader concludes the
model naturally stops around 16k, which is false.

**Report McNemar, not the bootstrap.** The design is paired. The bootstrap was
conservative so the conclusion is unchanged, but McNemar is the correct test.

**Do not plot the 4k forcing point beside 8k and 16k as one series.** Its baseline
came from a different run.

---

## The retracted metric

Keep this in the limitations. An early ceiling estimate divided rescued samples by
a denominator that turned out to be the already correct set, so numerator and
denominator were disjoint sets and the ratio meant nothing. You caught it,
recomputed against truncated and wrong samples, and reported both.

A marker who sees a caught error reads someone who checked their own work. Hiding
it gains nothing and costs you the one thing in the report that proves you were
paying attention.

---

## Order to write in

Chapter 4 first, while the numbers are in front of you and the writing is mostly
description. Then Chapter 3, which is procedure. Then Chapter 5, which follows
from Chapter 4. Then Chapter 2, once you know what gap your results actually fill.
Then Chapter 1, which is easiest once the rest exists. The abstract last, because
it summarises a document that by then exists.

Acknowledgements whenever you like.

---

## Before you submit

Set `\draftfalse` and recompile so the yellow boxes disappear.

Compile twice so the table of contents, list of figures and list of tables
resolve. On a single pass they show as question marks.

Search the source for hyphen characters and confirm every survivor is either a
proper noun or LaTeX syntax such as `\cmidrule` or a TikZ path.

Read every number in your prose against the list above.

Check that each of the ten references is cited somewhere in your text. The
skeleton cites all ten already, but if you delete a paragraph you may orphan one.
