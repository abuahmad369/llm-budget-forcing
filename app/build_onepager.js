const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
} = require("docx");

const ROOT = "D:\\semester\\14. Summer 2026\\CSE465\\ViveThinker upgration";
const OUT = ROOT + "\\report\\Result_Summary_OnePager.docx";
const FIG = ROOT + "\\figures\\fig6_summary_print.png";

const F = "Cambria";
const MONO = "Consolas";
const NAVY = "1F3864";
const GREEN = "1D7A4D";
const RED = "B3341F";

const c = [];

/* ---------- helpers ---------- */
function run(o) {
  return new TextRun({
    text: o.t, font: o.mono ? MONO : F, size: o.size || 19,
    bold: !!o.b, italics: !!o.i, color: o.c || "000000",
  });
}
function P(runs, opts) {
  opts = opts || {};
  return new Paragraph({
    alignment: opts.align || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after === undefined ? 100 : opts.after, line: opts.line || 250 },
    children: (Array.isArray(runs) ? runs : [{ t: runs }]).map(run),
  });
}
function H(txt, opts) {
  opts = opts || {};
  return new Paragraph({
    spacing: { before: opts.before === undefined ? 140 : opts.before, after: 70 },
    children: [new TextRun({ text: txt, font: F, size: 20, bold: true, color: NAVY })],
  });
}
function cell(o) {
  return new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.head ? { type: ShadingType.CLEAR, fill: "E8EEF7" }
      : (o.fill ? { type: ShadingType.CLEAR, fill: o.fill } : undefined),
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({
      alignment: o.left ? AlignmentType.LEFT : AlignmentType.CENTER,
      spacing: { after: 0, line: 220 },
      children: [new TextRun({
        text: o.t, font: o.mono ? MONO : F, size: 18,
        bold: !!o.b || !!o.head, color: o.c || "000000",
      })],
    })],
  });
}
function table(widths, rows) {
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce(function (a, b) { return a + b; }, 0), type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: "7A8798" },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: "7A8798" },
      left: { style: BorderStyle.SINGLE, size: 2, color: "C3CBD6" },
      right: { style: BorderStyle.SINGLE, size: 2, color: "C3CBD6" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "C3CBD6" },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: "C3CBD6" },
    },
    rows: rows.map(function (r, ri) {
      return new TableRow({
        children: r.map(function (x, ci) {
          var o = (typeof x === "string") ? { t: x } : x;
          o.w = widths[ci];
          o.head = (ri === 0);
          o.left = (ci === 0);
          return cell(o);
        }),
      });
    }),
  });
}

/* ---------- title ---------- */
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({
    text: "Reasoning Termination, Not Reasoning Capability",
    font: F, size: 30, bold: true, color: NAVY,
  })],
}));
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 30 },
  children: [new TextRun({
    text: "Budget forcing on VibeThinker-3B, AIME 2025",
    font: F, size: 21, italics: true, color: "444444",
  })],
}));
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 130 },
  children: [new TextRun({
    text: "Abu Ahmad  \u00b7  2121725042  \u00b7  CSE 465  \u00b7  North South University",
    font: F, size: 17, color: "666666",
  })],
}));

/* ---------- paragraph ---------- */
c.push(H("The finding", { before: 0 }));
c.push(P([
  { t: "VibeThinker-3B scores " },
  { t: "91.4", b: true },
  { t: " on AIME 2025 in its paper; we reproduced " },
  { t: "80.8", b: true },
  { t: ". The gap is not weaker reasoning \u2014 the model solves the problem and then never writes the answer down. Of 120 sampled attempts, essentially every one that finished was correct, while every one cut off by the token budget scored zero for want of a boxed answer. Across 1,200 sample-budget observations the model volunteered " },
  { t: "exactly one wrong answer", b: true },
  { t: ": it states the correct answer or states nothing at all. Interrupting it at the budget and forcing it to commit recovers " },
  { t: "7 to 13 points", b: true },
  { t: ", has " },
  { t: "never", b: true, c: GREEN },
  { t: " destroyed a correct answer, and costs about five tokens." },
], { after: 120 }));

/* ---------- figure ---------- */
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 50 },
  children: [new ImageRun({
    data: fs.readFileSync(FIG), type: "png",
    transformation: { width: 510, height: 397 },
  })],
}));
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({
    text: "Top: accuracy rises with the token budget as truncation falls \u2014 mirror images. "
        + "Bottom: forcing converts the amber \u201cno answer\u201d block into correct and wrong, and the green never shrinks.",
    font: F, size: 16, italics: true, color: "444444",
  })],
}));

/* ---------- tables ---------- */
c.push(H("Table 1  \u00b7  Main result"));
c.push(table([1500, 1750, 1750, 1300, 1750], [
  ["Budget", "No forcing", "+ Forcing", "Gain", "McNemar p"],
  ["4k", "27.5%", "40.8%", "+13.3", "\u2014"],
  [{ t: "8k", b: true }, { t: "46.7%", b: true }, { t: "55.8%", b: true },
   { t: "+9.1", b: true, c: GREEN }, { t: "0.0026", b: true, mono: true }],
  [{ t: "16k", b: true }, { t: "63.3%", b: true }, { t: "70.8%", b: true },
   { t: "+7.5", b: true, c: GREEN }, { t: "0.0077", b: true, mono: true }],
  ["32k", "80.8%", "not run", "\u2014", "\u2014"],
]));

c.push(H("Table 2  \u00b7  Does forcing ever hurt?"));
c.push(table([1500, 1750, 1750, 1550, 1500], [
  ["Budget", "Truncated", "Rescued", "Damaged", "p"],
  ["8k", "81", { t: "11", c: GREEN, b: true }, { t: "0", b: true, c: GREEN, fill: "DCEFE4" }, { t: "0.0026", mono: true }],
  ["16k", "58", { t: "9", c: GREEN, b: true }, { t: "0", b: true, c: GREEN, fill: "DCEFE4" }, { t: "0.0077", mono: true }],
]));
c.push(P([{ t: "Zero damage across 139 interventions. Forcing only ever touches a sample that had already failed.", i: true, size: 17 }], { after: 60 }));

c.push(H("Table 3  \u00b7  What the 120 samples are made of"));
c.push(table([2600, 1750, 1750, 1950], [
  ["Condition", "Correct", "Wrong", "No answer"],
  ["8k, no forcing", "56", { t: "0", b: true, c: RED, fill: "F7DED8" }, "64"],
  ["8k, forced", { t: "67", b: true }, "53", "0"],
  ["16k, no forcing", "76", { t: "0", b: true, c: RED, fill: "F7DED8" }, "44"],
  ["16k, forced", { t: "85", b: true }, "33", "2"],
]));
c.push(P([{ t: "The zero column is the finding: left alone, the model never guesses.", i: true, size: 17 }], { after: 90 }));

/* ---------- footer ---------- */
c.push(new Paragraph({
  spacing: { before: 60 },
  border: { top: { style: BorderStyle.SINGLE, size: 6, color: "C3CBD6", space: 6 } },
  children: [new TextRun({
    text: "All values measured on a Kaggle Tesla T4, float16, vLLM \u2014 30 problems, K = 4, 22.6 GPU-hours. "
        + "Nothing is estimated. Code and data: github.com/abuahmad369/llm-budget-forcing",
    font: F, size: 15, color: "666666",
  })],
}));

/* ---------- doc ---------- */
const doc = new Document({
  styles: { default: { document: { run: { font: F, size: 19 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 850, bottom: 750, left: 900, right: 900 },
      },
    },
    children: c,
  }],
});

Packer.toBuffer(doc).then(function (b) {
  fs.writeFileSync(OUT, b);
  console.log("written:", OUT, b.length, "bytes");
});
