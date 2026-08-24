/* One printable A4 sheet: the finding, then every result as a table.
 *
 * No figure. The accuracy curve therefore becomes a table of its own, so the
 * sheet still carries everything the illustrated version shows.
 *
 * Build:  node build_tables_sheet.js
 */
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
} = require("docx");

const ROOT = "D:\\semester\\14. Summer 2026\\CSE465\\ViveThinker upgration";
const OUT = ROOT + "\\report\\Result_Tables_Print.docx";

const F = "Cambria";
const MONO = "Consolas";
const NAVY = "1F3864";
const GREEN = "1D7A4D";
const RED = "B3341F";
const AMBER = "8A6D1F";
const W = 10100;                    // usable width in twips

const c = [];

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
    spacing: { after: opts.after === undefined ? 90 : opts.after, line: opts.line || 250 },
    children: (Array.isArray(runs) ? runs : [{ t: runs }]).map(run),
  });
}
function H(txt, opts) {
  opts = opts || {};
  return new Paragraph({
    spacing: { before: opts.before === undefined ? 150 : opts.before, after: 60 },
    children: [new TextRun({ text: txt, font: F, size: 20, bold: true, color: NAVY })],
  });
}
function cell(o) {
  return new TableCell({
    width: { size: o.w, type: WidthType.DXA },
    shading: o.head ? { type: ShadingType.CLEAR, fill: "E8EEF7" }
      : (o.fill ? { type: ShadingType.CLEAR, fill: o.fill } : undefined),
    margins: { top: 34, bottom: 34, left: 60, right: 60 },
    children: [new Paragraph({
      alignment: o.left ? AlignmentType.LEFT : AlignmentType.CENTER,
      spacing: { after: 0, line: 210 },
      children: [new TextRun({
        text: o.t, font: o.mono ? MONO : F, size: o.size || 17,
        bold: !!o.b || !!o.head, color: o.c || "000000",
      })],
    })],
  });
}
function table(widths, rows, opts) {
  opts = opts || {};
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
          var o = (typeof x === "string") ? { t: x } : Object.assign({}, x);
          o.w = widths[ci];
          if (ri === 0 && !opts.noHead) o.head = true;
          if (ci === 0) { o.left = true; if (opts.boldFirst) o.b = true; }
          if (opts.size) o.size = o.size || opts.size;
          return cell(o);
        }),
      });
    }),
  });
}

/* ---------------- title ---------------- */
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 35 },
  children: [new TextRun({
    text: "Reasoning Termination, Not Reasoning Capability",
    font: F, size: 30, bold: true, color: NAVY,
  })],
}));
c.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 25 },
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

/* ---------------- the finding ---------------- */
c.push(H("The finding", { before: 0 }));
c.push(P([
  { t: "VibeThinker-3B scores " }, { t: "91.4", b: true },
  { t: " on AIME 2025 in its paper; we reproduced " }, { t: "80.8", b: true },
  { t: ". The gap is not weaker reasoning \u2014 the model solves the problem and then never writes the answer down. Of 120 sampled attempts, essentially every one that finished was correct, while every one cut off by the token budget scored zero for want of a boxed answer. Across 1,200 sample-budget observations the model volunteered " },
  { t: "exactly one wrong answer", b: true },
  { t: ": it states the correct answer or states nothing at all. Interrupting it at the budget and forcing it to commit recovers " },
  { t: "7 to 13 points", b: true }, { t: ", has " },
  { t: "never", b: true, c: GREEN },
  { t: " destroyed a correct answer, and costs about five tokens." },
], { after: 130 }));

/* ---------------- table 1: the curve ---------------- */
c.push(H("Table 1  \u00b7  Accuracy against token budget, without forcing"));
var bud = ["1k", "2k", "3k", "4k", "6k", "8k", "10k", "12k", "14k", "16k", "32k"];
var acc = ["3.3", "13.3", "23.3", "30.0", "38.3", "46.7", "51.7", "56.7", "58.3", "63.3", "80.8"];
var cut = ["100", "98.3", "92.5", "87.5", "77.5", "67.5", "63.3", "59.2", "53.3", "48.3", "20.8"];
var wCurve = [1560];
for (var i = 0; i < 11; i++) wCurve.push(Math.floor((W - 1560) / 11));
c.push(table(wCurve, [
  ["Budget"].concat(bud),
  [{ t: "Accuracy %", b: true }].concat(acc.map(function (v) { return { t: v, mono: true }; })),
  [{ t: "Cut off %", b: true, c: AMBER }].concat(cut.map(function (v) {
    return { t: v, mono: true, c: AMBER };
  })),
], { size: 16 }));
c.push(P([{ t: "Accuracy and truncation are mirror images. This is the whole argument: what changes across the row is the budget, not the model.", i: true, size: 16 }], { after: 70 }));

/* ---------------- table 2: forcing ---------------- */
c.push(H("Table 2  \u00b7  Effect of budget forcing"));
c.push(table([1500, 1900, 1900, 1500, 1750], [
  ["Budget", "No forcing", "+ Forcing", "Gain", "McNemar p"],
  ["4k", "27.5%", "40.8%", "+13.3", "\u2014"],
  [{ t: "8k", b: true }, { t: "46.7%", b: true }, { t: "55.8%", b: true },
   { t: "+9.1", b: true, c: GREEN }, { t: "0.0026", b: true, mono: true }],
  [{ t: "16k", b: true }, { t: "63.3%", b: true }, { t: "70.8%", b: true },
   { t: "+7.5", b: true, c: GREEN }, { t: "0.0077", b: true, mono: true }],
  ["32k", "80.8%", "not run", "\u2014", "\u2014"],
]));

/* ---------------- table 3: damage ---------------- */
c.push(H("Table 3  \u00b7  Does forcing ever destroy a correct answer?"));
c.push(table([1500, 1750, 1750, 1700, 1550], [
  ["Budget", "Truncated", "Rescued", "Damaged", "McNemar p"],
  ["8k", "81", { t: "11", b: true, c: GREEN }, { t: "0", b: true, c: GREEN, fill: "DCEFE4" }, { t: "0.0026", mono: true }],
  ["16k", "58", { t: "9", b: true, c: GREEN }, { t: "0", b: true, c: GREEN, fill: "DCEFE4" }, { t: "0.0077", mono: true }],
]));
c.push(P([{ t: "Zero damage across 139 interventions. Forcing only ever acts on a sample that had already failed; one that stopped on its own is left untouched.", i: true, size: 16 }], { after: 70 }));

/* ---------------- table 4: composition ---------------- */
c.push(H("Table 4  \u00b7  What the 120 samples are made of"));
c.push(table([2500, 1800, 1800, 1900], [
  ["Condition", "Correct", "Wrong", "No answer"],
  ["8k, no forcing", "56", { t: "0", b: true, c: RED, fill: "F7DED8" }, { t: "64", c: AMBER }],
  ["8k, forced", { t: "67", b: true, c: GREEN }, "53", "0"],
  ["16k, no forcing", "76", { t: "0", b: true, c: RED, fill: "F7DED8" }, { t: "44", c: AMBER }],
  ["16k, forced", { t: "85", b: true, c: GREEN }, "33", "2"],
]));
c.push(P([{ t: "The zero column is the sharpest result in the project: left alone, the model never guesses. Forcing converts silence into a commitment \u2014 and a commitment can be wrong.", i: true, size: 16 }], { after: 70 }));

/* ---------------- table 5: recovery ---------------- */
c.push(H("Table 5  \u00b7  Recovery rate among samples forcing can actually help"));
c.push(table([1400, 1600, 2000, 1700, 1650, 1750], [
  ["Budget", "Truncated", "Already correct", "Rescuable", "Rescued", "Rate"],
  ["8k", "81", "17", "64", "11", { t: "17.2%", b: true, c: GREEN }],
  ["16k", "58", "14", "44", "9", { t: "20.5%", b: true, c: GREEN }],
]));
c.push(P([{ t: "Roughly one failed sample in five is recovered. Random guessing on AIME, where answers are integers from 0 to 999, would succeed 0.1% of the time \u2014 so forcing is about 150 times better than chance, and is reading something real out of the partial trace.", i: true, size: 16 }], { after: 100 }));

/* ---------------- footer ---------------- */
c.push(new Paragraph({
  spacing: { before: 50 },
  border: { top: { style: BorderStyle.SINGLE, size: 6, color: "C3CBD6", space: 6 } },
  children: [new TextRun({
    text: "All values measured on a Kaggle Tesla T4, float16, vLLM \u2014 30 problems, K = 4, 22.6 GPU-hours. "
        + "Nothing is estimated; quantities that were not measured are marked as such. "
        + "Code and data: github.com/abuahmad369/llm-budget-forcing",
    font: F, size: 15, color: "666666",
  })],
}));

const doc = new Document({
  styles: { default: { document: { run: { font: F, size: 19 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 800, bottom: 700, left: 900, right: 900 },
      },
    },
    children: c,
  }],
});

Packer.toBuffer(doc).then(function (b) {
  fs.writeFileSync(OUT, b);
  console.log("written:", OUT, b.length, "bytes");
});
