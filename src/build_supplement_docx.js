// Builds Supplementary_File_S1.docx from supplement_tables.json + Figures S1-S3.
// Usage: node build_supplement_docx.js <manuscript_dir>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, PageOrientation, HeadingLevel, ImageRun, PageBreak, BorderStyle, Footer, PageNumber,
} = require("docx");

const dir = process.argv[2];
const data = JSON.parse(fs.readFileSync(path.join(dir, "supplement_tables.json"), "utf8"));
const W = 15398; // A4 landscape text width in DXA with 0.8" margins (16838 - 2*720)
const FONT = "Arial";
const border = { style: BorderStyle.SINGLE, size: 2, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

function colWidths(part) {
  const n = part.columns.length;
  const len = part.columns.map((c, j) => {
    let m = c.length;
    for (const r of part.rows.slice(0, 200)) m = Math.max(m, (r[j] || "").length);
    return Math.min(Math.max(m, 4), 60);
  });
  const tot = len.reduce((a, b) => a + b, 0);
  const w = len.map((l) => Math.max(500, Math.floor((W * l) / tot)));
  const s = w.reduce((a, b) => a + b, 0);
  w[w.length - 1] += W - s;
  return w;
}

function cell(text, width, header) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders,
    shading: header ? { fill: "E8F1FB", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    children: [new Paragraph({ children: [new TextRun({ text: String(text), font: FONT, size: 14, bold: header })] })],
  });
}

function table(part) {
  const w = colWidths(part);
  const rows = [new TableRow({ tableHeader: true, children: part.columns.map((c, j) => cell(c, w[j], true)) })];
  for (const r of part.rows) rows.push(new TableRow({ children: r.map((v, j) => cell(v, w[j], false)) }));
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: w, rows });
}

const P = (text, opts = {}) => new Paragraph({ spacing: { after: 120 }, ...opts, children: [new TextRun({ text, font: FONT, size: opts.size || 20, bold: opts.bold, italics: opts.italics })] });

const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun({ text: "Supplementary File S1", font: FONT })] }),
  P("Routine Blood Tests, Cox Regression and Machine Learning for Predicting Progression-Free Survival in Metastatic Breast Cancer: An Internal–External Validation Study Across Five Phase III Trials", { bold: true }),
  P("Deniz Kenan Kılıç"),
  P("Contents: Tables S1–S9 and Figures S1–S3. Trial codes: PFE111, Pfizer A6181107 (NCT00373113); PFE113, Pfizer A6181099 (NCT00435409); SAN135, Sanofi EFC6089 (NCT00081796); PFE112, Pfizer A6181094 (NCT00373256); LLY168, Eli Lilly ROSE/TRIO-12 (NCT00703326). Models: M0 points score; M1 Cox, clinical; M2 Cox, clinical + blood; M3 LASSO-Cox; M4 random survival forest; M5 gradient-boosted Cox; M6 DeepSurv. Only aggregate results are reported; no patient-level data are included."),
];
for (const t of data) {
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: `${t.code}. ${t.title}`, font: FONT })] }));
  for (const part of t.parts) {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 80 }, children: [new TextRun({ text: part.label, font: FONT })] }));
    children.push(table(part));
  }
}
const figs = [
  ["FigureS1.png", "Figure S1. Harrell's C-index of all models in each held-out trial (symbols) and random-effects pooled estimate with 95% CI (diamond and thick line); dotted line, 95% prediction interval (progression-free survival)."],
  ["FigureS2.png", "Figure S2. Harrell's C-index in held-out trials for overall survival (three trials with long-term survival follow-up); prediction intervals are not shown because they are not estimable with three trials."],
  ["FigureS3.png", "Figure S3. Calibration of predicted 12-month mortality risk in held-out trials (overall survival)."],
];
for (const [f, cap] of figs) {
  const buf = fs.readFileSync(path.join(dir, "figures", f));
  const wpx = buf.readUInt32BE(16), hpx = buf.readUInt32BE(20);
  const maxW = 900, maxH = 520;
  const sc = Math.min(maxW / wpx, maxH / hpx);
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: buf, transformation: { width: Math.round(wpx * sc), height: Math.round(hpx * sc) } })] }));
  children.push(P(cap, { size: 18 }));
}

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838, orientation: PageOrientation.LANDSCAPE }, margin: { top: 720, bottom: 720, left: 720, right: 720 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(path.join(dir, "Supplementary_File_S1.docx"), b); console.log("docx written"); });
