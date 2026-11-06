#!/usr/bin/env python3
"""Word version of the report: report/report.tex -> report/report.docx.

    make report && python3 scripts/make_docx.py

The LaTeX source is flattened into what pandoc reads well (generated macros and tables inlined,
cross-references resolved from report.aux, figures as PNG, the algorithm as a table, side-by-side
minipages as separate floats), converted with pandoc (equations become native Word equations),
and the result is restyled with python-docx: Arial, compact spacing, booktabs-like tables,
captions, header and page numbers.
"""
import os
import re
import subprocess
import tempfile

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "report")
FIG = os.path.join(ROOT, "figures")
FONT = "Arial"


def macros():
    m = {}
    for line in open(os.path.join(REP, "generated", "macros.tex")):
        r = re.match(r"\\(?:newcommand|providecommand)\{\\(\w+)\}\{(.*)\}\s*$", line)
        if r:
            m[r.group(1)] = r.group(2)
    return m


def labels():
    lab = {}
    for line in open(os.path.join(REP, "report.aux")):
        r = re.match(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}", line)
        if r:
            lab[r.group(1)] = r.group(2)
    return lab


KIND = {"fig": "Figure", "tab": "Table", "sec": "Section", "alg": "Algorithm", "eq": "Eq."}


def ref(lab, key, cap=False):
    n = lab.get(key, "?")
    kind = KIND.get(key.split(":")[0], "")
    if kind == "Eq.":
        return f"Eq. ({n})"
    return f"{kind} {n}"


def flatten(tex):
    M, L = macros(), labels()
    body = tex[tex.index("\\begin{document}") + len("\\begin{document}"):tex.index("\\end{document}")]
    # generated tables
    body = re.sub(r"\\tabinput\{([^}]*)\}", lambda r: open(os.path.join(REP, r.group(1))).read(), body)
    # cross-references
    body = re.sub(r"\\[cC]ref\{([^}]*)\}", lambda r: " and ".join(ref(L, k.strip()) for k in r.group(1).split(",")), body)
    body = re.sub(r"\\eqref\{([^}]*)\}", lambda r: f"({L.get(r.group(1), '?')})", body)
    body = re.sub(r"\\ref\{([^}]*)\}", lambda r: L.get(r.group(1), "?"), body)
    # macros (longest names first), with or without a trailing {}
    for k in sorted(M, key=len, reverse=True):
        body = re.sub(r"\\" + k + r"(?:\{\})?(?![A-Za-z])", lambda _r, v=M[k]: v, body)
    body = body.replace("\\textbackslash", "\\")
    # project macros
    body = re.sub(r"\\code\{", r"\\texttt{", body)
    body = re.sub(r"\\vb(?![A-Za-z])", r"\\mathbf", body)
    body = body.replace("\\textonehalf", "1/2")
    # title block
    title = re.search(r"\\begin\{center\}(.*?)\\end\{center\}", body, re.S).group(0)
    body = body.replace(title, "TITLEBLOCK\n")
    body = re.sub(r"\\thispagestyle\{[^}]*\}", "", body)
    body = re.sub(r"\\noindent\{\\large\\bfseries Abstract\}\\par\s*\\noindent", "\\\\section*{Abstract}\n", body)
    body = re.sub(r"\\vspace\{[^}]*\}", "", body)
    # algorithm -> table
    alg = re.search(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", body, re.S).group(0)
    cap = re.search(r"\\caption\{(.*?)\}\\label", alg, re.S).group(1)
    rows, n = [], 0
    for line in alg.splitlines():
        line = line.strip()
        m = re.match(r"\\(State|For)\s*(.*)", line)
        if m:
            n += 1
            txt = m.group(2)
            com = ""
            c = re.search(r"\\Comment\{(.*)\}\s*$", txt)
            if c:
                com = re.sub(r"\\fbox\{([^}]*)\}", r"\\textbf{\1}", c.group(1))
                txt = txt[:c.start()].strip()
            if m.group(1) == "For":
                txt = "\\textbf{for} " + txt.replace("\\textbf{until}", "\\textbf{until}")
            indent = "\\quad " if m.group(1) == "State" and n > 1 else ""
            rows.append(f"{n} & {indent}{txt} & {com} \\\\")
    rows.append(f"{n + 1} & \\textbf{{end for}} & \\\\")
    algtab = ("\\begin{table}\n\\caption{Algorithm 1. " + cap + "}\n\\begin{tabular}{rll}\n\\toprule\n"
              + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")
    body = body.replace(alg, algtab)
    # side-by-side minipages -> separate floats
    def split_minipages(fig):
        parts = re.findall(r"\\begin\{minipage\}(?:\[[^\]]*\])?\{[^}]*\}(.*?)\\end\{minipage\}", fig, re.S)
        out = []
        for p in parts:
            p = p.replace("\\centering", "").replace("\\vspace{0pt}", "")
            p = re.sub(r"\\(small|footnotesize)\b", "", p)
            if "\\captionof{table}" in p:
                p = p.replace("\\captionof{table}", "\\caption")
                out.append("\\begin{table}\n" + p.strip() + "\n\\end{table}")
            else:
                out.append("\\begin{figure}\n" + p.strip() + "\n\\end{figure}")
        return "\n\n".join(out)
    body = re.sub(r"\\begin\{figure\}\[t\]\s*\\centering\s*\\begin\{minipage\}.*?\\end\{figure\}",
                  lambda r: split_minipages(r.group(0)), body, flags=re.S)
    # figures as PNG
    body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]*)\}",
                  lambda r: f"\\includegraphics{{{os.path.join(FIG, r.group(2))}.png}}", body)
    # table helpers pandoc does not need
    body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", body)
    body = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", body)
    body = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", body)
    body = body.replace("\\\\[2pt]", "\\\\")
    body = re.sub(r"\\(small|footnotesize)\b", "", body)
    body = re.sub(r"\\setlength\{\\itemsep\}\{0pt\}\\setlength\{\\parsep\}\{0pt\}", "", body)
    body = body.replace("\\textquotedbl", '"')
    pre = "\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n"
    return pre + body + "\n\\end{document}\n", title


def set_font(run, size=None, bold=None, italic=None, color=None):
    run.font.name = FONT
    r = run._element.get_or_add_rPr()
    rf = r.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        r.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def style_font(st, size, bold=False, color="1A1A1A", before=0, after=4, keep=False, italic=False):
    st.font.name = FONT
    st.font.size = Pt(size)
    st.font.bold = bold
    st.font.italic = italic
    st.font.color.rgb = RGBColor.from_string(color)
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    pf = st.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = keep


def border(cell_or_tbl, edges, sz=8, color="333333"):
    el = cell_or_tbl._tc if hasattr(cell_or_tbl, "_tc") else cell_or_tbl._tbl
    pr = el.tcPr if hasattr(cell_or_tbl, "_tc") else el.tblPr
    if hasattr(cell_or_tbl, "_tc"):
        pr = el.get_or_add_tcPr()
        tag = "w:tcBorders"
    else:
        tag = "w:tblBorders"
    b = pr.find(qn(tag))
    if b is None:
        b = OxmlElement(tag)
        pr.append(b)
    for e in edges:
        x = b.find(qn("w:" + e))
        if x is None:
            x = OxmlElement("w:" + e)
            b.append(x)
        x.set(qn("w:val"), "single" if sz else "nil")
        x.set(qn("w:sz"), str(sz))
        x.set(qn("w:space"), "0")
        x.set(qn("w:color"), color)


def restyle(path, title):
    d = Document(path)
    sec = d.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.2))
    sec.top_margin, sec.bottom_margin = Cm(2.4), Cm(2.2)
    st = d.styles
    for name, kw in [("Normal", dict(size=10, after=5)), ("Body Text", dict(size=10, after=5)),
                     ("First Paragraph", dict(size=10, after=5)), ("Compact", dict(size=9, after=1)),
                     ("Heading 1", dict(size=13, bold=True, before=12, after=5, keep=True)),
                     ("Heading 2", dict(size=11, bold=True, before=9, after=3, keep=True)),
                     ("Heading 3", dict(size=10, bold=True, before=6, after=2, keep=True)),
                     ("Heading 4", dict(size=10, bold=True, before=6, after=2, keep=True)),
                     ("Caption", dict(size=9, after=8, color="333333")),
                     ("Table Caption", dict(size=9, before=6, after=3, color="333333", keep=True)),
                     ("Image Caption", dict(size=9, after=8, color="333333")),
                     ("Abstract", dict(size=10, after=5))]:
        hit = [x for x in st if x.name and x.name.lower() == name.lower()]
        if hit:
            style_font(hit[0], **kw)
    for p in d.paragraphs:
        for r in p.runs:
            set_font(r)
        if p.style.name in ("Body Text", "First Paragraph", "Normal"):
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.line_spacing = 1.12
        if p.style.name in ("Captioned Figure", "Figure"):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # images: full text width, centred
    width = sec.page_width - sec.left_margin - sec.right_margin
    for shp in d.inline_shapes:
        ratio = shp.height / shp.width
        shp.width = int(width * (0.55 if shp.width < Cm(12) else 1.0))
        shp.height = int(shp.width * ratio)
    for p in d.paragraphs:
        if p._element.findall(".//" + qn("w:drawing")):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
    # tables: booktabs look, centred, compact, column widths from the longest entry of each column
    for t in d.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        ncol = len(t.columns)
        chars, need = [3] * ncol, [3] * ncol      # full text length / longest unbreakable word
        for row in t.rows:
            for j, c in enumerate(row.cells[:ncol]):
                txt = (c.text + " " + "".join(x.text or "" for x in c._tc.iter(qn("m:t")))).strip()
                need[j] = max(need[j], max((len(w) for w in txt.split()), default=0) + 1)
                chars[j] = max(chars[j], min(len(txt), 30))
        cw, pad = Cm(0.165), Cm(0.42)
        base = [pad + cw * n for n in need]
        full = [pad + cw * max(c, n) for c, n in zip(chars, need)]
        if sum(full) <= width:
            widths = [int(w) for w in full]
        else:                                   # give every column its longest word, share the rest
            extra = max(0, width - sum(base))
            want = [f - b0 for f, b0 in zip(full, base)]
            k = extra / sum(want) if sum(want) else 0
            widths = [int(b0 + k * w) for b0, w in zip(base, want)]
        if sum(widths) > width:                 # still too wide: shrink every column alike
            widths = [int(w * width / sum(widths)) for w in widths]
        tblpr = t._tbl.tblPr
        for tag in ("w:tblW", "w:tblLayout"):
            for x in tblpr.findall(qn(tag)):
                tblpr.remove(x)
        tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(int(sum(widths) / 635))); tw.set(qn("w:type"), "dxa")
        tblpr.append(tw)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblpr.append(lay)
        grid = t._tbl.tblGrid
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            if j < ncol:
                gc.set(qn("w:w"), str(int(widths[j] / 635)))
        for row in t.rows:
            for j, c in enumerate(row.cells[:ncol]):
                c.width = widths[j]
        tblpr = t._tbl.tblPr
        for x in tblpr.findall(qn("w:tblStyle")):
            tblpr.remove(x)
        border(t, ["top", "bottom"], 8)
        border(t, ["left", "right", "insideH", "insideV"], 0)
        for i, row in enumerate(t.rows):
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.space_before = Pt(0)
                    for r in p.runs:
                        set_font(r, size=8.5, bold=True if i == 0 and len(t.rows) > 2 else None)
            if i == 0:
                for c in row.cells:
                    border(c, ["bottom"], 4)
                trpr = row._tr.get_or_add_trPr()
                h = OxmlElement("w:tblHeader"); trpr.append(h)
            trpr = row._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); trpr.append(cs)
            if i < len(t.rows) - 1:             # keep short tables on one page
                for c in row.cells:
                    for p in c.paragraphs:
                        p.paragraph_format.keep_with_next = True
    # title block
    first = d.paragraphs[[i for i, p in enumerate(d.paragraphs) if "TITLEBLOCK" in p.text][0]]
    lines = [("Scaling Conjugate Gradients with MPI and OpenMP", 17, True),
             ("Decomposition, communication hiding, memory bandwidth and GPU offload in a matrix-free 3D Poisson solver", 11, False),
             ("WU YUHAN (A0351706H)     ZUO DEWEI (A0352507H)     ZHU CHEN (A0357394N)", 10, False),
             ("DSA5208 · Project 3", 10, False)]
    for k, (txt, size, bold) in enumerate(lines):
        p = first.insert_paragraph_before("")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8 if k in (1, 2) else 4)
        set_font(p.add_run(txt), size=size, bold=bold)
    first._element.getparent().remove(first._element)
    # schema order of table properties (elements were appended above)
    order = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize",
             "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
             "tblCaption", "tblDescription"]
    for t in d.tables:
        pr = t._tbl.tblPr
        kids = list(pr)
        kids.sort(key=lambda e: order.index(e.tag.split("}")[1]) if e.tag.split("}")[1] in order else 99)
        for k in kids:
            pr.remove(k)
        for k in kids:
            pr.append(k)
    sec.gutter = Cm(0)
    border_order = ["top", "left", "start", "bottom", "right", "end", "insideH", "insideV", "tl2br", "tr2bl"]
    ppr_order = ("pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers "
                 "pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE "
                 "autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents "
                 "suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle "
                 "rPr sectPr pPrChange").split()

    def sort_children(el, order):
        kids = list(el)
        kids.sort(key=lambda e: order.index(e.tag.split("}")[1]) if e.tag.split("}")[1] in order else 999)
        for k in kids:
            el.remove(k)
        for k in kids:
            el.append(k)
    body = d.element.body
    for tag in ("w:tblBorders", "w:tcBorders"):
        for el in body.iter(qn(tag)):
            sort_children(el, border_order)
    tcpr_order = "cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark".split()
    for el in body.iter(qn("w:tcPr")):
        sort_children(el, tcpr_order)
    sec.header_distance, sec.footer_distance = Cm(1.2), Cm(1.0)
    for root in (body, d.styles.element):
        for el in root.iter(qn("w:pPr")):
            sort_children(el, ppr_order)
    # header and page number
    hp = sec.header.paragraphs[0]
    hp.text = ""
    set_font(hp.add_run("DSA5208 | Parallel Conjugate Gradients\tMPI + OpenMP on NUS Vanda"), size=8.5, color="444444")
    hp.paragraph_format.tab_stops.add_tab_stop(width, alignment=2)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    set_font(run, size=8.5)
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            f = OxmlElement("w:fldChar")
            f.set(qn("w:fldCharType"), kind)
            run._r.append(f)
        else:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = text
            run._r.append(it)
    d.save(path)


def main():
    for f in os.listdir(FIG):
        if f.endswith(".pdf"):
            subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile", os.path.join(FIG, f),
                            os.path.join(FIG, f[:-4])], check=True)
    tex, title = flatten(open(os.path.join(REP, "report.tex")).read())
    out = os.path.join(REP, "report.docx")
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False) as t:
        t.write(tex)
    subprocess.run(["pandoc", t.name, "-f", "latex", "-t", "docx", "-o", out, "--resource-path", ROOT],
                   check=True)
    restyle(out, title)
    print("->", out)


if __name__ == "__main__":
    main()
