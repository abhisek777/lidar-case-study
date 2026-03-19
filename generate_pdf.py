"""Generate PDF report from REPORT.md using reportlab."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import re, os

INPUT  = os.path.join(os.path.dirname(__file__), "REPORT.md")
OUTPUT = os.path.join(os.path.dirname(__file__), "REPORT_Kalpana_Abhiseka_Maddi_DLMDSEAAD02.pdf")

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def S(name, **kw):
    base = styles[name] if name in styles else styles["Normal"]
    return ParagraphStyle(name + str(id(kw)), parent=base, **kw)

title_style   = S("Title",   fontSize=18, leading=24, spaceAfter=6,  textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
subtitle_style= S("Normal",  fontSize=11, leading=14, spaceAfter=4,  textColor=colors.HexColor("#444444"), alignment=TA_CENTER)
h1_style      = S("Heading1",fontSize=14, leading=18, spaceBefore=14,spaceAfter=4, textColor=colors.HexColor("#1a1a2e"), fontName="Helvetica-Bold")
h2_style      = S("Heading2",fontSize=12, leading=16, spaceBefore=10,spaceAfter=3, textColor=colors.HexColor("#2c3e6e"), fontName="Helvetica-Bold")
h3_style      = S("Heading3",fontSize=11, leading=14, spaceBefore=8, spaceAfter=2, textColor=colors.HexColor("#34495e"), fontName="Helvetica-BoldOblique")
body_style    = S("Normal",  fontSize=10, leading=14, spaceAfter=6,  alignment=TA_JUSTIFY)
bullet_style  = S("Normal",  fontSize=10, leading=13, spaceAfter=3,  leftIndent=18, firstLineIndent=-10)
code_style    = S("Code",    fontSize=8,  leading=11, spaceAfter=4,  fontName="Courier", leftIndent=12,
                  backColor=colors.HexColor("#f4f4f4"), textColor=colors.HexColor("#222222"))
ref_style     = S("Normal",  fontSize=9,  leading=13, spaceAfter=4,  alignment=TA_LEFT, leftIndent=18, firstLineIndent=-18)
bold_label    = S("Normal",  fontSize=10, leading=14, fontName="Helvetica-Bold")
note_style    = S("Normal",  fontSize=9,  leading=12, spaceAfter=4,  textColor=colors.HexColor("#555555"),
                  leftIndent=12, fontName="Helvetica-Oblique")

TABLE_HEADER_COLOR = colors.HexColor("#1a1a2e")
TABLE_ROW_COLORS   = [colors.white, colors.HexColor("#f0f4ff")]

# ── Markdown → flowables ──────────────────────────────────────────────────────

def escape(text):
    """Escape XML special chars for ReportLab paragraphs."""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def inline_fmt(text):
    """Convert inline markdown (**bold**, `code`) to RML tags.
    Code spans are protected first to prevent _ inside them being treated as italic."""
    text = escape(text)
    # 1. Extract code spans and replace with placeholders
    placeholders = {}
    def protect_code(m):
        key = f"\x00CODE{len(placeholders)}\x00"
        placeholders[key] = f'<font name="Courier" size="9">{m.group(1)}</font>'
        return key
    text = re.sub(r'`(.+?)`', protect_code, text)
    # 2. Bold and italic (safe now — no underscores inside codes)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # 3. Restore code placeholders
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text

def parse_table(lines):
    """Parse a markdown table into a ReportLab Table."""
    rows = []
    for line in lines:
        if re.match(r'^\s*\|[-| :]+\|\s*$', line):
            continue   # separator row
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        rows.append(cells)
    if not rows:
        return None

    col_count = max(len(r) for r in rows)
    # Pad short rows
    rows = [r + [''] * (col_count - len(r)) for r in rows]

    # Build data with inline formatting
    data = []
    for i, row in enumerate(rows):
        if i == 0:
            data.append([Paragraph(f'<b>{escape(c)}</b>', body_style) for c in row])
        else:
            data.append([Paragraph(inline_fmt(c), body_style) for c in row])

    page_w = A4[0] - 4 * cm
    col_w  = page_w / col_count

    t = Table(data, colWidths=[col_w] * col_count, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0),  TABLE_HEADER_COLOR),
        ('TEXTCOLOR',   (0, 0), (-1, 0),  colors.white),
        ('FONTNAME',    (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), TABLE_ROW_COLORS),
        ('GRID',        (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ('VALIGN',      (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING',  (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0,0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',(0, 0), (-1, -1), 6),
    ]))
    return t


def md_to_flowables(md_text):
    flowables = []
    lines = md_text.splitlines()
    i = 0

    # Title block (first lines before first ---)
    in_title_block = True

    while i < len(lines):
        line = lines[i]

        # ── Horizontal rule ───────────────────────────────────────────────────
        if re.match(r'^---+\s*$', line):
            if in_title_block:
                in_title_block = False
                flowables.append(HRFlowable(width="100%", thickness=1.5,
                                            color=colors.HexColor("#1a1a2e"), spaceAfter=10))
            else:
                flowables.append(HRFlowable(width="100%", thickness=0.5,
                                            color=colors.HexColor("#cccccc"), spaceAfter=6))
            i += 1
            continue

        # ── Headings ──────────────────────────────────────────────────────────
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            level, text = len(m.group(1)), m.group(2)
            # Strip anchor links like {#section}
            text = re.sub(r'\{#[^}]+\}', '', text).strip()
            if level == 1:
                flowables.append(Paragraph(inline_fmt(text), h1_style))
                flowables.append(HRFlowable(width="100%", thickness=0.8,
                                            color=colors.HexColor("#2c3e6e"), spaceAfter=4))
            elif level == 2:
                flowables.append(Paragraph(inline_fmt(text), h2_style))
            else:
                flowables.append(Paragraph(inline_fmt(text), h3_style))
            i += 1
            continue

        # ── Table ─────────────────────────────────────────────────────────────
        if line.startswith('|'):
            tbl_lines = []
            while i < len(lines) and lines[i].startswith('|'):
                tbl_lines.append(lines[i])
                i += 1
            t = parse_table(tbl_lines)
            if t:
                flowables.append(Spacer(1, 4))
                flowables.append(t)
                flowables.append(Spacer(1, 8))
            continue

        # ── Bullet list ───────────────────────────────────────────────────────
        m = re.match(r'^(\s*)[-*]\s+(.*)', line)
        if m:
            indent, text = len(m.group(1)), m.group(2)
            extra_left = indent * 6
            st = ParagraphStyle('bullet_dyn', parent=bullet_style,
                                leftIndent=18 + extra_left, firstLineIndent=-10)
            flowables.append(Paragraph("• " + inline_fmt(text), st))
            i += 1
            continue

        # ── Numbered list ─────────────────────────────────────────────────────
        m = re.match(r'^\d+\.\s+(.*)', line)
        if m:
            flowables.append(Paragraph(inline_fmt(m.group(1)), bullet_style))
            i += 1
            continue

        # ── Code block ────────────────────────────────────────────────────────
        if line.startswith('```'):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(escape(lines[i]))
                i += 1
            i += 1  # skip closing ```
            if code_lines:
                flowables.append(Paragraph('<br/>'.join(code_lines) or ' ', code_style))
            continue

        # ── Bold-only line (table of contents entries, etc.) ──────────────────
        if re.match(r'^\*\*.*\*\*$', line.strip()):
            flowables.append(Paragraph(inline_fmt(line.strip()), bold_label))
            i += 1
            continue

        # ── Title block metadata lines ────────────────────────────────────────
        if in_title_block and line.startswith('#'):
            flowables.append(Paragraph(inline_fmt(line.lstrip('#').strip()), title_style))
            i += 1
            continue
        if in_title_block and line.strip().startswith('**'):
            flowables.append(Paragraph(inline_fmt(line.strip()), subtitle_style))
            i += 1
            continue

        # ── Note / italic-only line ───────────────────────────────────────────
        stripped = line.strip()
        if stripped.startswith('> '):
            flowables.append(Paragraph(inline_fmt(stripped[2:]), note_style))
            i += 1
            continue

        # ── Empty line ────────────────────────────────────────────────────────
        if not stripped:
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # ── Regular paragraph ─────────────────────────────────────────────────
        # Collect continued lines into a paragraph
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith('#') or nxt.startswith('|') or
                    nxt.startswith('```') or nxt.startswith('---') or
                    re.match(r'^[-*]\s', lines[i]) or re.match(r'^\d+\.\s', lines[i])):
                break
            para_lines.append(nxt)
            i += 1
        full = ' '.join(para_lines)
        flowables.append(Paragraph(inline_fmt(full), body_style))

    return flowables


# ── Build PDF ─────────────────────────────────────────────────────────────────

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm,  bottomMargin=2.5*cm,
        title="LiDAR Object Detection and Tracking — DLMDSEAAD02",
        author="Kalpana Abhiseka Maddi",
    )

    with open(INPUT, encoding='utf-8') as f:
        md = f.read()

    # Split Table of Contents section (render as plain text, not deep nesting)
    flowables = md_to_flowables(md)

    # Page numbers
    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawCentredString(A4[0]/2, 1.5*cm,
                                 f"Kalpana Abhiseka Maddi · DLMDSEAAD02 · Page {doc.page}")
        canvas.restoreState()

    doc.build(flowables, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF written to: {OUTPUT}")


if __name__ == '__main__':
    build_pdf()
