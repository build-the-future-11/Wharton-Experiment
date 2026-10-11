"""Render reports/FINAL_RESEARCH_REPORT.pdf (from its .md) and the BLOCKED IPS draft PDF."""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "reports" / "FINAL_RESEARCH_REPORT.md"
REPORT_PDF = ROOT / "reports" / "FINAL_RESEARCH_REPORT.pdf"
IPS_PDF = ROOT / "reports" / "competition_drafts" / "IPS_DRAFT_BLOCKED.pdf"

PITCH_LIMIT = 50
IPS_LIMIT = 500

PITCH = (
    "DRAFT BLOCKED: Reserve cash for obligations, hold a strategic allocation, and size risk "
    "with EWMA covariance. Our tests found no reliable return-forecasting edge, so we do not "
    "trade on forecasts or promise multi-year funding from them."
)

IPS_PARAGRAPHS = [
    "STATUS: DRAFT — BLOCKED_MISSING_SOURCE. This document is not a competition submission. "
    "Client wealth, payment dates, operating obligations, contribution amounts, roster details, "
    "and eligible instruments were not available in the research workspace. No values were "
    "invented.",
    "Investment objective (research stance): Protect operating reliability first, then preserve "
    "flexibility for a responsible facility contribution when client facts are supplied. The "
    "portfolio is a strategic allocation sized with EWMA covariance. Leakage-free walk-forward "
    "tests found no return forecast that beat the historical mean, so the strategy does not "
    "rely on forecasts or on complex machine-learning architectures for client decisions.",
    "Liquidity and reserves: Maintain an explicit operating-reserve rule sized to near-term "
    "obligations once those schedules are provided. Do not fund operating payments from "
    "unverified short-horizon forecast alpha. Transaction-cost research scenarios of "
    "0/5/10/25/50 basis points per dollar traded (one-way) were registered with 10 as the base "
    "research assumption—not a verified trading-cost estimate.",
    "Risk: Prefer transparent risk forecasts (EWMA/shrinkage). Rebalance on a schedule rather "
    "than on predicted returns. Scenario and controller experiments use research-proxy "
    "liabilities only and are not real-world funding guarantees.",
    "Governance: Technical model identities remain in the internal research report. This IPS "
    "does not list twelve architectures. Trading Notes Analysis remains PROPOSED until three "
    "verified WInS executions are supplied. No login, trade, email, or submission was performed.",
]

STRATEGY_NOTES = (
    "Selected research components: strategic allocation; B_EWMA_COV risk sizing. Return "
    "forecast: none selected — ridge showed no skill over the historical mean in leakage-free "
    "walk-forward tests. The earlier synthetic lockbox edge was a data-leakage artefact and is "
    "withdrawn; the ETF lockbox result was negative. Controller/funding results remain "
    "research-proxy until liabilities arrive."
)


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    return text


def build_report_pdf() -> Path:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=12.5)
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10)
    heads = {1: styles["Title"], 2: styles["Heading2"], 3: styles["Heading3"]}
    story: list = []
    lines = REPORT_MD.read_text().splitlines()
    i = 0
    para: list[str] = []

    def flush() -> None:
        if para:
            story.append(Paragraph(_inline(" ".join(para)), body))
            story.append(Spacer(1, 4))
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            story.append(Preformatted("\n".join(lines[i + 1 : j]), styles["Code"]))
            i = j + 1
            continue
        if line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append([Paragraph(_inline(c), cell) for c in cells])
                i += 1
            t = Table(rows, repeatRows=1, hAlign="LEFT")
            t.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story += [t, Spacer(1, 6)]
            continue
        m_img = re.match(r"!\[.*\]\((.+)\)", line)
        m_head = re.match(r"(#{1,3}) (.*)", line)
        m_item = re.match(r"(\s*)(- |\d+\. )(.*)", line)
        if m_img:
            flush()
            img = Image(str(REPORT_MD.parent / m_img.group(1)))
            scale = 6.5 * inch / img.drawWidth
            img.drawWidth, img.drawHeight = img.drawWidth * scale, img.drawHeight * scale
            story += [img, Spacer(1, 4)]
        elif m_head:
            flush()
            story.append(Paragraph(_inline(m_head.group(2)), heads[len(m_head.group(1))]))
        elif m_item:
            flush()
            bullet = "•" if m_item.group(2) == "- " else m_item.group(2).strip()
            story.append(Paragraph(_inline(m_item.group(3)), body, bulletText=bullet))
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
        i += 1
    flush()
    doc = SimpleDocTemplate(
        str(REPORT_PDF), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch, title="Final Research Report (internal)",
    )
    doc.build(story)
    return REPORT_PDF


def build_ips_pdf() -> Path:
    pitch_words = _words(PITCH)
    ips_words = sum(_words(p) for p in IPS_PARAGRAPHS)
    if pitch_words > PITCH_LIMIT or ips_words > IPS_LIMIT:
        raise ValueError(f"word limits exceeded: pitch={pitch_words}, ips={ips_words}")
    base = ParagraphStyle("tnr", fontName="Times-Roman", fontSize=12, leading=24)
    bold = ParagraphStyle("tnrb", parent=base, fontName="Times-Bold")
    story = [
        Paragraph("TITLE PAGE (DRAFT — BLOCKED)", bold),
        Paragraph("Wharton Global Youth Investment Challenge — Investment Policy Statement", base),
        Paragraph("Team / roster: BLOCKED_MISSING_SOURCE", base),
        Paragraph("Date: 2026-09-27 (revised)", base),
        Paragraph("Authority: Missing 2026_WGY source PDFs and client facts.", base),
        PageBreak(),
        Paragraph(f"ELEVATOR PITCH (maximum {PITCH_LIMIT} words)", bold),
        Paragraph(PITCH, base),
        Paragraph(f"Word count: {pitch_words}", base),
        Paragraph(f"INVESTMENT POLICY STATEMENT (maximum {IPS_LIMIT} words)", bold),
        *[Paragraph(p, base) for p in IPS_PARAGRAPHS],
        Paragraph(f"Word count: {ips_words}", base),
        PageBreak(),
        Paragraph("STRATEGY NOTES (maximum 2 pages — research summary only)", bold),
        Paragraph(STRATEGY_NOTES, base),
    ]
    doc = SimpleDocTemplate(
        str(IPS_PDF), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch,
        bottomMargin=inch, title="IPS DRAFT — BLOCKED (not a submission)",
    )
    doc.build(story)
    return IPS_PDF


def main() -> int:
    print(f"Wrote {build_report_pdf()}")
    print("Historical IPS builder superseded. Run scripts/build_competition_artifacts.py for the current blocked draft.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
