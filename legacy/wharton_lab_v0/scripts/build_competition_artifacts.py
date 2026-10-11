"""Build internal reading PDFs, a blocked IPS and an offline evidence viewer."""
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "reports/competition_package"
OUT = ROOT / "output/pdf"


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\[([^]]+)\]\((https?[^)]+)\)", r'<link href="\2" color="#254765">\1</link>', text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def markdown_story(text, *, brief=False):
    body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5 if brief else 9.7,
                          leading=12.6 if brief else 13.4, spaceAfter=6, textColor=colors.HexColor("#202b35"))
    headings = {i: ParagraphStyle(f"h{i}", parent=body, fontName="Helvetica-Bold",
                                  fontSize={1: 22, 2: 12, 3: 10.5}[i],
                                  leading={1: 27, 2: 16, 3: 14}[i], spaceBefore=8,
                                  spaceAfter=6, keepWithNext=True) for i in (1, 2, 3)}
    cell = ParagraphStyle("cell", parent=body, fontSize=7.1, leading=9.2, spaceAfter=0)
    lines = text.splitlines(); story = []; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                items = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", x) for x in items): rows.append(items)
                i += 1
            if len(rows[0]) > 6:
                # Full-resolution table stays in Markdown/CSV; compact PDF selects key columns.
                keep = [0, 1, 2, 4, 5, 6]
                rows = [[row[j] for j in keep] for row in rows]
            table = Table([[Paragraph(inline(x), cell) for x in row] for row in rows],
                          colWidths=[516 / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf1")),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#bcc6cc")),
                                       ("LEFTPADDING", (0, 0), (-1, -1), 5),
                                       ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                                       ("TOPPADDING", (0, 0), (-1, -1), 5),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
            story += [table, Spacer(1, 8)]; continue
        head = re.match(r"(#{1,3}) (.*)", line)
        if head:
            story.append(Paragraph(inline(head[2]), headings[len(head[1])]))
        else:
            parts = [line]
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not lines[i + 1].startswith(("#", "|", "- "))
                   and not re.match(r"^\d+\. ", lines[i + 1])):
                i += 1; parts.append(lines[i].strip())
            story.append(Paragraph(inline(" ".join(parts)), body))
        i += 1
    return story


def footer(canvas, doc):
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#53616d"))
    canvas.drawString(48, 28, "WHARTON | AI-assisted working material | Not approved for submission")
    canvas.drawRightString(564, 28, str(doc.page))


def build_pdfs():
    OUT.mkdir(parents=True, exist_ok=True)
    brief = OUT / "WHARTON_EXECUTIVE_BRIEF.pdf"
    SimpleDocTemplate(str(brief), pagesize=letter, leftMargin=48, rightMargin=48,
                      topMargin=32, bottomMargin=42, title="Wharton - Executive working brief").build(
        markdown_story((PACK / "EXECUTIVE_BRIEF.md").read_text(), brief=True), onFirstPage=footer, onLaterPages=footer)
    assert len(PdfReader(brief).pages) == 1, "Executive brief must be one page"
    files = ["README.md", "WRITTEN_RESPONSES.md", "PITCH_CONTENT.md", "TIMED_PITCH.md",
             "DEMO_SCRIPT.md", "FINANCIAL_STRESS.md", "JUDGE_QA.md", "EVIDENCE_APPENDIX.md"]
    story = []
    for name in files:
        if story: story.append(PageBreak())
        story += markdown_story((PACK / name).read_text())
    pack = OUT / "WHARTON_PREPARATION_PACK.pdf"
    SimpleDocTemplate(str(pack), pagesize=letter, leftMargin=48, rightMargin=48,
                      topMargin=40, bottomMargin=44, title="Wharton - Internal preparation pack").build(story, onFirstPage=footer, onLaterPages=footer)

    fonts = Path("/System/Library/Fonts/Supplemental")
    for font, file in [("TimesNewRoman", "Times New Roman.ttf"), ("TimesNewRomanBold", "Times New Roman Bold.ttf")]:
        if not (fonts / file).exists():
            raise FileNotFoundError("True Times New Roman required; set fonts path to licensed installed fonts")
        pdfmetrics.registerFont(TTFont(font, str(fonts / file)))
    base = ParagraphStyle("ips", fontName="TimesNewRoman", fontSize=12, leading=24, spaceAfter=8)
    bold = ParagraphStyle("ipshead", parent=base, fontName="TimesNewRomanBold", keepWithNext=True)
    ips_text = (PACK / "IPS_DRAFT.md").read_text()
    pitch, statement = ips_text.split("## Investment Strategy Elevator Pitch\n", 1)[1].split("## Investment Policy Statement\n", 1)
    pitch, statement = pitch.strip(), statement.strip()
    words = {"pitch": len(pitch.split()), "ips": len(statement.split())}
    assert words["pitch"] <= 50 and words["ips"] <= 500
    ips = OUT / "WHARTON_IPS_DRAFT_BLOCKED.pdf"
    story = [Paragraph("INVESTMENT POLICY STATEMENT", bold),
             Paragraph("DRAFT - BLOCKED: complete client case and student review required", base),
             Paragraph("Official team name: NOT PROVIDED", base),
             Paragraph("Finalized student names (first name, last initial): NOT PROVIDED", base),
             Paragraph("WInS username: NOT PROVIDED", base),
             Paragraph("AI-assisted working draft. Not approved for submission.", base), PageBreak(),
             Paragraph("Investment Strategy Elevator Pitch", bold), Paragraph(inline(pitch), base),
             Paragraph("Investment Policy Statement", bold)]
    story += [Paragraph(inline(p), base) for p in statement.split("\n\n")]
    SimpleDocTemplate(str(ips), pagesize=letter, leftMargin=72, rightMargin=72,
                      topMargin=72, bottomMargin=72, title="IPS DRAFT - BLOCKED").build(story)
    reader = PdfReader(ips)
    assert len(reader.pages) <= 3
    assert ips.stat().st_size <= 5_000_000
    for p in reader.pages:
        assert not p.get("/Annots"), "IPS must not contain external links"
        assert not p.images, "IPS must not contain images"
    return {"pdfs": {str(p.relative_to(ROOT)): {"pages": len(PdfReader(p).pages), "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [brief, pack, ips]},
            "ips_word_counts": words, "ips_font": "embedded Times New Roman, 12 pt, 24 pt leading",
            "ips_margins_inches": 1, "submission_ready": False}


def build_html():
    source = ROOT / "runs/competition_stress/RESULTS.json"
    data = json.loads(source.read_text())
    # Closed local content only. No user-provided HTML or remote resources.
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wharton | Decision room</title><style>
body{max-width:1050px;margin:40px auto;padding:0 22px;font:16px/1.55 system-ui;color:#1b2d3b;background:#faf9f5}h1{font:48px/1.1 Georgia}h2{font:25px Georgia}.eyebrow{letter-spacing:.15em;font-size:12px}.notice{border-left:4px solid #a04f25;padding:12px 18px;background:#f2e9dc}.controls{display:flex;gap:18px;flex-wrap:wrap;margin:25px 0}label{display:grid;gap:6px;flex:1}select{font:inherit;padding:10px;border:1px solid #aeb7b9;background:white;max-width:100%}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;padding:22px 0;border-top:1px solid;border-bottom:1px solid}.metric strong{display:block;font-size:25px}.metric span{font-size:13px}svg{width:100%;background:white;border:1px solid #ddd}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:right;padding:7px;border-bottom:1px solid #ccc}th:first-child,td:first-child{text-align:left}code{overflow-wrap:anywhere;font-size:12px}.small{font-size:13px;color:#50616d}a{color:#254765}</style>
<p class="eyebrow">WHARTON / RESEARCH DECISION SUPPORT</p><h1>Evidence before allocation.</h1>
<p class="notice"><strong>Assumptions only. No client recommendation or executed trades.</strong><br>Initial wealth = 1. These are hypothetical paths, not historical returns or estimated probabilities. Full client case and student review remain pending.</p>
<div class="controls"><label>Assumed path<select id="scenario"></select></label><label>Policy<select id="policy"></select></label><label>Trading cost, bps<select id="cost"></select></label></div>
<div class="metrics" id="metrics" aria-live="polite"></div><h2>Wealth and unpaid obligations</h2><p class="small">Navy: remaining wealth. Rust: cumulative unpaid obligations. Both use units of initial wealth. All 120 monthly observations are shown.</p>
<svg id="chart" viewBox="0 0 960 320" role="img" aria-label="Wealth and unpaid obligations by month"></svg>
<h2>Selected assumptions</h2><p id="assumptions"></p><p class="small">Two hypothetical risky sleeves, equal risky weights. Reserve cash earns zero. No tax, contributions or borrowing. Reserve planning inflation stays at 2.5%; it does not know future returns. The calculator does not run EWMA, M11 or M12.</p>
<h2>Ledger checkpoints</h2><table><thead><tr><th>Month</th><th>Start</th><th>Paid</th><th>Unpaid</th><th>Fees</th><th>End</th></tr></thead><tbody id="ledger"></tbody></table>
<h2>Trace the evidence</h2><p>Config: <code>configs/competition_stress.json</code><br>Source: <code>runs/competition_stress/RESULTS.json</code></p><p class="small">Result SHA-256: <code>__HASH__</code></p><p class="small">Regenerate from the config, then rebuild this file. The selectors filter retained calculations; they never create live market estimates. Review <a href="EVIDENCE_APPENDIX.md">the evidence appendix</a> and <a href="FINANCIAL_STRESS.md">the full sensitivity table</a>.</p>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const byId=id=>document.getElementById(id),fmt=x=>Number(x).toFixed(4);
for(const [id,items] of [['scenario',data.config.scenarios.map(s=>s.id)],['policy',data.config.policies],['cost',data.config.cost_bps]]){
 for(const item of items){const o=document.createElement('option');o.value=item;o.textContent=String(item).replaceAll('_',' ');byId(id).append(o)}
 byId(id).addEventListener('change',render);
}
byId('cost').value='10';byId('policy').value='liability_reserve';
function render(){
 const r=data.results.find(x=>x.scenario===byId('scenario').value&&x.policy===byId('policy').value&&x.cost_bps===Number(byId('cost').value));
 byId('metrics').replaceChildren();
 for(const [label,value] of [['End wealth',fmt(r.terminal_wealth)],['Paid / due',(100*r.funded_fraction).toFixed(1)+'%'],['Unpaid balance',fmt(r.unpaid_at_end)],['First missed payment',r.first_shortfall_month===null?'None':'Month '+r.first_shortfall_month],['Net drawdown',(100*r.max_drawdown_ex_withdrawals).toFixed(1)+'%'],['Total fees',fmt(r.total_fees)]]){const d=document.createElement('div');d.className='metric';const strong=document.createElement('strong');strong.textContent=value;const span=document.createElement('span');span.textContent=label;d.append(strong,span);byId('metrics').append(d)}
 const ys=[1,...r.trajectory.flatMap(x=>[x.wealth_end,x.arrears])],max=Math.max(...ys)*1.08;
 const coords=(key)=>[[0,key==='wealth_end'?1:0],...r.trajectory.map(x=>[x.month,x[key]])].map(([x,y])=>(55+x/120*875)+','+(278-y/max*240)).join(' ');
 let drawing='<line x1="55" y1="278" x2="930" y2="278" stroke="#999"/>';
 for(let i=0;i<=4;i++){const y=278-i*60;drawing+='<line x1="55" y1="'+y+'" x2="930" y2="'+y+'" stroke="#eee"/><text x="8" y="'+(y+4)+'" font-size="12">'+(max*i/4).toFixed(2)+'</text>'}
 drawing+='<polyline fill="none" stroke="#183d54" stroke-width="3" points="'+coords('wealth_end')+'"/><polyline fill="none" stroke="#ab532f" stroke-width="3" points="'+coords('arrears')+'"/>';
 for(const m of [0,24,48,72,96,120])drawing+='<text x="'+(55+m/120*875)+'" y="304" font-size="12" text-anchor="middle">'+m+'</text>';
 byId('chart').innerHTML=drawing;
 const s=data.config.scenarios.find(x=>x.id===r.scenario);
 byId('assumptions').textContent='Annual sleeve returns: '+s.annual_asset_returns.map(x=>(100*x).toFixed(1)+'%').join(' / ')+'. Inflation: '+(100*s.annual_inflation).toFixed(1)+'%. Base monthly obligation '+data.config.monthly_obligation+' × '+s.obligation_multiplier+'. Shocks: '+JSON.stringify(s.shocks||[])+'. End wealth in initial purchasing-power units: '+fmt(r.terminal_real_wealth)+'.';
 byId('ledger').innerHTML=r.trajectory.filter(x=>x.month===1||x.month%12===0||x.month===r.first_shortfall_month).map(x=>'<tr>'+[x.month,...['wealth_start','paid','arrears','fees','wealth_end'].map(k=>fmt(x[k]))].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('');
}
render();</script></html>'''
    template = template.replace("__HASH__", hashlib.sha256(source.read_bytes()).hexdigest()).replace("__DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c"))
    (PACK / "DECISION_ROOM.html").write_text(template)


def main():
    build_html()
    receipt = build_pdfs()
    source_files = sorted(PACK.glob("*.md"))
    receipt["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (ROOT / "audit/astra_2026-09-27/ARTIFACT_QA.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["pdfs"], indent=2)); print(receipt["ips_word_counts"])


if __name__ == "__main__":
    main()
