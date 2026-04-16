"""Build an Axion Deep Labs-branded PDF of the Phase I-B G1 metric audit memo.

Reads the Markdown memo, emits a fully-self-contained styled HTML (inline
logo, inline CSS, Inter font from Google Fonts), then renders via headless
Chrome to PDF. Output goes alongside the .md file.
"""
from __future__ import annotations

import base64
import pathlib
import re
import subprocess
import sys
import textwrap
from html import escape

try:
    from markdown_it import MarkdownIt
except ImportError:
    print("markdown-it-py is required (pip install markdown-it-py).", file=sys.stderr)
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MEMO_MD = ROOT / "drafts" / "phase_1b_g1_metric_audit_memo.md"
OUT_HTML = ROOT / "drafts" / "phase_1b_g1_metric_audit_memo.html"
OUT_PDF = ROOT / "drafts" / "phase_1b_g1_metric_audit_memo.pdf"
LOGO_PATH = pathlib.Path(
    "/home/joshua/Corporate/AxionDeep/images/marketing-logos/logo-transparent.png"
)


# ─── Brand tokens ──────────────────────────────────────────────────────────
CYAN = "#06B6D4"
VIOLET = "#8B5CF6"
FUCHSIA = "#EC4899"
INK = "#111827"          # Gray 900, primary text
INK_2 = "#374151"        # Gray 700, body
MUTED = "#6B7280"        # Gray 500, captions
SUBTLE = "#9CA3AF"       # Gray 400
RULE = "#E5E7EB"         # Gray 200, hairlines
PAPER = "#FFFFFF"
TINT = "#F9FAFB"         # Gray 50 panel
AMBER = "#D97706"        # for warning callouts
AMBER_BG = "#FFFBEB"
CYAN_BG = "#ECFEFF"
VIOLET_BG = "#F5F3FF"


def logo_data_uri() -> str:
    data = LOGO_PATH.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


# ─── Markdown parsing ──────────────────────────────────────────────────────
def parse_memo(md_text: str) -> tuple[dict, str]:
    """Split off the front-matter block and return (meta, remaining markdown).

    The memo uses a series of **Key:** value lines at the top; we extract them
    for a structured cover-page card.
    """
    lines = md_text.splitlines()
    # Title is the first H1
    title = ""
    body_start = 0
    for i, ln in enumerate(lines):
        if ln.startswith("# "):
            title = ln[2:].strip()
            body_start = i + 1
            break
    # Meta lines: **Key:** value, until a blank or horizontal rule
    meta = {}
    i = body_start
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        if ln == "---":
            break
        m = re.match(r"\*\*([^:]+):\*\*\s*(.*)", ln)
        if m:
            meta[m.group(1).strip()] = m.group(2).strip()
            i += 1
            continue
        break
    # Remaining body starts after the first horizontal rule following meta
    while i < len(lines) and lines[i].strip() != "---":
        i += 1
    if i < len(lines) and lines[i].strip() == "---":
        i += 1
    body_md = "\n".join(lines[i:]).strip()
    return {"title": title, **meta}, body_md


def md_to_html(body_md: str) -> str:
    """Render markdown to HTML with specific post-processing for math blocks."""
    md = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")
    html = md.render(body_md)

    # Promote indented plaintext math blocks (rendered by markdown-it as <pre><code>)
    # that contain math-like characters into our .math-block class.
    def reclass_pre(match: re.Match) -> str:
        inner = match.group(1)
        decoded = (
            inner.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
        )
        # Heuristic: if it doesn't look like shell/code, it's math.
        is_shell = bool(
            re.search(r"(?m)^\s*(find|grep|ls|sbatch|python|pip|rsync|sacct|squeue)\b", decoded)
            or re.search(r"(?m)^\s*\$", decoded)
            or "()" in decoded and "def " in decoded
        )
        if is_shell:
            return f'<pre class="code-block"><code>{inner}</code></pre>'
        if any(s in decoded for s in ("argmax", "Σ", "θ", "φ", "𝟙", "ℝ", "∈", "AURC", "EWC_benefit", "CrossEntropy")):
            return f'<div class="math-block">{inner}</div>'
        return f'<pre class="code-block"><code>{inner}</code></pre>'

    html = re.sub(
        r"<pre><code[^>]*>([\s\S]*?)</code></pre>",
        reclass_pre,
        html,
    )

    # Style section headers (H2) with an aurora rule; H3 gets a lighter treatment.
    # Also tag specific sections for color-coded callouts via wrapper classes.
    return html


# ─── HTML assembly ─────────────────────────────────────────────────────────
def build_html(meta: dict, body_html: str) -> str:
    logo = logo_data_uri()
    title = meta.get("title", "Axion Deep Labs Memo")

    # Extract the TL;DR and Recommendation bullets from the body for the cover card.
    # (Light touch — we keep the full body rendering below.)

    meta_rows = ""
    for key in ("For", "From", "Date drafted", "Status", "Scope"):
        if key in meta:
            meta_rows += (
                f'<div class="meta-row">'
                f'<span class="meta-key">{escape(key)}</span>'
                f'<span class="meta-val">{escape(meta[key])}</span>'
                f"</div>"
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
{CSS}
</style>
</head>
<body>

<!-- ─── COVER PAGE ─────────────────────────────────────────────────── -->
<section class="cover">
  <div class="cover-top">
    <img class="cover-logo" src="{logo}" alt="Axion Deep Labs">
    <div class="cover-brand">AXION&nbsp;DEEP&nbsp;LABS</div>
    <div class="cover-tag">Advancing Intelligence Through Research &amp; Innovation</div>
  </div>

  <div class="aurora-bar"></div>

  <div class="cover-main">
    <div class="cover-kicker">Research Memo · Phase I-B · PERSIST</div>
    <h1 class="cover-title">Phase I-B G1 Metric Audit</h1>
    <div class="cover-subtitle">Decision Memo &mdash; Cross-Dataset Retention Under Recency Bias</div>

    <div class="cover-card">
      {meta_rows}
    </div>

    <div class="cover-note">
      <div class="cover-note-label">STATE</div>
      <p>Phase 4 analysis on Phase I-B is <strong>frozen</strong> pending this memo's outcome.
      Do not replace the existing metric. Parallel-metric robustness check only;
      feasibility gated on checkpoint availability and a step-0 sanity gate.</p>
    </div>
  </div>

  <div class="cover-footer">
    <span>Axion Deep Labs, Inc.</span>
    <span>axiondeep.com</span>
    <span>Confidential · Internal</span>
  </div>
</section>

<!-- ─── BODY ──────────────────────────────────────────────────────── -->
<section class="body">
  {body_html}
</section>

</body>
</html>
"""


CSS = f"""
/* ═══ Page setup ═════════════════════════════════════════════════════ */
@page {{
  size: letter;
  margin: 0.6in 0.7in 0.75in 0.7in;
  @bottom-left {{
    content: "Axion Deep Labs  ·  Phase I-B G1 Metric Audit";
    font-family: 'Inter', sans-serif;
    font-size: 8.5pt;
    color: {MUTED};
    letter-spacing: 0.02em;
  }}
  @bottom-right {{
    content: "Page " counter(page) " of " counter(pages);
    font-family: 'Inter', sans-serif;
    font-size: 8.5pt;
    color: {MUTED};
  }}
}}
@page :first {{
  margin: 0;
  @bottom-left {{ content: none; }}
  @bottom-right {{ content: none; }}
}}

* {{ box-sizing: border-box; }}

html, body {{
  margin: 0;
  padding: 0;
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 10.5pt;
  line-height: 1.65;
  color: {INK_2};
  background: {PAPER};
  -webkit-font-smoothing: antialiased;
}}

/* ═══ COVER PAGE ═════════════════════════════════════════════════════ */
.cover {{
  height: 100vh;
  min-height: 1040px;
  padding: 0.9in 0.8in 0.6in 0.8in;
  display: flex;
  flex-direction: column;
  page-break-after: always;
  position: relative;
  background: linear-gradient(180deg, #FFFFFF 0%, #FAFBFC 60%, #F4F6F9 100%);
}}

.cover-top {{
  text-align: center;
  margin-top: 0.4in;
}}

.cover-logo {{
  width: 110px;
  height: auto;
  margin-bottom: 18px;
}}

.cover-brand {{
  font-weight: 800;
  font-size: 22pt;
  letter-spacing: 0.14em;
  color: {INK};
}}

.cover-tag {{
  margin-top: 8px;
  font-size: 9.5pt;
  color: {MUTED};
  letter-spacing: 0.04em;
}}

.aurora-bar {{
  height: 6px;
  margin: 40px auto 0 auto;
  width: 60%;
  border-radius: 3px;
  background: linear-gradient(90deg, {CYAN} 0%, {VIOLET} 50%, {FUCHSIA} 100%);
}}

.cover-main {{
  flex: 1;
  padding-top: 60px;
}}

.cover-kicker {{
  font-size: 9pt;
  font-weight: 600;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: {VIOLET};
  margin-bottom: 14px;
}}

.cover-title {{
  font-size: 34pt;
  line-height: 1.1;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: {INK};
  margin: 0 0 10px 0;
}}

.cover-subtitle {{
  font-size: 13pt;
  font-weight: 500;
  color: {MUTED};
  margin-bottom: 32px;
}}

.cover-card {{
  background: {PAPER};
  border: 1px solid {RULE};
  border-radius: 10px;
  padding: 20px 24px;
  margin-bottom: 24px;
  box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
}}

.meta-row {{
  display: flex;
  padding: 6px 0;
  border-bottom: 1px dashed {RULE};
  font-size: 10pt;
}}
.meta-row:last-child {{ border-bottom: none; }}

.meta-key {{
  width: 120px;
  font-weight: 600;
  color: {INK};
  letter-spacing: 0.02em;
}}

.meta-val {{
  flex: 1;
  color: {INK_2};
}}

.cover-note {{
  background: {CYAN_BG};
  border-left: 4px solid {CYAN};
  border-radius: 8px;
  padding: 14px 18px;
  margin-top: 20px;
}}

.cover-note-label {{
  font-size: 8.5pt;
  font-weight: 700;
  letter-spacing: 0.22em;
  color: {CYAN};
  margin-bottom: 6px;
}}

.cover-note p {{
  margin: 0;
  font-size: 10pt;
  color: {INK_2};
  line-height: 1.55;
}}

.cover-footer {{
  display: flex;
  justify-content: space-between;
  font-size: 8.5pt;
  color: {MUTED};
  letter-spacing: 0.04em;
  padding-top: 20px;
  border-top: 1px solid {RULE};
}}

/* ═══ BODY ═══════════════════════════════════════════════════════════ */
.body {{ max-width: 6.8in; margin: 0 auto; }}

.body h2 {{
  font-size: 17pt;
  font-weight: 700;
  color: {INK};
  letter-spacing: -0.01em;
  margin: 28px 0 12px 0;
  padding-bottom: 10px;
  position: relative;
  page-break-after: avoid;
}}

.body h2::after {{
  content: "";
  position: absolute;
  left: 0;
  bottom: 0;
  width: 80px;
  height: 3px;
  border-radius: 2px;
  background: linear-gradient(90deg, {CYAN} 0%, {VIOLET} 70%, {FUCHSIA} 100%);
}}

.body h3 {{
  font-size: 12.5pt;
  font-weight: 600;
  color: {INK};
  margin: 20px 0 8px 0;
  page-break-after: avoid;
}}

.body p {{
  margin: 0 0 10px 0;
  hyphens: auto;
}}

.body strong {{ color: {INK}; font-weight: 600; }}

.body ul, .body ol {{
  margin: 6px 0 12px 0;
  padding-left: 22px;
}}
.body li {{ margin-bottom: 4px; }}

/* ═══ Math and code blocks ══════════════════════════════════════════ */
.math-block {{
  font-family: 'JetBrains Mono', 'Menlo', 'Consolas', monospace;
  font-size: 9.5pt;
  line-height: 1.5;
  background: {TINT};
  border-left: 3px solid {VIOLET};
  padding: 12px 16px;
  margin: 10px 0 14px 0;
  border-radius: 0 6px 6px 0;
  white-space: pre-wrap;
  color: {INK};
  page-break-inside: avoid;
}}

.code-block {{
  font-family: 'JetBrains Mono', 'Menlo', 'Consolas', monospace;
  font-size: 9pt;
  line-height: 1.5;
  background: {TINT};
  border: 1px solid {RULE};
  padding: 10px 14px;
  margin: 10px 0;
  border-radius: 6px;
  white-space: pre-wrap;
  overflow-wrap: break-word;
  color: {INK};
  page-break-inside: avoid;
}}

p code, li code, td code {{
  font-family: 'JetBrains Mono', 'Menlo', 'Consolas', monospace;
  font-size: 90%;
  background: {TINT};
  border: 1px solid {RULE};
  padding: 1px 5px;
  border-radius: 4px;
  color: {INK};
  white-space: nowrap;
}}

/* ═══ Tables ═════════════════════════════════════════════════════════ */
.body table {{
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0 16px 0;
  font-size: 9pt;
  page-break-inside: auto;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid {RULE};
  table-layout: fixed;
}}

.body thead tr {{
  background: {INK};
  color: {PAPER};
}}

.body th {{
  padding: 8px 10px;
  text-align: left;
  font-weight: 600;
  letter-spacing: 0.02em;
  font-size: 9pt;
  border: none;
  color: {PAPER};
}}

.body td {{
  padding: 8px 10px;
  border-top: 1px solid {RULE};
  vertical-align: top;
  overflow-wrap: anywhere;
  word-break: break-word;
}}

.body tbody tr:nth-child(even) {{ background: {TINT}; }}

/* Inside tables, inline code must wrap or it will push columns past the
   page edge (observed in the Artifacts table with long file paths). */
.body td code, .body th code {{
  white-space: normal;
  overflow-wrap: anywhere;
  word-break: break-word;
  font-size: 85%;
}}

/* ═══ Blockquote / callouts (styled from markdown > blocks if any) ═══ */
.body blockquote {{
  margin: 12px 0;
  padding: 10px 16px;
  background: {VIOLET_BG};
  border-left: 4px solid {VIOLET};
  border-radius: 0 6px 6px 0;
  color: {INK_2};
}}

/* ═══ Horizontal rules become subtle aurora tints ═══════════════════ */
.body hr {{
  border: none;
  height: 2px;
  margin: 22px 0;
  background: linear-gradient(90deg, transparent 0%, {RULE} 15%, {RULE} 85%, transparent 100%);
}}

/* ═══ Nicer heading for the first section (TL;DR) ═══════════════════ */
.body h2:first-of-type {{ margin-top: 4px; }}

/* Page-break hints. Tables are allowed to split — the artifacts table is
   too tall to fit on one page and forcing page-break-inside: avoid leaves
   large blank areas where Chrome pushes the whole table to a new page. */
.body h2 {{ page-break-before: auto; }}
.body .math-block, .body .code-block {{ page-break-inside: avoid; }}
.body tr {{ page-break-inside: avoid; }}
"""


def main() -> None:
    md_text = MEMO_MD.read_text(encoding="utf-8")
    meta, body_md = parse_memo(md_text)
    body_html = md_to_html(body_md)
    html = build_html(meta, body_html)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_HTML.relative_to(ROOT)}")

    cmd = [
        "google-chrome",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUT_PDF}",
        "--print-to-pdf-no-header",
        "--virtual-time-budget=10000",
        f"file://{OUT_HTML}",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Chrome stderr:\n" + result.stderr, file=sys.stderr)
        sys.exit(result.returncode)
    print(f"Wrote {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
