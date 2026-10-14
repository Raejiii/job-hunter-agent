"""Build the project report PDF from report.html.

    python report/build_report.py

Renders the front matter (no header/footer) and the body (running header, page numbers)
separately with Chrome via Playwright, fills the table of contents with real page numbers
read back from the rendered body, and merges both into one PDF.
"""

import html
import io
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

HERE = Path(__file__).parent
OUT = HERE.parent / "Rahul Joy Suneel - langchain_project report.pdf"
TITLE = "Job Hunter: A Multi-Agent Career Copilot Using LangChain, LangGraph and Ollama"

HEADER = f"""<div style="width:100%; font: 9px 'Times New Roman', serif; color:#444; margin: 0 22mm 0 25mm;
  border-bottom: 0.5px solid #999; padding-bottom: 3px;">{TITLE}</div>"""
FOOTER = """<div style="width:100%; font: 9.5px 'Times New Roman', serif; color:#444; margin: 0 22mm 0 25mm;
  display:flex; justify-content:space-between; border-top: 0.5px solid #999; padding-top: 3px;">
  <span>Dept. of CSE, DSCE 2026-2027</span><span class="pageNumber"></span></div>"""


def matcher_table() -> str:
    results = json.loads((HERE / "results.json").read_text())["matcher"]
    tick = lambda ok: "✔" if ok else "✘"  # noqa: E731
    rows = "".join(
        f"<tr><td>{html.escape(r['job'])}</td><td>{html.escape(r['requirement'])}</td>"
        f"<td class='center'>{'Present' if r['truth'] else 'Absent'}</td>"
        f"<td class='center'>{tick(r['naive'] == r['truth'])}</td><td class='center'>{tick(r['hybrid'] == r['truth'])}</td></tr>"
        for r in results["rows"])
    n = len(results["rows"])
    return (
        "<div style='break-inside: avoid'><table><tr><th>Posting</th><th>Requirement (verbatim)</th><th>Ground truth</th>"
        "<th>Naive correct</th><th>Hybrid correct</th></tr>" + rows +
        f"<tr><th colspan='3'>Accuracy</th><th class='center'>{results['naive_accuracy'] * n:.0f}/{n} "
        f"({results['naive_accuracy']:.0%})</th><th class='center'>{results['hybrid_accuracy'] * n:.0f}/{n} "
        f"({results['hybrid_accuracy']:.0%})</th></tr></table>"
        "<p class='center'>Table 6.2: Naive keyword matching vs. the hybrid skill matcher</p></div>")


def run_table() -> str:
    runs = json.loads((HERE / "runs.json").read_text())
    rows = "".join(
        f"<tr><td>{r['job']}</td><td class='center'>{r['score']}/100 ({r['band']})</td>"
        f"<td class='center'>{r['must_haves']}</td><td>{r['missing']}</td><td class='center'>{r['letter']}</td>"
        f"<td class='center'>{r['time']}</td></tr>" for r in runs)
    return (
        "<p><b>End-to-end runs.</b> Table 6.3 summarises the two recorded runs with the real model (the runs shown in "
        "Figures 6.3–6.9).</p>"
        "<div style='break-inside: avoid'><table><tr><th>Posting</th><th>Fit score</th><th>Must-haves met</th><th>Missing must-haves</th>"
        "<th>Final letter</th><th>Time</th></tr>" + rows + "</table>"
        "<p class='center'>Table 6.3: End-to-end results on Llama 3.2 3B (Apple M1, 8 GB)</p></div>")


def toc_html(entries: list[tuple[int, str, int]]) -> str:
    cls = {1: "ch", 2: "sub", 3: "subsub"}
    rows = "".join(f"<tr class='{cls[level]}'><td><div class='dots'><span>{html.escape(text)}</span></div></td>"
                   f"<td class='n'>{page}</td></tr>" for level, text, page in entries)
    return f"<table class='toc'>{rows}</table>"


def toc_entries(source: str) -> list[tuple[int, str]]:
    entries = []
    for m in re.finditer(r"<h([123])[^>]*data-toc=\"([^\"]+)\"(?:[^>]*data-level=\"(\d)\")?", source):
        tag, text, level = m.groups()
        entries.append((int(level or {"1": 1, "2": 2, "3": 3}[tag]), html.unescape(text)))
    return entries


def find_pages(pdf_bytes: bytes, entries: list[tuple[int, str]]) -> list[tuple[int, str, int]]:
    pages = [re.sub(r"\s+", " ", p.extract_text() or "") for p in PdfReader(io.BytesIO(pdf_bytes)).pages]
    out, start = [], 0
    for level, text in entries:
        # Search forward from the previous hit. Section numbers stay in the needle ("3.4 LangChain") because
        # the running header repeats words like "LangChain" on every page; chapter titles drop "1. ".
        needle = re.sub(r"^\d+\.\s+", "", text) if level == 1 else text
        needle_lc = needle.lower()[:40]
        for i in range(start, len(pages)):
            if needle_lc in pages[i].lower():
                start = i
                break
        out.append((level, text, start + 1))
    return out


def render(page, source: str, body: bool) -> bytes:
    (HERE / "_render.html").write_text(source)
    page.goto((HERE / "_render.html").as_uri(), wait_until="networkidle")
    opts = dict(format="A4", print_background=True, prefer_css_page_size=True)
    if body:
        opts |= dict(display_header_footer=True, header_template=HEADER, footer_template=FOOTER)
    return page.pdf(**opts)


def main() -> None:
    source = (HERE / "report.html").read_text()
    source = source.replace("<!--MATCHER_TABLE-->", matcher_table()).replace("<!--RUN_TABLE-->", run_table())
    head, rest = source.split('<div id="front">', 1)
    front, body = rest.split('<div id="body">', 1)
    front_doc = head + '<div id="front">' + front.rsplit("</div>", 1)[0] + "</div></body></html>"
    body_doc = head + '<div id="body">' + body

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        body_pdf = render(page, body_doc, body=True)
        entries = find_pages(body_pdf, toc_entries(body_doc))
        front_pdf = render(page, front_doc.replace("<!--TOC-->", toc_html(entries)), body=False)
        browser.close()
    (HERE / "_render.html").unlink(missing_ok=True)

    writer = PdfWriter()
    for blob in (front_pdf, body_pdf):
        for pg in PdfReader(io.BytesIO(blob)).pages:
            writer.add_page(pg)
    writer.add_metadata({"/Title": TITLE, "/Author": "Rahul Joy Suneel (1DS25SCS10)"})
    with open(OUT, "wb") as f:
        writer.write(f)
    print(f"wrote {OUT} ({len(writer.pages)} pages)")
    for level, text, pg in entries:
        print(f"{'  ' * (level - 1)}{text} .... {pg}")


if __name__ == "__main__":
    main()
