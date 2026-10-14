"""Capture screenshots of the running app for the project report.

    streamlit run app.py            # in another terminal
    python report/capture.py [genai|ds]
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "img"
OUT.mkdir(exist_ok=True)
JOB = sys.argv[1] if len(sys.argv) > 1 else "genai"
prefix = "" if JOB == "genai" else "ds_"


def shot(page, name, **kw):
    page.wait_for_timeout(700)
    page.screenshot(path=str(OUT / f"{prefix}{name}.png"), **kw)
    print("saved", name, flush=True)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
    page.goto("http://localhost:8501", wait_until="networkidle")
    page.get_by_text("Load sample resume + job").wait_for()
    if JOB != "genai":
        page.get_by_text("Senior Data Scientist · Fernwood Health").click()
    if JOB == "genai":
        shot(page, "01_home_empty")
    page.get_by_text("Load sample resume + job").click()
    page.wait_for_timeout(1500)
    if JOB == "genai":
        shot(page, "02_inputs_loaded")

    page.get_by_role("button", name="Run the agents").click()
    start = time.time()
    if JOB == "genai":
        # Catch the live view while the downstream agents run in parallel.
        page.locator(".agent.done").nth(2).wait_for(timeout=300_000)
        page.wait_for_timeout(1500)
        page.locator(".agents").scroll_into_view_if_needed()
        page.mouse.wheel(0, 250)
        shot(page, "03_pipeline_running")

    page.get_by_role("tab", name="Fit report").wait_for(timeout=600_000)
    print(f"run finished in {time.time() - start:.0f}s", flush=True)
    page.wait_for_timeout(2500)
    page.set_viewport_size({"width": 1440, "height": 2400})  # tall, so whole tab panels fit on screen

    def tab(name, file):
        page.get_by_role("tab", name=name).click()
        page.wait_for_timeout(1200)
        page.locator("[data-testid='stTabs']").first.screenshot(path=str(OUT / f"{prefix}{file}.png"))
        print("saved", file, flush=True)

    tab("Fit report", "04_fit_report")
    if JOB == "genai":
        tab("Learning plan", "05_learning_plan")
        page.get_by_role("tab", name="Interview prep").click()
        page.locator("[role='tabpanel']:visible [data-testid='stExpander'] summary").first.click()
        tab("Interview prep", "06_interview_prep")
        tab("Cover letter", "07_cover_letter")
        page.get_by_role("tab", name="Cover letter").click()
        for s in page.locator("[role='tabpanel']:visible [data-testid='stExpander'] summary").all()[:1]:
            s.click()
        tab("Cover letter", "08_revision_history")
        tab("Agent trace", "09_agent_trace")
    browser.close()
