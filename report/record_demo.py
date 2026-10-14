"""Record a short demo video of the app (the model's wait time is cut out).

    streamlit run app.py            # in another terminal
    python report/record_demo.py    # writes docs/demo.mp4 (needs ffmpeg)
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo.mp4"
SIZE = {"width": 1280, "height": 800}

tmp = Path(tempfile.mkdtemp())
marks: dict[str, float] = {}

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    ctx = browser.new_context(viewport=SIZE, record_video_dir=str(tmp), record_video_size=SIZE)
    page = ctx.new_page()
    t0 = time.time()
    page.goto("http://localhost:8501", wait_until="networkidle")
    page.get_by_text("Load sample resume + job").wait_for()
    page.wait_for_timeout(800)
    marks["start"] = time.time() - t0

    page.get_by_text("Load sample resume + job").click()
    page.wait_for_timeout(1600)
    page.get_by_role("button", name="Run the agents").click()
    marks["run"] = time.time() - t0
    page.wait_for_timeout(400)
    page.wait_for_timeout(300)
    page.locator(".agents").evaluate("e => e.scrollIntoView({block: 'start', behavior: 'smooth'})")
    # Show agents lighting up until the parallel fan-out is visible.
    page.locator(".agent.done").nth(2).wait_for(timeout=300_000)
    page.wait_for_timeout(1500)
    marks["cut"] = time.time() - t0

    page.get_by_role("tab", name="Fit report").wait_for(timeout=600_000)
    page.wait_for_timeout(1500)
    page.locator("h2").last.evaluate("e => e.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(600)
    marks["resume"] = time.time() - t0
    page.wait_for_timeout(2600)
    for tab, pause in [("Learning plan", 1300), ("Interview prep", 1300), ("Cover letter", 2200)]:
        page.get_by_role("tab", name=tab).click()
        page.wait_for_timeout(pause)
    marks["end"] = time.time() - t0
    video = page.video.path()
    ctx.close()
    browser.close()

print(json.dumps(marks))


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def stitch(video: Path, marks: dict[str, float], out: Path) -> float:
    """Setup at normal speed, the agents working as a fast-forwarded timelapse, then a crossfade to the results."""
    ff_from = marks["run"] + 3.0
    ff_len = 2.6  # seconds the timelapse should last
    speed = (marks["cut"] - ff_from) / ff_len
    enc = ["-r", "30", "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"]
    parts = {
        "s1": f"trim={marks['start']}:{ff_from},setpts=PTS-STARTPTS",
        "s2": f"trim={ff_from}:{marks['cut']},setpts=(PTS-STARTPTS)/{speed}",
        "s3": f"trim={marks['resume']}:{marks['end']},setpts=PTS-STARTPTS",
    }
    for name, vf in parts.items():  # one graph per segment: a single long-range graph stalls ffmpeg
        ffmpeg("-i", str(video), "-vf", vf, *enc, str(tmp / f"{name}.mp4"))
    (tmp / "list.txt").write_text("file 's1.mp4'\nfile 's2.mp4'\n")
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"), *enc, str(tmp / "a.mp4"))
    a = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(tmp / "a.mp4")], capture_output=True, text=True).stdout)
    fade = 0.4
    ffmpeg("-i", str(tmp / "a.mp4"), "-i", str(tmp / "s3.mp4"), "-filter_complex",
           f"[0:v][1:v]xfade=transition=fade:duration={fade}:offset={a - fade},format=yuv420p[v]",
           "-map", "[v]", *enc, "-movflags", "+faststart", str(out))
    return a + (marks["end"] - marks["resume"]) - fade


length = stitch(Path(video), marks, OUT)
shutil.rmtree(tmp, ignore_errors=True)
print(f"wrote {OUT} ({length:.1f}s)")
