"""System-level checks and a matcher comparison used in the project report.

    python report/verify.py        # writes report/results.json
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from job_hunter import config  # noqa: E402
from job_hunter.documents import extract_text  # noqa: E402
from job_hunter.guardrails import unsupported_claims  # noqa: E402
from job_hunter.scoring import expand_requirements, match_skills  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
resume = (ROOT / "samples/resume_ananya.txt").read_text()
results: dict = {}

# --- 1. Matcher comparison: naive keyword equality vs. the hybrid matcher -------------
# Raw "must have" bullets copied from the two sample postings, with hand-labelled truth
# for the sample candidate (True = the resume genuinely shows this skill).
CASES = {
    "GenAI Engineer": [
        ("Strong Python", True),
        ("LangChain or LangGraph", True),
        ("Retrieval-augmented generation (RAG) and vector databases", True),
        ("FastAPI or a similar Python web framework", True),
        ("Docker", True),
        ("Kubernetes", False),
        ("Experience evaluating LLM outputs (LLM evaluation)", False),
    ],
    "Senior Data Scientist": [
        ("Python, pandas, scikit-learn", True),
        ("Deep learning with PyTorch or TensorFlow", False),
        ("Statistics and A/B testing", False),
        ("Spark or other big-data tooling", False),
        ("SQL", True),
    ],
}
SKILLS = ["Python", "SQL", "JavaScript", "FastAPI", "Flask", "REST APIs", "PostgreSQL", "Redis", "LangChain",
          "scikit-learn", "pandas", "NumPy", "prompt engineering", "FAISS", "Docker", "Git", "GitHub Actions",
          "Linux", "pytest"]


def naive(bullet: str) -> bool:
    return bullet.lower() in {s.lower() for s in SKILLS}


def hybrid(bullet: str) -> bool:
    parts = expand_requirements([bullet])
    return bool(parts) and all(match_skills([p], [], SKILLS, resume).matched_required for p in parts)


rows = []
for job, cases in CASES.items():
    for bullet, truth in cases:
        rows.append({"job": job, "requirement": bullet, "truth": truth, "naive": naive(bullet), "hybrid": hybrid(bullet)})
results["matcher"] = {
    "rows": rows,
    "naive_accuracy": sum(r["naive"] == r["truth"] for r in rows) / len(rows),
    "hybrid_accuracy": sum(r["hybrid"] == r["truth"] for r in rows) / len(rows),
}

# --- 2. Functional checks --------------------------------------------------------------
checks = {}

# PDF upload: print the sample resume to PDF with Chrome, then extract it back.
with tempfile.TemporaryDirectory() as tmp:
    html = Path(tmp) / "resume.html"
    html.write_text("<pre style='font: 11pt sans-serif; white-space: pre-wrap'>" + resume + "</pre>")
    pdf = Path(tmp) / "resume.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], capture_output=True, timeout=60)
    text = extract_text("resume.pdf", pdf.read_bytes())
    checks["pdf_upload"] = all(k in text for k in ["ANANYA RAO", "FastAPI", "63%"])

# Ollama unreachable: the status check used by the sidebar.
import httpx  # noqa: E402

try:
    httpx.get("http://localhost:9/api/tags", timeout=2)
    checks["ollama_down_detected"] = False
except Exception:
    checks["ollama_down_detected"] = True

# Guardrail on a known hallucinated sentence produced by an early version of the writer.
checks["guardrail_catches_fake_metric"] = bool(unsupported_claims(
    "At XYZ Corporation my chatbot achieved a 25% increase in user engagement.", resume))
checks["guardrail_passes_real_metric"] = not unsupported_claims("I cut report latency by 63%.", resume)

# UI behaviour, exercised headlessly with Streamlit's AppTest.
from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
run_button = next(b for b in at.button if "Run the agents" in b.label)
checks["empty_input_blocks_run"] = run_button.disabled

at.text_area(key="resume_text").set_value("Python developer")
at.text_area(key="job_text").set_value("Need Python")
at.run()
next(b for b in at.button if "Run the agents" in b.label).click().run()
checks["short_input_warns"] = any("too short" in w.value for w in at.warning)

config.OLLAMA_BASE_URL = "http://localhost:9"  # nothing listens here
at_down = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
checks["ollama_down_shows_error"] = any("not running" in e.value for e in at_down.error)
config.OLLAMA_BASE_URL = "http://localhost:11434"

# Unit test suite.
proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
checks["pytest_summary"] = proc.stdout.strip().splitlines()[-1]

results["checks"] = checks
results["model"] = config.MODEL_NAME
(Path(__file__).parent / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps({k: v for k, v in results.items() if k != "matcher"} |
                 {"naive_acc": results["matcher"]["naive_accuracy"], "hybrid_acc": results["matcher"]["hybrid_accuracy"]},
                 indent=2))
for r in rows:
    print(f"{r['truth']!s:5} naive={r['naive']!s:5} hybrid={r['hybrid']!s:5}  {r['requirement']}")
