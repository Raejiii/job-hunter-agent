"""Run the Job Hunter pipeline from the terminal.

    python cli.py samples/resume_ananya.txt samples/job_genai_engineer.txt
"""

import sys
import time
from pathlib import Path

from job_hunter import build_graph
from job_hunter.documents import extract_text
from job_hunter.scoring import score_band


def main(resume_path: str, job_path: str) -> None:
    resume_text = extract_text(resume_path, Path(resume_path).read_bytes())
    job_text = extract_text(job_path, Path(job_path).read_bytes())

    graph = build_graph()
    start = time.perf_counter()
    state: dict = {"resume_text": resume_text, "job_text": job_text}
    for update in graph.stream(state, stream_mode="updates"):
        for node, delta in update.items():
            for entry in delta.get("trace", []):
                print(f"[{time.perf_counter() - start:6.1f}s] {entry['agent']:<20} {entry['message']}")
            state.update({k: v for k, v in delta.items() if k not in ("trace", "drafts", "critiques")})

    skills = state["skills"]
    print(f"\nFIT SCORE: {state['score']}/100 ({score_band(state['score'])})")
    print("Matched required:", ", ".join(skills.matched_required) or "-")
    print("Missing required:", ", ".join(skills.missing_required) or "-")
    print("\nSTRENGTHS:\n- " + "\n- ".join(state["judgement"].strengths))
    print("CONCERNS:\n- " + "\n- ".join(state["judgement"].concerns))
    print("\nLEARNING PLAN:")
    for item in state["plan"].items:
        print(f"- {item.skill} ({item.days}d): {item.action}")
    print("\nINTERVIEW QUESTIONS:")
    for q in state["interview"].questions:
        print(f"- [{q.category}] {q.question}")
    print(f"\nCOVER LETTER (final score {state['critique'].score}/10, draft #{state['best_draft']} of {state['revisions']}):\n")
    print(state["letter"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
