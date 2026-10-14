"""Streamlit front end for the Job Hunter multi-agent pipeline.

    streamlit run app.py
"""

import html
import time
from pathlib import Path

import httpx
import plotly.graph_objects as go
import streamlit as st

from job_hunter import build_graph, config
from job_hunter.documents import extract_text
from job_hunter.scoring import display_name, score_band

SAMPLES = Path(__file__).parent / "samples"
APPEND_CHANNELS = {"trace", "drafts", "critiques"}
PIPELINE = [
    ("parse_resume", "Resume Parser", "Extracts a structured profile"),
    ("analyze_job", "Job Analyst", "Splits must-have vs nice-to-have"),
    ("match", "Match Scorer", "Hybrid rule + LLM fit score"),
    ("gap_coach", "Gap Coach", "Plans mini-projects for gaps"),
    ("interview_coach", "Interview Coach", "Predicts likely questions"),
    ("write_letter", "Cover Letter Writer", "Drafts a tailored letter"),
    ("critique_letter", "Letter Critic", "Rubric review + fact-check"),
]

st.set_page_config(page_title="Job Hunter · Multi-agent career copilot", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero h1 {font-size: 2.3rem; margin-bottom: 0; letter-spacing: -0.02em;}
.hero p {color: #9aa4b2; margin-top: .25rem; font-size: 1.02rem;}
.pill {display:inline-block; padding:3px 11px; margin:3px 4px 3px 0; border-radius:999px; font-size:.84rem; border:1px solid;}
.pill.ok {background:rgba(34,197,94,.12); border-color:rgba(34,197,94,.45); color:#4ade80;}
.pill.miss {background:rgba(239,68,68,.12); border-color:rgba(239,68,68,.45); color:#f87171;}
.pill.pref {background:rgba(148,163,184,.10); border-color:rgba(148,163,184,.35); color:#cbd5e1;}
.agents {display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:8px; margin:.5rem 0 1rem;}
.agent {border:1px solid rgba(148,163,184,.25); border-radius:10px; padding:9px 11px; font-size:.82rem; background:rgba(148,163,184,.05);}
.agent b {display:block; font-size:.9rem; margin-bottom:2px;}
.agent span {color:#9aa4b2;}
.agent.running {border-color:#f59e0b; box-shadow:0 0 0 1px #f59e0b inset; animation:pulse 1.4s ease-in-out infinite;}
.agent.done {border-color:rgba(34,197,94,.6);}
.agent.done b::after {content:" ✓"; color:#4ade80;}
@keyframes pulse {50% {background:rgba(245,158,11,.10);}}
.card {border:1px solid rgba(148,163,184,.22); border-radius:12px; padding:14px 16px; margin-bottom:10px; background:rgba(148,163,184,.04);}
.card h4 {margin:0 0 4px; font-size:1rem;}
.muted {color:#9aa4b2; font-size:.88rem;}
.letter {white-space:pre-wrap; line-height:1.6; font-size:.97rem; border-left:3px solid #6366f1; padding:4px 0 4px 16px;}
</style>
""", unsafe_allow_html=True)


# --- Sidebar --------------------------------------------------------------------------

def ollama_status() -> tuple[bool, str]:
    try:
        tags = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=2).json()
    except Exception:
        return False, "Ollama is not running. Start it with `ollama serve`."
    names = [m["name"] for m in tags.get("models", [])]
    if not any(n.split(":")[0] == config.MODEL_NAME.split(":")[0] for n in names):
        return False, f"Model `{config.MODEL_NAME}` not found. Run `ollama pull {config.MODEL_NAME}`."
    return True, f"Connected · `{config.MODEL_NAME}` running locally"


with st.sidebar:
    st.markdown("### ⚙️ Setup")
    ok, msg = ollama_status()
    (st.success if ok else st.error)(msg)
    config.MAX_LETTER_REVISIONS = st.slider("Max cover-letter revisions", 0, 3, config.MAX_LETTER_REVISIONS,
                                            help="How many times the critic can send a letter back to the writer.")
    st.markdown("### 📂 Try a sample")
    sample_job = st.radio("Job description", ["GenAI Engineer · Nimbus Labs", "Senior Data Scientist · Fernwood Health"],
                          label_visibility="collapsed")
    if st.button("Load sample resume + job", use_container_width=True):
        st.session_state.resume_text = (SAMPLES / "resume_ananya.txt").read_text()
        job_file = "job_genai_engineer.txt" if sample_job.startswith("GenAI") else "job_data_scientist.txt"
        st.session_state.job_text = (SAMPLES / job_file).read_text()
        st.session_state.pop("result", None)
    st.markdown("---")
    st.caption("100% local: your resume never leaves this machine. Built with LangChain, LangGraph, Ollama and Streamlit.")


# --- Inputs ---------------------------------------------------------------------------

st.markdown("""<div class="hero"><h1>🎯 Job Hunter</h1>
<p>A team of AI agents that reads your resume and a job post, scores the fit, plans how to close
your skill gaps, preps you for the interview and writes a fact-checked cover letter.</p></div>""",
            unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    st.markdown("##### 📄 Your resume")
    upload = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt", "md"], label_visibility="collapsed")
    if upload is not None and st.session_state.get("uploaded_name") != upload.name:
        st.session_state.resume_text = extract_text(upload.name, upload.getvalue())
        st.session_state.uploaded_name = upload.name
    resume_text = st.text_area("Resume text", key="resume_text", height=260, placeholder="…or paste your resume here",
                               label_visibility="collapsed")
with right:
    st.markdown("##### 💼 Job description")
    job_text = st.text_area("Job description", key="job_text", height=330, placeholder="Paste the job posting here",
                            label_visibility="collapsed")

run = st.button("🚀 Run the agents", type="primary", use_container_width=True,
                disabled=not ok or not resume_text.strip() or not job_text.strip())


# --- Live pipeline --------------------------------------------------------------------

def render_pipeline(slot, status: dict[str, str]) -> None:
    cells = "".join(
        f'<div class="agent {status.get(node, "")}"><b>{label}</b><span>{desc}</span></div>'
        for node, label, desc in PIPELINE
    )
    slot.markdown(f'<div class="agents">{cells}</div>', unsafe_allow_html=True)


if run:
    if len(resume_text.split()) < 40 or len(job_text.split()) < 30:
        st.warning("That looks too short. Paste the full resume and job description for a meaningful analysis.")
        st.stop()
    pipeline_slot = st.empty()
    status: dict[str, str] = {}
    render_pipeline(pipeline_slot, status)
    state: dict = {"resume_text": resume_text, "job_text": job_text}
    started = time.perf_counter()
    try:
        with st.status("Agents are working… (a local 3B model takes ~2–3 minutes)", expanded=True) as box:
            graph = build_graph()
            for mode, chunk in graph.stream(state, stream_mode=["tasks", "updates"]):
                if mode == "tasks" and "input" in chunk:
                    status[chunk["name"] if chunk["name"] != "select_best" else "critique_letter"] = "running"
                elif mode == "updates":
                    for node, delta in chunk.items():
                        status[node if node != "select_best" else "critique_letter"] = "done"
                        for key, value in (delta or {}).items():
                            if key in APPEND_CHANNELS:
                                state.setdefault(key, []).extend(value)
                            else:
                                state[key] = value
                        for entry in (delta or {}).get("trace", []):
                            box.write(f"**{entry['agent']}** · {entry['message']}  \n"
                                      f"<span class='muted'>{time.perf_counter() - started:.0f}s</span>",
                                      unsafe_allow_html=True)
                render_pipeline(pipeline_slot, status)
            state["elapsed"] = time.perf_counter() - started
            box.update(label=f"Done in {state['elapsed']:.0f}s", state="complete", expanded=False)
    except Exception as exc:  # surface model/connection problems without a stack trace
        st.error(f"The agents hit a problem: {exc}")
        st.stop()
    pipeline_slot.empty()
    st.session_state.result = state


# --- Results --------------------------------------------------------------------------

def pills(*groups: tuple[list[str], str]) -> str:
    """Render (items, css-kind) groups as one row of skill chips."""
    chips = "".join(f'<span class="pill {kind}">{html.escape(display_name(i))}</span>' for items, kind in groups for i in items)
    return chips or '<span class="muted">none listed</span>'


def gauge(score: int) -> go.Figure:
    color = "#22c55e" if score >= 80 else "#84cc16" if score >= 60 else "#f59e0b" if score >= 40 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100", "font": {"size": 40}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0}, "bar": {"color": color, "thickness": 0.75},
               "bgcolor": "rgba(148,163,184,.12)", "borderwidth": 0,
               "steps": [{"range": [0, 40], "color": "rgba(239,68,68,.08)"},
                         {"range": [40, 60], "color": "rgba(245,158,11,.08)"},
                         {"range": [60, 100], "color": "rgba(34,197,94,.08)"}]},
    ))
    fig.update_layout(height=230, margin=dict(l=30, r=30, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#94a3b8"})
    return fig


result = st.session_state.get("result")
if result:
    profile, job, skills, judgement = result["profile"], result["job"], result["skills"], result["judgement"]
    st.markdown(f"## {html.escape(profile.name)} → {html.escape(job.title)} at {html.escape(job.company)}")

    overview, plan_tab, interview_tab, letter_tab, trace_tab = st.tabs(
        ["📊 Fit report", "🧭 Learning plan", "🎤 Interview prep", "✉️ Cover letter", "🔍 Agent trace"])

    with overview:
        g, s = st.columns([1, 1.6])
        with g:
            st.plotly_chart(gauge(result["score"]), use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"<div style='text-align:center'><b>{score_band(result['score'])}</b><br>"
                        f"<span class='muted'>{len(skills.matched_required)} of "
                        f"{len(skills.matched_required) + len(skills.missing_required)} must-haves · "
                        f"LLM experience fit {judgement.experience_fit}/100</span></div>", unsafe_allow_html=True)
        with s:
            st.markdown("**Must-have skills**")
            st.markdown(pills((skills.matched_required, "ok"), (skills.missing_required, "miss")),
                        unsafe_allow_html=True)
            st.markdown("**Nice-to-have skills**")
            st.markdown(pills((skills.matched_preferred, "ok"), (skills.missing_preferred, "pref")),
                        unsafe_allow_html=True)
            st.markdown(f"> {judgement.verdict}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### ✅ Strengths")
            for item in judgement.strengths:
                st.markdown(f"- {item}")
        with c2:
            st.markdown("##### ⚠️ Concerns")
            for item in judgement.concerns:
                st.markdown(f"- {item}")
        with st.expander("How is the score calculated?"):
            st.markdown(
                f"The fit score blends **deterministic skill coverage** with the model's judgement so it "
                f"can't be inflated by a hallucination:\n\n"
                f"- {config.WEIGHT_REQUIRED:.0%} × must-have coverage ({skills.required_coverage:.0%})\n"
                f"- {config.WEIGHT_PREFERRED:.0%} × nice-to-have coverage ({skills.preferred_coverage:.0%})\n"
                f"- {config.WEIGHT_LLM_JUDGEMENT:.0%} × LLM experience fit ({judgement.experience_fit}/100)\n\n"
                "Skills are matched with aliases (k8s → Kubernetes), skill families (FAISS counts as a vector "
                "database), alternatives ('LangChain or LangGraph') and a fallback search of the raw resume text.")

    with plan_tab:
        items = result["plan"].items
        if not items:
            st.success("No gaps to close. You already cover every listed skill.")
        else:
            st.markdown(f"**{len(items)} mini-projects · about {sum(i.days for i in items)} days of focused work**")
            day = 0
            for item in items:
                st.markdown(
                    f"<div class='card'><h4>{html.escape(item.skill)} "
                    f"<span class='muted'>· day {day + 1}–{day + item.days}</span></h4>"
                    f"<div class='muted'>{html.escape(item.why_it_matters)}</div>"
                    f"<div style='margin-top:6px'>🛠️ {html.escape(item.action)}</div></div>",
                    unsafe_allow_html=True)
                day += item.days

    with interview_tab:
        icons = {"technical": "🧠", "behavioural": "🤝", "behavioral": "🤝", "gap-probe": "🎯"}
        for q in result["interview"].questions:
            with st.expander(f"{icons.get(q.category, '❓')} {q.question}"):
                st.caption(q.category.title())
                st.markdown(f"**How to answer:** {q.answer_hint}")

    with letter_tab:
        review = result["critique"]
        st.markdown(f"**Final letter: draft #{result['best_draft']} of {result['revisions']} · "
                    f"critic score {review.score}/10**")
        st.markdown(f"<div class='letter'>{html.escape(result['letter'])}</div>", unsafe_allow_html=True)
        st.download_button("⬇️ Download cover letter", result["letter"],
                           file_name=f"cover_letter_{job.company.replace(' ', '_')}.txt", use_container_width=True)
        st.markdown("##### Revision history")
        for i, (draft, rev) in enumerate(zip(result["drafts"], result["critiques"]), start=1):
            flag = "🛑" if rev.fact_issues else "✅"
            with st.expander(f"Draft #{i} · {rev.score}/10 · fact-check {flag}"):
                st.markdown(" ".join(f"{'✅' if v else '❌'} {k}" for k, v in rev.checks.items()))
                if rev.feedback:
                    st.markdown("**Feedback sent to the writer:**\n" + "\n".join(f"- {f}" for f in rev.feedback))
                st.text(draft)

    with trace_tab:
        st.markdown(f"Completed in **{result['elapsed']:.0f}s** on `{config.MODEL_NAME}` via Ollama.")
        st.code("""START ─┬─> Resume Parser ─┐
       └─> Job Analyst ───┴─> Match Scorer ─┬─> Gap Coach ─────────> END
                                            ├─> Interview Coach ───> END
                                            └─> Cover Letter Writer <──┐
                                                       │             revise
                                                       v               │
                                                 Letter Critic ────────┴─ accept ─> pick best ─> END""",
                language=None)
        for entry in result["trace"]:
            st.markdown(f"- **{entry['agent']}**: {entry['message']}")
        with st.expander("Extracted candidate profile (JSON)"):
            st.json(profile.model_dump())
        with st.expander("Extracted job requirements (JSON)"):
            st.json(job.model_dump())
