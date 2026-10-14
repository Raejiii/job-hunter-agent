# Demo recording script (about 4–5 minutes)

**Before recording**
- Start Ollama (it runs in the menu bar, or run `ollama serve`) and check `ollama list` shows `llama3.2`.
- Open a terminal in the project folder: `source .venv/bin/activate && streamlit run app.py`
- Close other heavy apps. The model needs about 3 GB of RAM.
- Zoom the browser to about 110% so text is readable in the video.

**Optional speed-up:** a full run takes about 2–3 minutes. You can pause the recording while the agents work, or speed that
part up when editing. Mention it either way ("this runs on a 3B model on my laptop, so I've sped it up").

---

### 1. Intro (20 s)
> "This is Job Hunter, a multi-agent career copilot I built with LangChain and LangGraph. It runs entirely on my laptop
> using Llama 3.2 through Ollama, so no resume data leaves the machine."

Point at the green **Connected · llama3.2** box in the sidebar.

### 2. Architecture (40 s)
Show `job_hunter/graph.py` in your editor (or the README diagram):
> "There are seven agents. The resume parser and job analyst run in parallel, then the match scorer combines them. Then
> three agents run in parallel: a gap coach, an interview coach and a cover-letter writer. The writer is in a loop with a
> critic that sends weak drafts back for revision."

### 3. Run it (40 s plus waiting)
- Click **Load sample resume + job** (GenAI Engineer). Scroll through both text boxes briefly.
- Click **Run the agents**.
> "You can see the agents light up live. Amber means running and a green tick means done. These three are running in
> parallel right now."

### 4. Fit report (40 s)
> "The fit score is hybrid. 70% comes from deterministic skill matching in Python and only 30% from the LLM's judgement, so it
> can't hallucinate a high score."
- Point at the green/red chips: "It understood that FAISS counts as a vector database, and that 'LangChain or LangGraph' is
  satisfied by either."
- Expand **How is the score calculated?**

### 5. Learning plan and interview prep (30 s)
> "For each missing skill it suggests a concrete mini-project. The interview coach predicts questions, including
> gap-probe questions aimed at my weak spots."

### 6. Cover letter (50 s)
> "The critic scores each draft with a five-point rubric, and code-based guardrails check for invented numbers,
> placeholder company names, clichés and false skill claims."
- Open the revision history and expand a draft to show the ✅/❌ checks and the feedback sent to the writer.
> "Every number in the final letter is traceable to the resume. Early versions of this project invented things like a
> '25% increase in engagement', and the guardrail now catches that."
- Click **Download cover letter**.

### 7. Second job (20 s, optional; reuse a previous run or skip the wait)
- Select **Senior Data Scientist**, load and run. Show the **Weak match**:
> "For a senior data science role it correctly says this candidate isn't a fit yet."

### 8. Tests and wrap-up (20 s)
In the terminal run `pytest -q` (20 tests pass in under a second):
> "The whole graph is tested with a fake model, so the logic is verified without needing the LLM. Thanks for watching."
