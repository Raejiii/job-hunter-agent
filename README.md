# 🎯 Job Hunter: a multi-agent career copilot

Job Hunter is a team of seven AI agents. Give it your resume and a job description, and it:

1. **scores your fit** with an explainable hybrid score (deterministic skill matching plus an LLM judgement)
2. **finds your skill gaps** and turns each one into a short, concrete portfolio mini-project
3. **predicts interview questions**, including "gap-probe" questions aimed at your weak spots
4. **writes a tailored cover letter**, which a critic agent reviews and sends back for revision. A
   hallucination guardrail rejects any metric or employer that isn't in your resume.

It runs **100% locally** on [Ollama](https://ollama.com) (Llama 3.2 3B), so your resume never leaves
your machine. It's built with **LangChain**, **LangGraph** and **Streamlit**.

## Architecture

```
START ─┬─> Resume Parser ─┐
       └─> Job Analyst ───┴─> Match Scorer ─┬─> Gap Coach ─────────> END
                                            ├─> Interview Coach ───> END
                                            └─> Cover Letter Writer <──┐
                                                       │             revise
                                                       v               │
                                                 Letter Critic ────────┴─ accept ─> pick best ─> END
```

| Agent | What it does | LangChain / LangGraph features |
|---|---|---|
| Resume Parser | Resume → `CandidateProfile` | `ChatPromptTemplate`, `with_structured_output` (Pydantic) |
| Job Analyst | JD → must-have vs. nice-to-have skills | structured output, rule-based post-processing |
| Match Scorer | Hybrid fit score + strengths/concerns | deterministic matcher plus LLM-as-judge |
| Gap Coach | Mini-project plan per missing skill | structured output |
| Interview Coach | 6 likely questions with answer hints | structured output |
| Cover Letter Writer | Drafts and revises the letter | LCEL `prompt \| llm \| StrOutputParser` |
| Letter Critic | 5-point rubric plus fact-check guardrail | conditional edge (revise/accept loop) |

The resume and job agents run **in parallel** (fan-out from `START`, join at `match`), and so do the three
downstream agents. Parallel branches write to append-only state channels (`Annotated[list, operator.add]`).

### Reliability on a small local model

A 3B-parameter model is fast and private, but it isn't always reliable. Job Hunter adds deterministic
code wherever correctness matters:

- **Skill matching is code, not vibes.** It handles aliases (`k8s` → Kubernetes), skill families (FAISS
  counts as a *vector database*), alternatives (`LangChain or LangGraph`), compound requirements
  (`RAG and vector databases` → two skills), and a fallback search of the raw resume text.
- **Hybrid score:** `55% must-have coverage + 15% nice-to-have coverage + 30% LLM experience fit`. The LLM
  can't inflate the score.
- **Rubric critic:** the model answers five yes/no checks and Python computes the score. This avoids
  "everything is a 7/10".
- **Hallucination guardrail:** every number in the cover letter must appear in the resume or the JD.
  Placeholder employers ("XYZ Corp") and `[brackets]` are rejected. A flagged draft is capped at 4/10 and
  sent back with the exact problem quoted.
- **Best-draft selection:** revisions can make a letter worse, so the highest-scoring draft wins.
- **Runaway-generation guard:** output is capped at `num_predict`, with a request timeout and an automatic
  retry at a different temperature (`with_fallbacks`) if the JSON fails to parse.

## Quick start

```bash
# 1. Install Ollama and pull the model
brew install ollama          # or see https://ollama.com/download
ollama serve &               # skip if the Ollama app is already running
ollama pull llama3.2

# 2. Install Python deps (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the web app
streamlit run app.py
```

Click **Load sample resume + job** in the sidebar, then **Run the agents**.

To use the CLI instead:

```bash
python cli.py samples/resume_ananya.txt samples/job_genai_engineer.txt
```

### Configuration

| Env var | Default | Purpose |
|---|---|---|
| `JOB_HUNTER_MODEL` | `llama3.2` | Any Ollama chat model (e.g. `qwen2.5:7b`, `llama3.1:8b`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `JOB_HUNTER_MAX_REVISIONS` | `2` | Max critic → writer loops |
| `JOB_HUNTER_MAX_TOKENS` | `1200` | Output cap per call |

## Tests

```bash
pytest -q
```

The graph tests use a scripted fake chat model, so they run in under a second without Ollama. They cover
the parallel fan-out, the revise loop and its limit, best-draft selection, and the guardrail overriding a
lenient critic.

## Project layout

```
job_hunter/
  agents.py      # the seven agents (LangChain chains wrapped as graph nodes)
  graph.py       # LangGraph StateGraph wiring
  prompts.py     # ChatPromptTemplates
  schemas.py     # Pydantic structured-output contracts
  scoring.py     # deterministic skill matcher and hybrid score
  guardrails.py  # hallucination fact-checker
  documents.py   # PDF/TXT text extraction
  config.py
app.py           # Streamlit UI with a live agent pipeline view
cli.py           # terminal runner
samples/         # fictional resume and job descriptions for demos
tests/
```

## Limitations

- A 3B model is quick but sometimes misses requirements or gives generic advice. Larger Ollama models
  (`llama3.1:8b`, `qwen2.5:7b`) improve quality if you have 16 GB+ RAM.
- The numeric guardrail checks figures and placeholders, but not every possible factual claim.
- A full run takes about 2–3 minutes on an 8 GB Apple M1.

---
Built by Rahul Joy Suneel · M.Tech CSE, Dayananda Sagar College of Engineering
