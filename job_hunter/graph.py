"""LangGraph workflow wiring the agents together.

    START ─┬─> parse_resume ─┐
           └─> analyze_job ──┴─> match ─┬─> gap_coach ───────> END
                                        ├─> interview_coach ─> END
                                        └─> write_letter <──┐
                                               │          revise
                                               v            │
                                         critique_letter ───┴─ accept ─> select_best ─> END
"""

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import Agents, LLMFactory, default_llm_factory, should_revise


class JobHunterState(TypedDict, total=False):
    resume_text: str
    job_text: str
    profile: Any
    job: Any
    skills: Any
    judgement: Any
    score: int
    plan: Any
    interview: Any
    letter: str
    revisions: int
    critique: Any
    best_draft: int
    # Append-only channels so parallel agents can all contribute.
    drafts: Annotated[list, operator.add]
    critiques: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]


def build_graph(llm_factory: LLMFactory = default_llm_factory):
    agents = Agents(llm_factory)
    g = StateGraph(JobHunterState)

    g.add_node("parse_resume", agents.parse_resume)
    g.add_node("analyze_job", agents.analyze_job)
    g.add_node("match", agents.match)
    g.add_node("gap_coach", agents.gap_coach)
    g.add_node("interview_coach", agents.interview_coach)
    g.add_node("write_letter", agents.write_letter)
    g.add_node("critique_letter", agents.critique_letter)
    g.add_node("select_best", agents.select_best)

    g.add_edge(START, "parse_resume")
    g.add_edge(START, "analyze_job")
    g.add_edge(["parse_resume", "analyze_job"], "match")
    g.add_edge("match", "gap_coach")
    g.add_edge("match", "interview_coach")
    g.add_edge("match", "write_letter")
    g.add_edge("write_letter", "critique_letter")
    g.add_conditional_edges("critique_letter", should_revise, {"revise": "write_letter", "accept": "select_best"})
    g.add_edge("select_best", END)
    g.add_edge("gap_coach", END)
    g.add_edge("interview_coach", END)

    return g.compile()


NODE_LABELS = {
    "parse_resume": "Resume Parser",
    "analyze_job": "Job Analyst",
    "match": "Match Scorer",
    "gap_coach": "Gap Coach",
    "interview_coach": "Interview Coach",
    "write_letter": "Cover Letter Writer",
    "critique_letter": "Letter Critic",
    "select_best": "Letter Critic",
}
