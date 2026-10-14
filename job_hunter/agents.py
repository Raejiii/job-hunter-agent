"""Agent node implementations.

Each agent is a LangChain runnable chain (prompt | model) wrapped in a function that reads
from and writes to the shared LangGraph state. The chat model is injected through
`llm_factory` so tests can swap in a fake model.
"""

import re
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from . import config, prompts
from .guardrails import clean_letter, find_cliches, gap_claims, unsupported_claims
from .schemas import (
    CandidateProfile,
    FitJudgement,
    InterviewKit,
    JobRequirements,
    LearningPlan,
    LetterCritique,
    LetterReview,
)
from .scoring import expand_requirements, fit_score, match_skills, score_band

LLMFactory = Callable[[float], BaseChatModel]


def default_llm_factory(temperature: float) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.MODEL_NAME,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        # Small models occasionally loop while emitting JSON; cap the output so a bad
        # generation fails fast and the fallback below gets a chance.
        num_predict=config.MAX_OUTPUT_TOKENS,
        client_kwargs={"timeout": config.REQUEST_TIMEOUT_S},
    )


def _join(items) -> str:
    return "; ".join(items) if items else "none"


def _title_from_posting(text: str) -> str:
    """Postings usually open with 'Title - Company (Location)'."""
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return re.split(r"\s+[-|@–—]\s+|\s+at\s+", first)[0].strip()[:80]


def _company_context(job_text: str, limit: int = 600) -> str:
    """The posting's 'About' paragraph (or its opening) gives the writer something specific to hook into."""
    m = re.search(r"about\b[^\n]*\n(.+?)(?:\n\s*\n|$)", job_text, re.I | re.S)
    return (m.group(1) if m else job_text)[:limit].strip()


def _log(agent: str, message: str) -> dict:
    return {"trace": [{"agent": agent, "message": message}]}


class Agents:
    def __init__(self, llm_factory: LLMFactory = default_llm_factory):
        extract = llm_factory(config.EXTRACT_TEMPERATURE)
        write = llm_factory(config.WRITE_TEMPERATURE)
        retry = llm_factory(config.RETRY_TEMPERATURE)

        def structured(prompt, llm, schema):
            # If the output is truncated or fails validation, retry once at a different
            # temperature: re-running at the same temperature would reproduce the same loop.
            primary = prompt | llm.with_structured_output(schema)
            return primary.with_fallbacks([prompt | retry.with_structured_output(schema)])

        self.resume_chain = structured(prompts.RESUME_PARSER, extract, CandidateProfile)
        self.job_chain = structured(prompts.JOB_ANALYST, extract, JobRequirements)
        self.match_chain = structured(prompts.MATCHER, extract, FitJudgement)
        self.gap_chain = structured(prompts.GAP_COACH, write, LearningPlan)
        self.interview_chain = structured(prompts.INTERVIEW_COACH, write, InterviewKit)
        self.letter_chain = prompts.LETTER_WRITER | write | StrOutputParser()
        self.critic_chain = structured(prompts.LETTER_CRITIC, extract, LetterCritique)

    # --- Stage 1: understand the inputs (run in parallel) -------------------------------

    def parse_resume(self, state: dict) -> dict:
        profile: CandidateProfile = self.resume_chain.invoke({"resume_text": state["resume_text"]})
        return {"profile": profile} | _log(
            "Resume Parser", f"Found {len(profile.skills)} skills and {len(profile.achievements)} achievements for {profile.name}."
        )

    def analyze_job(self, state: dict) -> dict:
        job: JobRequirements = self.job_chain.invoke({"job_text": state["job_text"]})
        if not job.title.strip() or re.search(r"job description|analysis|posting", job.title, re.I):
            job.title = _title_from_posting(state["job_text"]) or job.title
        job.required_skills = expand_requirements(job.required_skills)
        required = {s.lower() for s in job.required_skills}
        job.preferred_skills = [s for s in expand_requirements(job.preferred_skills) if s.lower() not in required]
        return {"job": job} | _log(
            "Job Analyst",
            f"{job.title} at {job.company}: {len(job.required_skills)} required, {len(job.preferred_skills)} preferred skills.",
        )

    # --- Stage 2: score the fit -------------------------------------------------------

    def match(self, state: dict) -> dict:
        profile: CandidateProfile = state["profile"]
        job: JobRequirements = state["job"]
        skills = match_skills(job.required_skills, job.preferred_skills, profile.skills, state["resume_text"])
        meets = profile.years_experience >= job.min_years_experience
        judgement: FitJudgement = self.match_chain.invoke({
            "title": job.title,
            "company": job.company,
            "seniority": job.seniority,
            "min_years": job.min_years_experience,
            "responsibilities": _join(job.responsibilities),
            "name": profile.name,
            "headline": profile.headline,
            "years": profile.years_experience,
            "experience_check": "MEETS the minimum" if meets else "BELOW the minimum",
            "experience": _join(profile.experience),
            "achievements": _join(profile.achievements),
            "matched": _join(skills.matched_required),
            "missing": _join(skills.missing_required),
            "matched_pref": _join(skills.matched_preferred),
        })
        if 0 < judgement.experience_fit <= 10:  # small models sometimes answer on a 0-10 scale
            judgement.experience_fit *= 10
        score = fit_score(skills, judgement.experience_fit)
        return {"skills": skills, "judgement": judgement, "score": score} | _log(
            "Match Scorer",
            f"Fit score {score}/100 ({score_band(score)}). Covers {len(skills.matched_required)} of "
            f"{len(skills.matched_required) + len(skills.missing_required)} required skills.",
        )

    # --- Stage 3: act on the analysis (run in parallel) -------------------------------

    def gap_coach(self, state: dict) -> dict:
        skills = state["skills"]
        missing = skills.missing_required + skills.missing_preferred[:2]
        if not missing:
            return {"plan": LearningPlan(items=[])} | _log("Gap Coach", "No skill gaps to close.")
        plan: LearningPlan = self.gap_chain.invoke({
            "title": state["job"].title,
            "company": state["job"].company,
            "known": _join(state["profile"].skills),
            "missing": _join(missing),
        })
        for item in plan.items:
            item.days = max(1, min(21, item.days))
        total_days = sum(i.days for i in plan.items)
        return {"plan": plan} | _log("Gap Coach", f"Built a {len(plan.items)}-step plan (~{total_days} days).")

    def interview_coach(self, state: dict) -> dict:
        job = state["job"]
        kit: InterviewKit = self.interview_chain.invoke({
            "title": job.title,
            "company": job.company,
            "responsibilities": _join(job.responsibilities),
            "achievements": _join(state["profile"].achievements),
            "missing": _join(state["skills"].missing_required),
        })
        for q in kit.questions:
            q.category = q.category.strip().lower()
        return {"interview": kit} | _log("Interview Coach", f"Prepared {len(kit.questions)} likely questions.")

    def write_letter(self, state: dict) -> dict:
        profile, job = state["profile"], state["job"]
        critique: LetterReview | None = state.get("critique")
        revision_note = ""
        if critique and state.get("letter"):
            revision_note = (
                "REVISE this previous draft:\n" + state["letter"] +
                "\n\nAddress every point of this reviewer feedback:\n- " + "\n- ".join(critique.feedback)
            )
        letter = self.letter_chain.invoke({
            "title": job.title,
            "company": job.company,
            "responsibilities": _join(job.responsibilities),
            "name": profile.name,
            "headline": profile.headline,
            "experience": _join(profile.experience),
            "achievements": "\n- " + "\n- ".join(profile.achievements) if profile.achievements else "none",
            "strengths": _join(state["judgement"].strengths),
            "missing": _join(state["skills"].missing_required),
            "company_context": _company_context(state["job_text"]),
            "revision_note": revision_note,
        })
        letter = clean_letter(letter)
        draft_no = state.get("revisions", 0) + 1
        return {"letter": letter, "revisions": draft_no, "drafts": [letter]} | _log(
            "Cover Letter Writer", f"Wrote draft #{draft_no} ({len(letter.split())} words)."
        )

    def critique_letter(self, state: dict) -> dict:
        letter = state["letter"]
        rubric: LetterCritique = self.critic_chain.invoke({
            "title": state["job"].title,
            "company": state["job"].company,
            "missing": _join(state["skills"].missing_required),
            "letter": letter,
        })
        # Rules that can be checked in code override the model's opinion.
        cliches = find_cliches(letter)
        claims = gap_claims(letter, state["skills"].missing_required)
        rubric.avoids_cliches = not cliches
        rubric.honest_about_gaps = not claims
        checks = rubric.checks()
        score = 2 * sum(checks.values())

        feedback = []
        if not rubric.specific_opening:
            feedback.append(f"Rewrite the first sentence so it names something concrete {state['job'].company} "
                            "does (from the company context) and links it to one of your achievements.")
        if not rubric.cites_real_achievements:
            feedback.append("Cite at least two achievements from the ACHIEVEMENTS list with their exact numbers.")
        if cliches:
            feedback.append("Delete these cliches and say something specific instead: " + ", ".join(f"'{c}'" for c in cliches) + ".")
        feedback += claims
        if not rubric.strong_closing:
            feedback.append("End with one direct sentence asking for an interview, then sign off with your name.")
        feedback += [f for f in rubric.feedback if len(f.split()) > 6][:2]  # keep the model's substantive notes

        issues = unsupported_claims(letter, state["resume_text"], state["job_text"])
        if issues:
            feedback = issues + feedback
            score = min(score, 4)
        words = len(letter.split())
        if words > 330 or words < 150:
            feedback.append(f"Letter is {words} words; keep it between 180 and 300.")
            score = min(score, 6)
        review = LetterReview(score=score, checks=checks, feedback=feedback, fact_issues=issues)
        note = f" Guardrail flagged {len(issues)} unsupported claim(s)." if issues else " Fact-check passed."
        return {"critique": review, "critiques": [review]} | _log(
            "Letter Critic", f"Scored draft #{state['revisions']} at {score}/10.{note}"
        )

    def select_best(self, state: dict) -> dict:
        """Keep the highest-scoring draft; ties go to fewer fact-check flags, then the later revision."""
        scored = list(zip(state["drafts"], state["critiques"]))
        best = max(range(len(scored)), key=lambda i: (scored[i][1].score, -len(scored[i][1].fact_issues), i))
        letter, critique = scored[best]
        return {"letter": letter, "critique": critique, "best_draft": best + 1} | _log(
            "Letter Critic", f"Selected draft #{best + 1} ({critique.score}/10) as the final letter."
        )


def should_revise(state: dict) -> str:
    critique: LetterReview = state["critique"]
    if critique.score >= config.LETTER_PASS_SCORE or state["revisions"] > config.MAX_LETTER_REVISIONS:
        return "accept"
    return "revise"
