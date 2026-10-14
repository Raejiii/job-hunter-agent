"""End-to-end graph tests with a scripted fake model (no Ollama needed)."""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from job_hunter import build_graph
from job_hunter.schemas import (
    CandidateProfile,
    FitJudgement,
    InterviewKit,
    InterviewQuestion,
    JobRequirements,
    LearningItem,
    LearningPlan,
    LetterCritique,
)

RESUME = "Asha Rao. Python developer. Cut API latency by 40% using FastAPI and Docker."
JOB = "Backend Engineer at Acmeless. Must have: Python, FastAPI, Kubernetes. Nice to have: AWS."

GOOD_LETTER = " ".join(["Your platform team ships fast APIs, and I cut API latency by 40% with FastAPI."] * 15) + " Asha Rao"
FAKE_METRIC_LETTER = GOOD_LETTER.replace("40%", "95%")


def critique(passing: bool) -> LetterCritique:
    return LetterCritique(
        specific_opening=passing, cites_real_achievements=True, avoids_cliches=True,
        honest_about_gaps=True, strong_closing=passing, feedback=[] if passing else ["Cut the cliches."],
    )


class FakeLLM(Runnable):
    """Returns canned objects per schema; letters and critiques come from scripted queues."""

    def __init__(self, letters: list[str], critiques: list[LetterCritique]):
        self.letters, self.critiques = letters, critiques
        self.canned: dict[type, Any] = {
            CandidateProfile: CandidateProfile(
                name="Asha Rao", headline="Python developer", years_experience=2,
                skills=["Python", "FastAPI", "Docker"], achievements=["Cut API latency by 40%"],
            ),
            JobRequirements: JobRequirements(
                title="Backend Engineer", company="Acmeless", seniority="junior",
                required_skills=["Python", "FastAPI", "Kubernetes"], preferred_skills=["AWS"],
            ),
            FitJudgement: FitJudgement(experience_fit=70, strengths=["FastAPI"], concerns=["No k8s"], verdict="Solid."),
            LearningPlan: LearningPlan(items=[LearningItem(skill="Kubernetes", why_it_matters="Deploys", action="Minikube", days=90)]),
            InterviewKit: InterviewKit(questions=[InterviewQuestion(question="Why k8s?", category=" Gap-Probe", answer_hint="Be honest")]),
        }

    def invoke(self, input, config=None, **kwargs):
        return AIMessage(self.letters.pop(0))

    def with_structured_output(self, schema, **kwargs):
        if schema is LetterCritique:
            return RunnableLambda(lambda _: self.critiques.pop(0))
        return RunnableLambda(lambda _: self.canned[schema].model_copy(deep=True))


def run(letters, critiques):
    fake = FakeLLM(letters, critiques)
    return build_graph(lambda temperature: fake).invoke({"resume_text": RESUME, "job_text": JOB})


def test_full_pipeline_accepts_first_good_letter():
    out = run([GOOD_LETTER], [critique(True)])
    assert out["skills"].missing_required == ["Kubernetes"]
    assert out["score"] == round(0.55 * 200 / 3 + 0 + 0.30 * 70)  # 2/3 required, 0/1 preferred
    assert out["revisions"] == 1 and out["critique"].score == 10
    assert out["plan"].items[0].days == 21  # clamped from 90
    assert out["interview"].questions[0].category == "gap-probe"
    agents = [t["agent"] for t in out["trace"]]
    assert {"Resume Parser", "Job Analyst", "Match Scorer", "Gap Coach", "Interview Coach"} <= set(agents)


def test_critic_loop_revises_then_stops_at_limit_and_keeps_best():
    letters = [GOOD_LETTER, GOOD_LETTER + " v2", GOOD_LETTER + " v3"]
    out = run(letters, [critique(False), critique(False), critique(False)])
    assert out["revisions"] == 3  # 1 draft + 2 revisions (MAX_LETTER_REVISIONS)
    assert len(out["drafts"]) == 3
    assert out["best_draft"] == 3  # equal scores -> latest revision wins


def test_guardrail_overrides_a_lenient_critic():
    out = run([FAKE_METRIC_LETTER, GOOD_LETTER], [critique(True), critique(True)])
    first, second = out["critiques"]
    assert first.score == 4 and first.fact_issues  # critic said 10, guardrail capped it
    assert second.score == 10
    assert out["letter"] == GOOD_LETTER and out["best_draft"] == 2


def test_generic_title_falls_back_to_posting_header():
    fake = FakeLLM([GOOD_LETTER], [critique(True)])
    fake.canned[JobRequirements].title = "Job Description Analysis"
    fake.canned[FitJudgement].experience_fit = 7  # answered on a 0-10 scale
    out = build_graph(lambda temperature: fake).invoke({"resume_text": RESUME, "job_text": JOB})
    assert out["job"].title == "Backend Engineer"
    assert out["judgement"].experience_fit == 70


def test_code_checks_override_the_critic():
    # The fake critic approves everything; the deterministic checks still catch both problems.
    bad = GOOD_LETTER.replace("Your platform team", "I am thrilled that your platform team", 1)
    bad += " I have run Kubernetes clusters in production."
    out = run([bad, GOOD_LETTER], [critique(True)] * 2)
    first, second = out["critiques"]
    assert not first.checks["Avoids cliches"] and not first.checks["Honest about gaps"]
    assert first.score == 6 and any("Kubernetes" in f for f in first.feedback)
    assert second.score == 10 and out["revisions"] == 2 and out["best_draft"] == 2
