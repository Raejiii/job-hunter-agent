"""Pydantic schemas used as structured-output contracts between agents."""

from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    name: str = Field(description="Candidate's full name, or 'Candidate' if not stated")
    headline: str = Field(description="One-line professional summary of the candidate")
    years_experience: float = Field(description="Total years of professional experience; 0 for students")
    skills: list[str] = Field(description="Individual technical skills, tools, languages and frameworks")
    soft_skills: list[str] = Field(default_factory=list, description="Soft skills such as leadership or communication")
    education: list[str] = Field(default_factory=list, description="Degrees with institution names")
    experience: list[str] = Field(
        default_factory=list,
        description="Each job as 'Role - Company (dates)', copied exactly from the resume",
    )
    achievements: list[str] = Field(
        default_factory=list,
        description="Every quantified bullet from jobs, projects and awards, copied with its exact numbers",
    )


class JobRequirements(BaseModel):
    title: str = Field(description="The exact job title as written in the posting, e.g. 'GenAI Engineer'")
    company: str = Field(description="Hiring company name, or 'the company' if not stated")
    seniority: str = Field(description="One of: intern, junior, mid, senior, lead")
    required_skills: list[str] = Field(
        description="Must-have technical skills as short names like 'Python' or 'Kubernetes'. Exclude years of experience "
        "and generic disciplines such as 'Software Engineering'."
    )
    preferred_skills: list[str] = Field(default_factory=list, description="Nice-to-have technical skills, short names")
    responsibilities: list[str] = Field(default_factory=list, description="Key responsibilities of the role")
    min_years_experience: float = Field(default=0, description="Minimum years of experience asked for; 0 if not stated")


class FitJudgement(BaseModel):
    """The LLM's qualitative read on fit; combined with the deterministic skill score."""

    experience_fit: int = Field(description="0-100: how well the candidate's experience level and projects fit the role")
    strengths: list[str] = Field(description="3-5 specific reasons this candidate fits, citing resume evidence")
    concerns: list[str] = Field(description="2-4 specific risks a recruiter would raise")
    verdict: str = Field(description="Two-sentence hiring-manager style summary")


class LearningItem(BaseModel):
    skill: str = Field(description="The missing skill to learn")
    why_it_matters: str = Field(description="Why this role needs it, in one sentence")
    action: str = Field(description="A concrete mini-project or resource to close the gap")
    days: int = Field(description="Days of focused effort for the mini-project, between 2 and 14")


class LearningPlan(BaseModel):
    items: list[LearningItem] = Field(description="One item per important missing skill, most important first")


class InterviewQuestion(BaseModel):
    question: str
    category: str = Field(description="One of: technical, behavioural, gap-probe")
    answer_hint: str = Field(description="How this candidate should approach the answer, using their own experience")


class InterviewKit(BaseModel):
    questions: list[InterviewQuestion] = Field(description="Exactly 6 likely interview questions")


class LetterCritique(BaseModel):
    """Rubric the critic model fills in. The score is computed from it in Python, not by the model."""

    specific_opening: bool = Field(description="True if the first sentence is tied to this company's actual work, not generic")
    cites_real_achievements: bool = Field(description="True if it cites at least two concrete, quantified achievements")
    avoids_cliches: bool = Field(description="True if it avoids cliches like 'passionate', 'team player', 'I am writing to apply'")
    honest_about_gaps: bool = Field(description="True if it never claims the skills the candidate lacks")
    strong_closing: bool = Field(description="True if it ends with a clear call to action and the candidate's name")
    feedback: list[str] = Field(description="One specific, actionable fix for every check that failed")

    def checks(self) -> dict[str, bool]:
        return {
            "Specific opening": self.specific_opening,
            "Cites real achievements": self.cites_real_achievements,
            "Avoids cliches": self.avoids_cliches,
            "Honest about gaps": self.honest_about_gaps,
            "Strong closing": self.strong_closing,
        }


class LetterReview(BaseModel):
    """Final verdict on one draft: rubric checks plus deterministic guardrail results."""

    score: int
    checks: dict[str, bool]
    feedback: list[str]
    fact_issues: list[str] = []
