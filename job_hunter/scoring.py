"""Deterministic skill matching and the hybrid fit score.

The LLM is good at reading prose but unreliable at arithmetic and consistency, so the
skill-coverage part of the score is computed in plain Python. The LLM only contributes
a bounded qualitative "experience fit" component.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from . import config

ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "node": "node.js",
    "nodejs": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "llm": "large language models",
    "llms": "large language models",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "aws cloud": "aws",
    "amazon web services": "aws",
    "gcp": "google cloud",
    "ci/cd": "ci cd",
    "cicd": "ci cd",
    "rest": "rest apis",
    "rest api": "rest apis",
    "restful apis": "rest apis",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "golang": "go",
    "py": "python",
    "rag": "retrieval augmented generation",
    "retrieval-augmented generation": "retrieval augmented generation",
}

# A requirement naming a category is satisfied by any concrete member of it.
FAMILIES = {
    "vector database": ["faiss", "pinecone", "chroma", "chromadb", "weaviate", "qdrant", "milvus", "pgvector"],
    "python web framework": ["fastapi", "flask", "django"],
    "web framework": ["fastapi", "flask", "django", "express", "spring boot"],
    "cloud": ["aws", "azure", "google cloud"],
    "ci cd": ["github actions", "jenkins", "gitlab ci", "circleci"],
    "deep learning": ["pytorch", "tensorflow", "keras"],
    "big data": ["spark", "hadoop", "pyspark"],
    "sql": ["postgresql", "mysql", "sqlite", "sql server"],
    "relational database": ["postgresql", "mysql", "sqlite", "sql server"],
}

# Broad disciplines are not checkable skills; the experience judgement covers them.
GENERIC = {"software engineering", "software development", "programming", "coding", "computer science",
           "engineering", "problem solving", "development"}

# Words that carry no meaning when comparing skills ("experience with Docker" == "Docker").
FILLER = {"experience", "with", "in", "of", "and", "knowledge", "proficiency", "strong", "familiarity", "using", "skills"}


def normalize(skill: str) -> str:
    s = skill.lower().strip()
    s = re.sub(r"\(.*?\)", "", s)  # drop parentheticals like "(preferred)"
    s = re.sub(r"[^a-z0-9+#./ -]", " ", s)
    s = " ".join(w for w in s.split() if w not in FILLER)
    s = s.strip(" .-/")
    return ALIASES.get(s, s)


def display_name(skill: str) -> str:
    """'Strong Python' -> 'Python', 'Experience with Docker' -> 'Docker' (for showing to the user)."""
    words = re.sub(r"\s*\(.*?\)", "", skill).split()
    while len(words) > 1 and words[0].lower() in FILLER:
        words.pop(0)
    return " ".join(words).strip(" .") or skill


def expand_requirements(items: list[str]) -> list[str]:
    """Split compound requirements and drop ones that are not skills.

    'RAG and vector databases' -> ['RAG', 'vector databases']; '1+ years of experience' -> dropped.
    Alternatives ('LangChain or LangGraph') stay together and are resolved during matching.
    """
    out: list[str] = []
    for item in items:
        if re.search(r"\d+\s*\+?\s*(years?|yrs)", item, re.I):
            continue
        # Keep a short acronym in parentheses attached to the term it explains:
        # 'Retrieval-augmented generation (RAG) and vector databases' -> 'Retrieval-augmented generation (RAG)'.
        acronym = re.search(r"\(([A-Z][A-Za-z0-9/+.-]{1,9})\)", item)
        base = re.sub(r"\s*\(.*?\)", "", item)
        parts = [display_name(p.strip()) for p in re.split(r"\s+and\s+|\s*&\s*|,\s*", base) if p.strip()]
        parts = [p for p in parts if normalize(p) not in GENERIC]
        if acronym and parts:
            parts[0] += f" ({acronym.group(1)})"
        out.extend(parts)
    return list(dict.fromkeys(out))


def _terms(skill: str) -> set[str]:
    """Every normalised spelling that should count as this skill."""
    terms = set()
    alternatives = re.split(r"\s+or\s+|\s+/\s+", re.sub(r"\s*\((.*?)\)", r" or \1", skill))
    for alt in alternatives:
        n = normalize(alt)
        if not n:
            continue
        terms.add(n)
        terms.update(k for k, v in ALIASES.items() if v == n)
        for family, members in FAMILIES.items():
            if family in n or n.rstrip("s") == family:
                terms.update(members)
    return terms


def _similar(a: str, b: str) -> bool:
    if a == b:
        return True
    # "python" vs "python 3", "docker" vs "docker compose"
    if len(a) >= 3 and len(b) >= 3 and (re.search(rf"\b{re.escape(a)}\b", b) or re.search(rf"\b{re.escape(b)}\b", a)):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.88


def skill_in(skill: str, candidate_skills: list[str], resume_text: str = "") -> bool:
    terms = _terms(skill)
    have = [normalize(c) for c in candidate_skills]
    if any(_similar(t, c) for t in terms for c in have):
        return True
    # Fall back to the raw resume: the extractor may have missed a skill mentioned in prose.
    text = resume_text.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", text) for t in terms if len(t) >= 2)


@dataclass
class SkillMatch:
    matched_required: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    matched_preferred: list[str] = field(default_factory=list)
    missing_preferred: list[str] = field(default_factory=list)

    @property
    def required_coverage(self) -> float:
        total = len(self.matched_required) + len(self.missing_required)
        return len(self.matched_required) / total if total else 1.0

    @property
    def preferred_coverage(self) -> float:
        total = len(self.matched_preferred) + len(self.missing_preferred)
        return len(self.matched_preferred) / total if total else 1.0


def match_skills(required: list[str], preferred: list[str], candidate_skills: list[str], resume_text: str = "") -> SkillMatch:
    m = SkillMatch()
    for s in dict.fromkeys(required):
        (m.matched_required if skill_in(s, candidate_skills, resume_text) else m.missing_required).append(s)
    for s in dict.fromkeys(preferred):
        (m.matched_preferred if skill_in(s, candidate_skills, resume_text) else m.missing_preferred).append(s)
    return m


def fit_score(match: SkillMatch, experience_fit: int) -> int:
    experience_fit = max(0, min(100, experience_fit))
    score = (
        config.WEIGHT_REQUIRED * match.required_coverage * 100
        + config.WEIGHT_PREFERRED * match.preferred_coverage * 100
        + config.WEIGHT_LLM_JUDGEMENT * experience_fit
    )
    return round(score)


def score_band(score: int) -> str:
    if score >= 80:
        return "Strong match"
    if score >= 60:
        return "Good match"
    if score >= 40:
        return "Stretch role"
    return "Weak match"
