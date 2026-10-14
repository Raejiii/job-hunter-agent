from job_hunter.scoring import expand_requirements, fit_score, match_skills, normalize, score_band, skill_in


def test_normalize_aliases_and_filler():
    assert normalize("Strong Python") == "python"
    assert normalize("k8s") == "kubernetes"
    assert normalize("Postgres") == "postgresql"
    assert normalize("Experience with Docker") == "docker"


def test_expand_requirements_splits_compounds_and_drops_non_skills():
    out = expand_requirements([
        "1+ years of professional experience",
        "Software Engineering",
        "Strong Python",
        "Retrieval-augmented generation (RAG) and vector databases",
        "LangChain or LangGraph",
    ])
    assert out == ["Python", "Retrieval-augmented generation (RAG)", "vector databases", "LangChain or LangGraph"]


def test_alternatives_match_any_option():
    assert skill_in("LangChain or LangGraph", ["LangGraph"])
    assert skill_in("FastAPI or a similar Python web framework", ["Flask"])


def test_skill_families():
    assert skill_in("vector databases", ["FAISS"])
    assert skill_in("Deep learning", ["PyTorch"])
    assert not skill_in("Kubernetes", ["Docker"])


def test_falls_back_to_resume_text_and_acronyms():
    resume = "Built a RAG chatbot over lecture notes."
    assert skill_in("Retrieval-augmented generation (RAG)", [], resume)
    assert not skill_in("Kubernetes", [], resume)


def test_no_partial_word_false_positives():
    assert not skill_in("Go", [], "Going forward I want to learn Rust")
    assert not skill_in("R", [], "React and Redux")


def test_fit_score_weights_and_bands():
    m = match_skills(["Python", "Docker", "Kubernetes", "SQL"], ["AWS"], ["Python", "Docker", "SQL"])
    assert m.missing_required == ["Kubernetes"]
    assert m.required_coverage == 0.75
    # 0.55*75 + 0.15*0 + 0.30*80 = 65.25
    assert fit_score(m, 80) == 65
    assert score_band(65) == "Good match"
    assert fit_score(m, 500) == fit_score(m, 100)  # LLM judgement is clamped


def test_empty_requirements_count_as_full_coverage():
    m = match_skills([], [], ["Python"])
    assert m.required_coverage == 1.0 and m.preferred_coverage == 1.0
