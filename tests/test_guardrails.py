from job_hunter.guardrails import clean_letter, find_cliches, gap_claims, unsupported_claims

RESUME = "Cut latency by 63% for services handling 2M+ requests/day. Raised coverage from 41% to 78%."
JOB = "Used by 120+ warehouses."


def test_real_figures_pass():
    letter = "I cut latency by 63% on services handling 2M+ requests, and our tools serve your 120+ warehouses."
    assert unsupported_claims(letter, RESUME, JOB) == []


def test_invented_metric_is_flagged():
    issues = unsupported_claims("My chatbot increased engagement by 25%.", RESUME, JOB)
    assert len(issues) == 1 and "25%" in issues[0]


def test_placeholder_employer_and_brackets_are_flagged():
    issues = unsupported_claims("At XYZ Corporation I led [Project Name].", RESUME)
    assert any("XYZ Corporation" in i for i in issues)
    assert any("[Project Name]" in i for i in issues)


def test_each_bad_number_reported_once():
    issues = unsupported_claims("25% here and 25% there", RESUME)
    assert len(issues) == 1


def test_cliches_detected():
    assert find_cliches("I'm thrilled and passionate about your innovative team") == ["thrilled", "passionate", "innovative"]
    assert find_cliches("I cut latency by 63%.") == []


def test_gap_claims_allow_learning_language():
    missing = ["Kubernetes", "LLM evaluation"]
    assert gap_claims("I am currently learning Kubernetes on Minikube.", missing) == []
    assert len(gap_claims("I have deep Kubernetes expertise.", missing)) == 1


def test_clean_letter_strips_prompt_artifacts():
    assert clean_letter("Here is the cover letter: Cut latency by 63% (#1).") == "Cut latency by 63%."
