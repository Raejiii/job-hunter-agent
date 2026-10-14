"""Deterministic hallucination checks for generated text.

Small local models happily invent metrics ("boosted engagement by 25%") or placeholder
employers ("XYZ Corporation"). Every figure in the cover letter must be traceable to the
resume or the job description; anything else is flagged and sent back to the writer.
"""

import re

NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%|\+|x\b|k\b|m\b|lpa\b)?", re.I)
PLACEHOLDER_ORG = re.compile(r"\b(XYZ|ABC|Acme|Company Name|Your Company)\b(?:\s+(?:Corp\w*|Inc|Ltd|Pvt))?", re.I)
PLACEHOLDER_BRACKETS = re.compile(r"\[[^\]]{1,40}\]")


def _numbers(text: str) -> set[str]:
    return {m.group(1).replace(",", "") for m in NUMBER.finditer(text)}


def unsupported_claims(letter: str, *sources: str) -> list[str]:
    """Return human-readable descriptions of claims in `letter` not backed by `sources`."""
    known = set().union(*(_numbers(s) for s in sources))
    issues: list[str] = []
    seen: set[str] = set()
    for m in NUMBER.finditer(letter):
        value = m.group(1).replace(",", "")
        if value in known or value in seen:
            continue
        seen.add(value)
        issues.append(f"The figure '{m.group(0).strip()}' does not appear in the resume; remove it or use a real number.")
    for m in PLACEHOLDER_ORG.finditer(letter):
        issues.append(f"'{m.group(0)}' is an invented placeholder organisation; use the real employer from the resume.")
    for m in PLACEHOLDER_BRACKETS.finditer(letter):
        issues.append(f"Remove the placeholder '{m.group(0)}'.")
    return issues


CLICHES = [
    "i am writing to", "i'm writing to", "thrilled", "passionate", "passion for", "team player",
    "innovative", "excited to", "i'm excited", "i am excited", "dynamic", "go-getter", "think outside the box",
    "perfect fit", "ideal candidate", "hard-working", "hardworking", "synergy", "exciting projects",
]
LEARNING_CUES = re.compile(r"\b(learn\w*|develop\w*|grow\w*|explor\w*|build\w* up|currently|studying|course|"
                           r"working towards|gap|new to|areas? (?:i'd|i would|to))\b", re.I)


def find_cliches(letter: str) -> list[str]:
    text = letter.lower()
    return [c for c in CLICHES if c in text]


def gap_claims(letter: str, missing_skills: list[str]) -> list[str]:
    """Sentences that mention a skill the candidate lacks without framing it as something they're learning."""
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+", letter):
        for skill in missing_skills:
            name = re.sub(r"\s*\(.*?\)", "", skill).strip()
            if name and re.search(rf"\b{re.escape(name)}\b", sentence, re.I) and not LEARNING_CUES.search(sentence):
                claims.append(f"'{sentence.strip()[:120]}' implies experience with {name}, which the resume lacks.")
    return claims


def clean_letter(letter: str) -> str:
    """Remove artifacts small models leave behind, e.g. '(#1)' references to the prompt's bullet list."""
    letter = re.sub(r"\s*\((?:#|achievement\s*)\d+\)", "", letter, flags=re.I)
    letter = re.sub(r"^(?:here is|here's) (?:the|a|my) (?:revised )?(?:cover )?letter:?\s*", "", letter, flags=re.I)
    return letter.strip()
