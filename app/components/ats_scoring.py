from typing import Dict, List, Tuple
import re
import textstat

from .utils import normalize_token


# -------------------------------------------------
# SECTION HINTS
# -------------------------------------------------

SECTION_HINTS = {
    "summary": ["summary", "profile", "objective", "about"],
    "experience": ["experience", "work", "employment", "professional", "internship"],
    "projects": ["projects", "project work", "academic projects", "personal projects"],
    "skills": ["skills", "technical skills", "tools", "technologies", "stack"],
    "education": ["education", "qualifications", "academics", "b.tech", "bachelor", "master", "university", "college"],
    "certifications": ["certifications", "courses", "licenses"],
    "achievements": ["achievements", "awards", "honors"],
}


# -------------------------------------------------
# SECTION DETECTION
# -------------------------------------------------

def detect_sections(text: str) -> Dict[str, bool]:
    if not text:
        return {k: False for k in SECTION_HINTS}

    text_lower = text.lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    presence = {}

    for sec, hints in SECTION_HINTS.items():
        found = False

        # 1) Explicit headers near top
        for ln in lines[:60]:
            ln_norm = ln.lower().rstrip(":")
            if ln_norm == sec or ln_norm in hints:
                found = True
                break

        # 2) Keyword presence fallback
        if not found:
            for h in hints:
                if re.search(r"\b" + re.escape(h) + r"\b", text_lower):
                    found = True
                    break

        # 3) Heuristic fallbacks
        if not found:
            if sec == "experience":
                if re.search(r"\b(develop|built|worked|engineer|implemented|\d+\s+years?)\b", text_lower):
                    found = True
            elif sec == "education":
                if re.search(r"\b(bachelor|master|degree|gpa|percentage)\b", text_lower):
                    found = True
            elif sec == "skills":
                tokens = re.findall(r"[A-Za-z0-9\+\#\./-]{2,}", text)
                if len(set(tokens)) > 8:
                    found = True

        presence[sec] = found

    return presence


# -------------------------------------------------
# KEYWORD MATCHING
# -------------------------------------------------

def keyword_match_rate(text: str, skills: List[str]) -> float:
    if not text or not skills:
        return 0.0

    text_norm = normalize_token(text)
    skills_norm = {normalize_token(s) for s in skills if s}

    if not skills_norm:
        return 0.0

    matched = sum(1 for s in skills_norm if s and s in text_norm)
    return matched / len(skills_norm)


# -------------------------------------------------
# QUANTIFICATION SIGNAL
# -------------------------------------------------

def quantify_bullets_ratio(text: str) -> float:
    if not text:
        return 0.0

    nums = len(re.findall(r"\b\d+(\.\d+)?%?\b", text))
    bullets = len(re.findall(r"(^\s*[-•*])", text, flags=re.MULTILINE))
    sentences = max(1, len(re.findall(r"[.!?]", text)))

    quantified = nums + bullets
    return min(1.0, quantified / (sentences * 0.6))


# -------------------------------------------------
# FORMATTING HEURISTICS
# -------------------------------------------------

def formatting_penalty(text: str) -> float:
    if not text:
        return 0.0

    words = text.split()
    all_caps = len(re.findall(r"\b[A-Z]{3,}\b", text))
    long_sentences = sum(
        1 for s in re.split(r"[.!?]", text) if len(s.split()) > 35
    )

    caps_ratio = all_caps / max(1, len(words))
    long_ratio = long_sentences / max(1, len(re.split(r"[.!?]", text)))

    penalty = (caps_ratio * 0.15) + (long_ratio * 0.15)
    return min(0.15, penalty)


# -------------------------------------------------
# READABILITY
# -------------------------------------------------

def readability_score(text: str) -> float:
    try:
        score = textstat.flesch_reading_ease(text)
        return max(0.0, min(100.0, score))
    except Exception:
        return 50.0


# -------------------------------------------------
# MAIN ATS SCORING
# -------------------------------------------------

def ats_score(text: str, required_skills: List[str]) -> Tuple[float, Dict]:
    if not text:
        return 0.0, {"error": "Empty resume text"}

    # ---- Signals ----
    sections = detect_sections(text)
    coverage = sum(1 for v in sections.values() if v) / len(sections)

    keyword_rate = keyword_match_rate(text, required_skills)
    quantify = quantify_bullets_ratio(text)

    read = readability_score(text)
    read_norm = max(0.0, min(1.0, (read - 30) / 70))

    penalty = formatting_penalty(text)

    # ---- Weighted score (0..1) ----
    score_01 = (
        0.35 * keyword_rate +
        0.25 * coverage +
        0.20 * quantify +
        0.20 * read_norm
    )

    score_01 = max(0.0, min(1.0, score_01 - penalty))
    score_100 = round(score_01 * 100, 1)

    # ---- Detailed breakdown ----
    detail = {
        "sections_detected": sections,
        "section_coverage": round(coverage, 3),
        "keyword_match_rate": round(keyword_rate, 3),
        "quantification_signal": round(quantify, 3),
        "readability": round(read, 1),
        "formatting_penalty": round(penalty, 3),
    }

    return score_100, detail
