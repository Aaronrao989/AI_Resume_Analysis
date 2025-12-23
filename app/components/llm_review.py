import os, json, re
from typing import Dict, Any, List, Optional
import numpy as np
import joblib
import streamlit as st
from dotenv import load_dotenv

from .ats_scoring import ats_score, detect_sections
from .utils import clean_text
from .utils import normalize_token  # added in STEP 2 later (safe import)
from .utils import load_json

# -------------------------------------------------
# ENV + ARTIFACT SETUP
# -------------------------------------------------

APP_DIR = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"))

ART_DIR = os.path.join(APP_DIR, "../artifacts")

VECTOR_PATH = os.path.join(ART_DIR, "vectorizer.pkl")
CLF_PATH = os.path.join(ART_DIR, "role_match_clf.pkl")
FAISS_META_PATH = os.path.join(ART_DIR, "faiss_meta.json")

# -------------------------------------------------
# LOAD ARTIFACTS (DEFENSIVE)
# -------------------------------------------------

def _load_artifacts():
    vectorizer = clf = None
    meta = []
    try:
        vectorizer = joblib.load(VECTOR_PATH)
    except Exception:
        pass
    try:
        clf = joblib.load(CLF_PATH)
    except Exception:
        pass
    try:
        meta = load_json(FAISS_META_PATH, default=[]) or []
    except Exception:
        meta = []
    return vectorizer, clf, meta


vectorizer, clf, faiss_meta = _load_artifacts()

# -------------------------------------------------
# BACKEND SETUP (GROQ ENABLED)
# -------------------------------------------------

def choose_backend():
    backend = os.getenv("MODEL_BACKEND", "groq").lower()
    model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    return backend, model


def call_llm(prompt: str, system: Optional[str] = None) -> str:
    backend, model = choose_backend()

    if backend == "groq":
        try:
            from groq import Groq
        except Exception:
            return "[LLM not configured] groq package missing"

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "[LLM not configured] GROQ_API_KEY missing"

        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    return "[LLM not configured] unsupported backend"


# -------------------------------------------------
# ROLE LOGIC
# -------------------------------------------------

def predict_role(text: str) -> Optional[str]:
    if vectorizer is None or clf is None:
        return None
    try:
        X = vectorizer.transform([text])
        return clf.predict(X)[0]
    except Exception:
        return None


def get_role_meta(role: str, meta: List[dict]) -> Optional[dict]:
    role_norm = role.lower().strip()
    for m in meta:
        if m.get("job_position", "").lower().strip() == role_norm:
            return m
    return None


# -------------------------------------------------
# PROMPT BUILDER
# -------------------------------------------------

def build_prompt(
    resume_text: str,
    target_role: str,
    ml_role: Optional[str],
    guidance_blobs: List[str],
    jd_text: str,
) -> str:

    guidance = "\n\n".join(guidance_blobs[:2])

    role_note = ""
    if ml_role and ml_role.lower() != target_role.lower():
        role_note = f"\n(ML predicted role: {ml_role})"

    return f"""
You are an expert ATS-aware resume reviewer.

Target Role: {target_role}{role_note}

Optional Job Description:
{jd_text[:2000]}

Role Expectations (internal knowledge base):
{guidance}

TASKS:
1) Give section-wise feedback (Summary, Experience, Education, Skills, Projects, Certifications).
2) List missing skills/keywords for this role.
3) Rewrite 3-5 bullets to be quantifiable and role-specific.
4) Flag vague or redundant language.
5) Suggest formatting improvements.
6) Provide a concise 3-line tailored summary.

Resume:
{resume_text[:6000]}

Return STRICT JSON with keys:
feedback_by_section, missing_keywords, bullet_rewrites, tailored_summary
"""


# -------------------------------------------------
# MAIN REVIEW FUNCTION
# -------------------------------------------------

def review_resume(
    resume_text: str,
    guidance_blobs: List[str],
    jd_text: str = "",
    job_role: Optional[str] = None,
) -> Dict[str, Any]:

    resume_text = clean_text(resume_text)

    ml_role = predict_role(resume_text)
    target_role = job_role or ml_role or "Software Engineer"

    # ---- FAISS META RESOLUTION ----
    role_meta = get_role_meta(target_role, faiss_meta)

    if role_meta:
        guidance_blobs = [role_meta.get("text", "")]
        required_skills = role_meta.get("skills", [])
    else:
        required_skills = []

    # ---- ATS SCORE ----
    ats_score_raw, ats_detail = ats_score(
        resume_text + "\n" + jd_text,
        required_skills,
    )

    ats_score_final = min(100.0, float(ats_score_raw))

    # ---- LLM CALL ----
    prompt = build_prompt(
        resume_text=resume_text,
        target_role=target_role,
        ml_role=ml_role,
        guidance_blobs=guidance_blobs,
        jd_text=jd_text,
    )

    system = f"You are a meticulous resume coach for {target_role}."
    llm_output = call_llm(prompt, system=system)

    llm_used = not llm_output.startswith("[LLM not configured]")

    # ---- FALLBACK (ONLY IF NEEDED) ----
    if not llm_used:
        sections = detect_sections(resume_text)
        feedback = {
            k: (
                "Present. Add quantified impact."
                if v else
                "Missing or weak. Add a concise section."
            )
            for k, v in sections.items()
        }

        missing = []
        resume_norm = normalize_token(resume_text)
        for s in required_skills:
            if normalize_token(s) not in resume_norm:
                missing.append(s)

        llm_output = json.dumps({
            "feedback_by_section": feedback,
            "missing_keywords": missing,
            "bullet_rewrites": [
                f"Led development of {target_role} APIs, improving reliability and performance with measurable impact."
            ],
            "tailored_summary": f"{target_role} with experience building scalable backend systems and APIs."
        })

    return {
        "predicted_role": ml_role,
        "target_role": target_role,
        "llm_used": llm_used,
        "ats": {
            "score": ats_score_final,
            "detail": ats_detail,
        },
        "llm_feedback_raw": llm_output,
    }
