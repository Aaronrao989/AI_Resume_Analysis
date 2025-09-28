import os, json, re
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# --- Load .env explicitly ---
APP_DIR = os.path.dirname(os.path.dirname(__file__))  # .../app
DOTENV_PATH = os.path.join(APP_DIR, ".env")
if os.path.exists(DOTENV_PATH):
    load_dotenv(DOTENV_PATH)
else:
    print(f"⚠️ Warning: .env file not found at {DOTENV_PATH}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ Warning: OPENAI_API_KEY not found in .env! LLM calls may fail.")

# --- Artifact paths ---
PROJECT_ROOT = os.path.dirname(APP_DIR)  # .../smart resume viewer
ART_DIR = os.path.join(PROJECT_ROOT, "artifacts")

VECTOR_PATH = os.path.join(ART_DIR, "vectorizer.pkl")
CLF_PATH = os.path.join(ART_DIR, "role_match_clf.pkl")
X_DENSE_PATH = os.path.join(ART_DIR, "X_dense.npy")
Y_POSITIONS_PATH = os.path.join(ART_DIR, "y_positions.npy")

# --- Load trained classifier & embeddings ---
vectorizer = joblib.load(VECTOR_PATH)
clf = joblib.load(CLF_PATH)
X_dense = np.load(X_DENSE_PATH, allow_pickle=False)       # numeric embeddings
y_positions = np.load(Y_POSITIONS_PATH, allow_pickle=True)  # object array of labels

# --- Utilities ---
from .utils import load_json, clean_text
from .ats_scoring import ats_score


# ---------------- BACKEND SETUP ----------------
def choose_backend():
    """Decide which backend + model to use based on .env variables."""
    backend = st.secrets.get("MODEL_BACKEND", "groq").lower()
    name = st.secrets.get("MODEL_NAME", "llama-3.1-8b-instant")
    return backend, name

def call_llm(prompt: str, system: Optional[str] = None) -> str:
    backend, model = choose_backend()

    # --- OpenAI ---
    if backend == "openai":
        from openai import OpenAI
        if not OPENAI_API_KEY:
            return "[ERROR] OPENAI_API_KEY not set. Cannot call LLM."
        client = OpenAI(api_key=OPENAI_API_KEY)
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.2)
        return resp.choices[0].message.content.strip()

    # --- Groq ---
    elif backend == "groq":
        from groq import Groq
        # GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
        MODEL_BACKEND = st.secrets.get("MODEL_BACKEND", "groq")
        MODEL_NAME = st.secrets.get("MODEL_NAME", "llama-3.1-8b-instant")
        if not GROQ_API_KEY:
            return "[ERROR] GROQ_API_KEY not set. Cannot call Groq LLM."
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    
    # --- Anthropic ---
    elif backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        sysmsg = system or "You are a helpful assistant."
        msg = client.messages.create(
            model=model,
            max_tokens=1200,
            system=sysmsg,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return msg.content[0].text.strip()

    # --- Mistral ---
    elif backend == "mistral":
        from mistralai import Mistral
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        sysmsg = system or "You are a helpful assistant."
        chat_response = client.chat.complete(
            model=model,
            messages=[{"role": "system", "content": sysmsg}, {"role": "user", "content": prompt}],
            temperature=0.2
        )
        return chat_response.choices[0].message.content.strip()

    # --- Fallback ---
    else:
        return "[LLM not configured]\n" + prompt[:2000]


# ---------------- PREDICTION + PROMPT BUILDING ----------------
def predict_role(text: str) -> str:
    """Predict the candidate's role using the trained classifier."""
    X = vectorizer.transform([text])
    pred = clf.predict(X)[0]
    return pred


def build_prompt(resume_text: str, job_role: str, guidance_blobs: List[str], jd_text: str = "") -> str:
    guidance = "\n\n".join(guidance_blobs[:3])
    template = f"""You are an expert resume reviewer.
Target Role: {job_role}
Optional Job Description (JD):
{jd_text[:2000]}

Domain Guidance (from internal knowledge base):
{guidance}

TASKS:
1) Give section-wise feedback (Summary, Experience, Education, Skills, Projects, Certifications).
2) List *missing* skills/keywords for this role (high priority).
3) Rewrite 3-5 bullets to be quantifiable and tailored to the JD. Use STAR actions when possible.
4) Flag vague or redundant language and suggest concise alternatives.
5) Suggest formatting/clarity improvements.
6) Provide a brief 3-line profile summary for the candidate tailored to the role.

Resume Text:
{resume_text[:6000]}

Return JSON with keys: feedback_by_section, missing_keywords, bullet_rewrites, language_fixes, formatting_suggestions, tailored_summary.
"""
    return template


# ---------------- MAIN REVIEW FUNCTION ----------------
def review_resume(resume_text: str, guidance_blobs: List[str], jd_text: str = "") -> Dict[str, Any]:
    job_role = predict_role(resume_text)
    prompt = build_prompt(resume_text, job_role, guidance_blobs, jd_text)
    system = f"You are a meticulous ATS-savvy resume coach for {job_role}."
    llm_json = call_llm(prompt, system=system)

    # Extract skills for ATS scoring
    required_skills = []
    for blob in guidance_blobs:
        required_skills += re.findall(r"\b[A-Za-z0-9\+\#\-/]+\b", blob)
    required_skills = list(set(required_skills))

    score, detail = ats_score(resume_text + "\n" + jd_text, required_skills)

    return {
        "predicted_role": job_role,
        "ats": {"score": score, "detail": detail},
        "llm_feedback_raw": llm_json
    }
