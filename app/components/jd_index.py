import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .utils import clean_text, split_csv_list, save_json, load_json, normalize_token


# -------------------------------------------------
# ARTIFACT DIRECTORY
# -------------------------------------------------

ART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)


# -------------------------------------------------
# JD INDEX CLASS
# -------------------------------------------------

class JDIndex:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model
        self.model = None
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.role_match_clf = None

    # -------------------------------------------------
    # EMBEDDING (LAZY LOAD)
    # -------------------------------------------------

    def load_embedder(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.embed_model_name)

    def _embed(self, texts: List[str]) -> np.ndarray:
        self.load_embedder()
        embs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embs, dtype="float32")

    # -------------------------------------------------
    # BUILD FROM CSV
    # -------------------------------------------------

    def build_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path, encoding="utf-8")
        df.fillna("", inplace=True)

        records = []
        skills_vocab = set()

        for _, r in df.iterrows():
            job = clean_text(str(r.get("job_position", "")))
            job_norm = normalize_token(job)

            skills_raw = split_csv_list(str(r.get("relevant_skills", "")))
            skills = []
            for s in skills_raw:
                skills.append(s)
                skills_vocab.add(normalize_token(s))

            quals = clean_text(str(r.get("required_qualifications", "")))
            resp = clean_text(str(r.get("job_responsibilities", "")))
            ideal = clean_text(str(r.get("ideal_candidate_summary", "")))

            blob = (
                f"Job Position: {job}\n"
                f"Skills: {', '.join(skills)}\n"
                f"Qualifications: {quals}\n"
                f"Responsibilities: {resp}\n"
                f"Summary: {ideal}"
            )

            records.append({
                "job_position": job,
                "job_position_norm": job_norm,
                "skills": skills,
                "skills_norm": [normalize_token(s) for s in skills],
                "text": blob,
            })

        # ---------- FAISS INDEX ----------
        texts = [rec["text"] for rec in records]
        embs = self._embed(texts)
        dim = embs.shape[1]

        import faiss
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)

        self.meta = records

        faiss.write_index(self.index, os.path.join(ART_DIR, "faiss_index.bin"))
        save_json(os.path.join(ART_DIR, "faiss_meta.json"), self.meta)

        # ---------- TF-IDF ROLE MATCHER ----------
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            min_df=1,
        )

        X = self.vectorizer.fit_transform(texts)
        y = [rec["job_position"] for rec in records]

        self.role_match_clf = LogisticRegression(max_iter=300)
        self.role_match_clf.fit(X, y)

        joblib.dump(self.vectorizer, os.path.join(ART_DIR, "vectorizer.pkl"))
        joblib.dump(self.role_match_clf, os.path.join(ART_DIR, "role_match_clf.pkl"))

        save_json(os.path.join(ART_DIR, "skills_vocab.json"), sorted(skills_vocab))

        # ---------- ROLE PROMPTS ----------
        prompts = {}
        for rec in records:
            jp = rec["job_position"]
            prompts[jp] = {
                "system": "You are an expert resume reviewer for this role.",
                "instruction": (
                    f"Focus on {jp}. "
                    "Tailor feedback using concrete bullet rewrites and missing skills."
                ),
            }

        save_json(os.path.join(ART_DIR, "role_prompts.json"), prompts)

    # -------------------------------------------------
    # LOAD ARTIFACTS
    # -------------------------------------------------

    def load(self):
        import faiss

        self.index = faiss.read_index(os.path.join(ART_DIR, "faiss_index.bin"))
        self.meta = load_json(os.path.join(ART_DIR, "faiss_meta.json"), default=[])

        self.vectorizer = joblib.load(os.path.join(ART_DIR, "vectorizer.pkl"))
        self.role_match_clf = joblib.load(os.path.join(ART_DIR, "role_match_clf.pkl"))

    # -------------------------------------------------
    # FAISS QUERY
    # -------------------------------------------------

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded")

        embs = self._embed([text])
        sims, ids = self.index.search(embs, k)

        results = []
        for score, idx in zip(sims[0], ids[0]):
            if idx == -1:
                continue
            m = self.meta[idx].copy()
            m["score"] = float(score)
            results.append(m)

        return results

    # -------------------------------------------------
    # ROLE MATCHING (ROBUST)
    # -------------------------------------------------

    def match_role(self, text: str) -> Tuple[str, float]:
        if self.vectorizer is None or self.role_match_clf is None:
            raise RuntimeError("Role classifier not loaded")

        X = self.vectorizer.transform([text])
        proba = self.role_match_clf.predict_proba(X)[0]
        classes = list(self.role_match_clf.classes_)

        idx = int(np.argmax(proba))
        return classes[idx], float(proba[idx])
