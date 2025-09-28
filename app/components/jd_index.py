import os, json, re
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

from .utils import split_csv_list, save_json, load_json, clean_text

ART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

class JDIndex:
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model
        self.model = None
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self.vectorizer = None
        self.role_match_clf = None

    def load_embedder(self):
        if self.model is None:
            self.model = SentenceTransformer(self.embed_model_name)

    def _embed(self, texts: List[str]) -> np.ndarray:
        self.load_embedder()
        return np.array(self.model.encode(texts, normalize_embeddings=True))

    def build_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path, encoding="utf-8")
        df.fillna("", inplace=True)

        records = []
        skills_vocab = set()

        for _, r in df.iterrows():
            job = clean_text(str(r.get("job_position","")))
            skills = split_csv_list(str(r.get("relevant_skills","")))
            quals = clean_text(str(r.get("required_qualifications","")))
            resp = clean_text(str(r.get("job_responsibilities","")))
            ideal = clean_text(str(r.get("ideal_candidate_summary","")))

            skills_vocab.update([s.lower() for s in skills])

            blob = f"Job Position: {job}\nSkills: {', '.join(skills)}\nQualifications: {quals}\nResponsibilities: {resp}\nSummary: {ideal}"
            records.append({
                "job_position": job,
                "skills": skills,
                "text": blob,
            })

        # Build FAISS index
        texts = [rec["text"] for rec in records]
        embs = self._embed(texts).astype("float32")
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)
        self.meta = records

        # Save FAISS + meta
        faiss.write_index(self.index, os.path.join(ART_DIR, "faiss_index.bin"))
        with open(os.path.join(ART_DIR, "faiss_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

        # Build TF-IDF matcher (job_position)
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=1)
        X = self.vectorizer.fit_transform([rec["text"] for rec in records])
        y = [rec["job_position"] for rec in records]
        self.role_match_clf = LogisticRegression(max_iter=200).fit(X, y)

        joblib.dump(self.vectorizer, os.path.join(ART_DIR, "tfidf_vectorizer.pkl"))
        joblib.dump(self.role_match_clf, os.path.join(ART_DIR, "tfidf_job_match.pkl"))

        # Save skills vocab
        save_json(os.path.join(ART_DIR, "skills_vocab.json"), sorted(list(skills_vocab)))

        # role prompts
        prompts = {}
        for rec in records:
            jp = rec["job_position"]
            prompts[jp] = {
                "system": "You are an expert resume reviewer for the role.",
                "instruction": f"Focus on {jp} expectations. Tailor recommendations with concrete bullet rewrites and keywords.",
            }
        save_json(os.path.join(ART_DIR, "role_prompts.json"), prompts)

    def load(self):
        # load FAISS + meta
        self.index = faiss.read_index(os.path.join(ART_DIR, "faiss_index.bin"))
        with open(os.path.join(ART_DIR, "faiss_meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # load tfidf matcher
        import joblib
        self.vectorizer = joblib.load(os.path.join(ART_DIR, "tfidf_vectorizer.pkl"))
        self.role_match_clf = joblib.load(os.path.join(ART_DIR, "tfidf_job_match.pkl"))

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        embs = self._embed([text]).astype("float32")
        sims, ids = self.index.search(embs, k)
        results = []
        for score, idx in zip(sims[0], ids[0]):
            if idx == -1: continue
            m = self.meta[idx].copy()
            m["score"] = float(score)
            results.append(m)
        return results

    def match_role(self, text: str) -> Tuple[str, float]:
        X = self.vectorizer.transform([text])
        proba = self.role_match_clf.predict_proba(X)[0]
        classes = list(self.role_match_clf.classes_)
        i = int(np.argmax(proba))
        return classes[i], float(proba[i])
