# LLM-Powered Resume Reviewer (with ATS-style Scoring)

An interactive web app that helps candidates analyze and improve resumes using LLMs + ATS scoring.
Upload your resume, choose a target role, and optionally paste a job description — the app will provide section-wise feedback, missing keywords, bullet rewrites, and a tailored professional summary.

👉 Live Demo: llmresumeanalysis.streamlit.app
---
✨ Features

📂 Upload a PDF resume or paste raw text.
🎯 Predict the most relevant job role automatically using ML classifier.
📊 Compute ATS score based on keywords & required skills.
🤖 LLM-powered resume feedback:
1)Section-wise improvements (Summary, Skills, Experience, Education, etc.)
2)Missing skills/keywords tailored to the role
3)Rewritten, quantifiable bullet points (STAR format)
4)Language fixes (conciseness & clarity)
5)Formatting & readability suggestions
6)Auto-generated 3-line tailored summary
☁️ Deployable on Streamlit Cloud with .env API key support.

---
🛠️ Tech Stack

Frontend: Streamlit
ML Role Classifier: Scikit-learn + Joblib (trained on resume datasets)
LLM Backends Supported:
OpenAI (gpt-4o-mini / gpt-4)
Groq (latest LLaMA models)
Anthropic (Claude models)
Mistral (Open source LLMs)
ATS Scoring: Keyword extraction + semantic match
PDF Parsing: PyMuPDF / pdfminer

---
## Project Structure
```
smart-resume-viewer/
│── app/
│   │── app.py                 # Main Streamlit app
│   │── components/
│   │   │── llm_review.py       # LLM + ATS logic
│   │   │── ats_scoring.py      # ATS score calculator
│   │   │── resume_parser.py    # Extract text from PDF
│   │   │── utils.py            # Helper functions
│   │── .env                    # API keys & configs
│
│── artifacts/                  # Trained ML artifacts
│   │── vectorizer.pkl
│   │── role_match_clf.pkl
│   │── X_dense.npy
│   │── y_positions.npy
│
│── requirements.txt
│── README.md
```

---
🚀 Getting Started (Local Setup)
1️⃣ Clone the repo
git clone https://github.com/<your-username>/llm-resume-reviewer.git
cd llm-resume-reviewer/app

---

2️⃣ Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

---

pip install -r requirements.txt

3️⃣ Setup environment variables
Create a .env file inside app/:
OPENAI_API_KEY=your_openai_api_key
MODEL_BACKEND=openai
MODEL_NAME=gpt-4o-mini

---

4️⃣ Run locally
streamlit run app.py

---

🔑 You can switch backend to groq, anthropic, or mistral by updating MODEL_BACKEND.
## Dataset Format
Your CSV must be UTF-8 with at least these columns:
- `job_position`
- `relevant_skills` (comma or pipe-separated)
- `required_qualifications`
- `job_responsibilities`
- `ideal_candidate_summary`

Place your CSV at `./data/jobs.csv` (or update paths accordingly).

---

🌐 Deployment
The app is deployed on Streamlit Cloud:
👉 llmresumeanalysis.streamlit.app

Steps to deploy:
Push your repo to GitHub.
Connect repo to Streamlit Cloud
Add your OPENAI_API_KEY in Streamlit Cloud Secrets Manager.

📸 Screenshots
<img width="3420" height="1976" alt="image" src="https://github.com/user-attachments/assets/08b41d9b-54bd-4829-90ef-031601afc41b" />

Feedback & ATS Score
Deploy & share!

🤝 Contributing
Contributions are welcome!
Fork the repo
Create a feature branch
Submit a PR

📜 License
This project is licensed under the MIT License.
