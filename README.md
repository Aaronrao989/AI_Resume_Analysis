# 📄 LLM-Powered Resume Reviewer

> **An intelligent web application that leverages LLMs and ATS-style scoring to help candidates analyze and enhance their resumes.**

Upload your resume, select a target role, and optionally provide a job description—receive comprehensive section-wise feedback, keyword analysis, bullet point rewrites, and a tailored professional summary.

<div align="center">

**[🚀 Live Demo](https://llmresumeanalysis.streamlit.app/)**

</div>

-----


## ✨ Features

- **📂 Flexible Input** - Upload PDF resumes or paste raw text directly
- **🎯 Smart Role Detection** - Automatically predict the most relevant job role using ML classifier
- **📊 ATS Scoring** - Compute compatibility scores based on keywords and required skills

### 🤖 AI-Powered Resume Analysis

| Feature | Description |
|---------|-------------|
| **Section-wise Improvements** | Detailed feedback on Summary, Skills, Experience, Education, etc. |
| **Missing Skills Detection** | Identify gaps in skills and keywords tailored to your target role |
| **Bullet Point Enhancement** | Rewrite achievements in quantifiable STAR format |
| **Language Optimization** | Improve conciseness, clarity, and professional tone |
| **Formatting Suggestions** | Enhance readability and visual appeal |
| **Auto-generated Summary** | Create a compelling 3-line professional summary |

- **☁️ Cloud-Ready** - Easily deployable on Streamlit Cloud with `.env` API key support

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **ML Classifier** | Scikit-learn + Joblib (trained on resume datasets) |
| **LLM Backends** | OpenAI (GPT-4o-mini/GPT-4), Groq (LLaMA), Anthropic (Claude), Mistral |
| **ATS Scoring** | Keyword extraction + semantic matching |
| **PDF Parsing** | PyMuPDF / pdfminer |

---

## 📁 Project Structure

```
smart-resume-viewer/
│
├── app/
│   ├── app.py                    # Main Streamlit application
│   ├── components/
│   │   ├── llm_review.py         # LLM + ATS logic
│   │   ├── ats_scoring.py        # ATS score calculator
│   │   ├── resume_parser.py      # PDF text extraction
│   │   └── utils.py              # Helper functions
│   └── .env                      # API keys & configurations
│
├── artifacts/                    # Trained ML artifacts
│   ├── vectorizer.pkl
│   ├── role_match_clf.pkl
│   ├── X_dense.npy
│   └── y_positions.npy
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/llm-resume-reviewer.git
cd llm-resume-reviewer/app
```

### 2️⃣ Set Up Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file inside the `app/` directory:

```env
OPENAI_API_KEY=your_openai_api_key
MODEL_BACKEND=openai
MODEL_NAME=gpt-4o-mini
```

> **💡 Tip:** You can switch backends to `groq`, `anthropic`, or `mistral` by updating `MODEL_BACKEND`

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## 📊 Dataset Format

Your CSV must be UTF-8 encoded with at least these columns:

- `job_position`
- `relevant_skills` (comma or pipe-separated)
- `required_qualifications`
- `job_responsibilities`
- `ideal_candidate_summary`

**Default location:** `./data/jobs.csv` (update paths as needed)

---

## 📦 Handling Large Model Artifacts via Google Drive

### Why Google Drive?

The trained models and embeddings exceed 500MB, which can cause:
- Repository bloat and slow cloning
- Deployment issues on Streamlit Cloud
- Storage limit problems

### SOLUTION OVERVIEW

By hosting artifacts on Google Drive and downloading them at runtime, we achieve:

✅ **Lightweight repository** - No large binaries in version control  
✅ **Seamless deployment** - Avoids storage limits on hosting platforms  
✅ **Easy updates** - Modify artifacts without changing code  
✅ **Reproducibility** - Always fetch the latest artifact versions  

### Implementation Steps

#### Step 1: Upload Artifacts to Google Drive

Upload these files to a Google Drive folder:
- `vectorizer.pkl`
- `role_match_clf.pkl`
- `X_dense.npy`
- `y_positions.npy`

Compress them into `artifacts.zip` and get a shareable link with **"Anyone with the link can view"** permission.

#### Step 2: Add Download Logic

In `llm_review.py`, add the following code:

```python
import os
import gdown
import zipfile

ZIP_URL = "YOUR_GOOGLE_DRIVE_ZIP_LINK"
ZIP_PATH = "artifacts.zip"
ART_DIR = "artifacts"

if not os.path.exists(ART_DIR):
    gdown.download(ZIP_URL, ZIP_PATH, quiet=False)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(".")  # Extract into current directory
```

#### Step 3: Load Models as Usual

After extraction, load models from the local `artifacts/` folder normally.

---

## 🌐 Deployment

The application is live on Streamlit Cloud:

**[🔗 llmresumeanalysis.streamlit.app](https://llmresumeanalysis.streamlit.app/)**

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Link your GitHub repository

3. **Configure Secrets**
   - Navigate to App Settings → Secrets
   - Add your `OPENAI_API_KEY` and other environment variables

4. **Deploy**
   - Click "Deploy" and wait for the build to complete

---

## 📸 Screenshots

<div align="center">

![Application Screenshot](https://github.com/user-attachments/assets/08b41d9b-54bd-4829-90ef-031601afc41b)

*Resume analysis interface with feedback and ATS scoring*

</div>

---

## 🤝 Contributing

We welcome contributions from the community!

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution

- Additional LLM provider integrations
- Enhanced ATS scoring algorithms
- UI/UX improvements
- Documentation enhancements
- Bug fixes and performance optimizations

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📧 Support

If you encounter any issues or have questions:

- **Open an issue** on GitHub
- **Check existing issues** for solutions
- **Review the documentation** thoroughly

---

<div align="center">

**Made with ❤️ by developers, for developers**

⭐ Star this repo if you find it helpful!

</div>
