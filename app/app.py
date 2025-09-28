# import streamlit as st
# import os, json
# import pandas as pd
# from components.resume_parser import extract_text_from_pdf
# from components.utils import load_json, clean_text
# from components.llm_review import review_resume, ART_DIR
# import joblib
# import numpy as np

# st.set_page_config(page_title="Resume Reviewer", layout="wide")

# st.title("🔎 LLM-Powered Resume Reviewer (ATS-aware)")
# st.caption("Upload your resume, choose a target role, optionally paste a JD, and get tailored feedback.")

# # --- Load trained classifier to get all roles ---
# try:
#     CLF_PATH = os.path.join(ART_DIR, "role_match_clf.pkl")
#     clf = joblib.load(CLF_PATH)
#     roles = sorted(clf.classes_)
#     st.sidebar.success(f"Knowledge base loaded: {len(roles)} roles")
#     kb_loaded = True
# except Exception as e:
#     st.sidebar.error(f"Artifacts missing or cannot load classifier: {e}")
#     roles = []
#     kb_loaded = False

# # Optional: load FAISS meta for guidance blobs or skill hints
# try:
#     meta = load_json(os.path.join(ART_DIR, "faiss_meta.json"), default=[])
#     skills_vocab = load_json(os.path.join(ART_DIR, "skills_vocab.json"), default=[])
#     prompts = load_json(os.path.join(ART_DIR, "role_prompts.json"), default={})
# except Exception:
#     meta = []
#     skills_vocab = []
#     prompts = {}

# # --- Inputs ---
# colA, colB = st.columns(2)
# with colA:
#     job_role = st.selectbox(
#         "Target Job Role",
#         options=roles if roles else ["(Run training first)"]
#     )
#     jd_text = st.text_area(
#         "Paste Job Description (optional)",
#         height=160,
#         placeholder="Paste JD here..."
#     )

# with colB:
#     up = st.file_uploader(
#         "Upload Resume (PDF)",
#         type=["pdf"],
#         accept_multiple_files=False
#     )
#     resume_text = st.text_area(
#         "...or Paste Resume Text",
#         height=200,
#         placeholder="Paste resume here if not uploading PDF..."
#     )

# # --- Extract PDF if uploaded ---
# if up is not None:
#     tmp_path = os.path.join("/tmp", up.name)
#     with open(tmp_path, "wb") as f:
#         f.write(up.read())
#     text, pages = extract_text_from_pdf(tmp_path)
#     if text:
#         resume_text = text
#         st.success(f"Extracted text from PDF ({pages} pages). You can still edit below.")
#     else:
#         st.error("Could not extract text from PDF. Paste your resume text instead.")

# st.divider()

# # --- Review button ---
# if st.button("Review", type="primary"):
#     if not kb_loaded:
#         st.error("Artifacts not found. Please run the notebook to build the knowledge base.")
#     elif not resume_text.strip():
#         st.error("Provide resume text (upload or paste).")
#     else:
#         # Prepare guidance blobs & required skills from FAISS meta (optional)
#         guidance_blobs = []
#         required_skills = []
#         for m in meta:
#             if m.get("job_position") == job_role:
#                 guidance_blobs.append(m.get("text", ""))
#                 required_skills.extend(m.get("skills", []))

#         with st.spinner("Reviewing with LLM + ATS scoring..."):
#             resp = review_resume(
#                 resume_text=resume_text,
#                 guidance_blobs=guidance_blobs,
#                 jd_text=jd_text
#             )

#         # --- Display ATS score ---
#         st.subheader("ATS Score")
#         st.metric("Overall", f"{resp['ats']['score']:.1f}/100")
#         st.json(resp["ats"]["detail"], expanded=False)

#         # --- Display LLM feedback ---
#         st.subheader("LLM Feedback (Raw JSON)")
#         st.caption("This is the JSON returned by the model. You can post-process to pretty panels.")
#         st.code(resp["llm_feedback_raw"], language="json")

# st.sidebar.markdown("---")
# st.sidebar.info(
#     "**Privacy**: Files are processed in-memory. If you enable cloud LLMs via API keys, model providers may receive your text."
# )
import streamlit as st
import os, json
import pandas as pd
from components.resume_parser import extract_text_from_pdf
from components.utils import load_json, clean_text
from components.llm_review import review_resume, ART_DIR
import joblib
import numpy as np

# Page config with custom theme
st.set_page_config(
    page_title="Resume Reviewer", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# Resume Reviewer\nAI-powered resume analysis tool"
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        text-align: center;
        font-weight: 300;
    }
    
    /* Card styling */
    .info-card {
        background: #34495e;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #ecf0f1;
    }
    
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    
    .error-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    
    /* Input section styling */
    .input-section {
        background: #2c3e50;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #34495e;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .section-title {
        color: #ecf0f1;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Fix for input labels */
    .input-section strong {
        color: #ecf0f1 !important;
        font-weight: 600;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    /* Ensure all text in input sections is readable */
    .input-section * {
        color: #ecf0f1 !important;
    }
    
    /* Fix Streamlit default text colors */
    .stTextArea > label, .stSelectbox > label {
        color: #ecf0f1 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Results section */
    .results-section {
        background: #2c3e50;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        margin: 2rem 0;
        border-top: 4px solid #667eea;
        color: #ecf0f1;
    }
    
    /* Status indicators */
    .status-success {
        display: inline-flex;
        align-items: center;
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.5rem;
    }
    
    .status-error {
        display: inline-flex;
        align-items: center;
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.5rem;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Privacy notice */
    .privacy-notice {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 2rem;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Divider styling */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border: none;
        border-radius: 2px;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="main-header">
    <div class="main-title">🔍 AI Resume Reviewer</div>
    <div class="main-subtitle">Upload your resume, select a target role, and get instant AI-powered feedback with ATS scoring</div>
</div>
""", unsafe_allow_html=True)

# --- Load trained classifier to get all roles ---
try:
    CLF_PATH = os.path.join(ART_DIR, "role_match_clf.pkl")
    clf = joblib.load(CLF_PATH)
    roles = sorted(clf.classes_)
    st.sidebar.markdown(f"""
    <div class="status-success">
        ✅ Knowledge Base Loaded: {len(roles)} roles
    </div>
    """, unsafe_allow_html=True)
    kb_loaded = True
except Exception as e:
    st.sidebar.markdown(f"""
    <div class="status-error">
        ❌ Artifacts missing: {str(e)}
    </div>
    """, unsafe_allow_html=True)
    roles = []
    kb_loaded = False

# Optional: load FAISS meta for guidance blobs or skill hints
try:
    meta = load_json(os.path.join(ART_DIR, "faiss_meta.json"), default=[])
    skills_vocab = load_json(os.path.join(ART_DIR, "skills_vocab.json"), default=[])
    prompts = load_json(os.path.join(ART_DIR, "role_prompts.json"), default={})
except Exception:
    meta = []
    skills_vocab = []
    prompts = {}

# --- Enhanced Input Sections ---
st.markdown('<div class="section-title">📋 Job Configuration</div>', unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<strong style="color: #ecf0f1;">🎯 Target Role</strong>', unsafe_allow_html=True)
    job_role = st.selectbox(
        "Select your target job role",
        options=roles if roles else ["(Run training first)"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with colB:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<strong style="color: #2c3e50;">📄 Job Description (Optional)</strong>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Paste the job description for better analysis",
        height=120,
        placeholder="Paste job description here to get more targeted feedback...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

st.markdown('<div class="section-title">📤 Resume Upload</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<strong style="color: #2c3e50;">📁 Upload PDF Resume</strong>', unsafe_allow_html=True)
    up = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<strong style="color: #2c3e50;">✏️ Or Paste Resume Text</strong>', unsafe_allow_html=True)
    resume_text = st.text_area(
        "Paste your resume text here",
        height=180,
        placeholder="Paste your resume text here if not uploading a PDF...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# --- Extract PDF if uploaded ---
if up is not None:
    tmp_path = os.path.join("/tmp", up.name)
    with open(tmp_path, "wb") as f:
        f.write(up.read())
    text, pages = extract_text_from_pdf(tmp_path)
    if text:
        resume_text = text
        st.markdown(f"""
        <div class="success-card">
            ✅ Successfully extracted text from PDF ({pages} pages). You can still edit the text below if needed.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-card">
            ❌ Could not extract text from PDF. Please paste your resume text manually.
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# --- Enhanced Review button ---
col_center = st.columns([1, 2, 1])[1]
with col_center:
    review_clicked = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)

if review_clicked:
    if not kb_loaded:
        st.markdown("""
        <div class="error-card">
            ❌ Knowledge base not found. Please run the training notebook to build the knowledge base first.
        </div>
        """, unsafe_allow_html=True)
    elif not resume_text.strip():
        st.markdown("""
        <div class="error-card">
            ❌ Please provide your resume text (either upload a PDF or paste the text).
        </div>
        """, unsafe_allow_html=True)
    else:
        # Prepare guidance blobs & required skills from FAISS meta (optional)
        guidance_blobs = []
        required_skills = []
        for m in meta:
            if m.get("job_position") == job_role:
                guidance_blobs.append(m.get("text", ""))
                required_skills.extend(m.get("skills", []))

        with st.spinner("🤖 AI is analyzing your resume... This may take a few moments."):
            resp = review_resume(
                resume_text=resume_text,
                guidance_blobs=guidance_blobs,
                jd_text=jd_text
            )

        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        
        # --- Enhanced ATS Score Display ---
        st.markdown('<div class="section-title">📊 ATS Compatibility Score</div>', unsafe_allow_html=True)
        
        # Create metric card
        ats_score = resp['ats']['score']
        score_color = "#2ecc71" if ats_score >= 80 else "#f39c12" if ats_score >= 60 else "#e74c3c"
        
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, {score_color} 0%, {score_color}dd 100%);">
            <div class="metric-value">{ats_score:.1f}/100</div>
            <div class="metric-label">ATS Compatibility Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ATS Details
        with st.expander("📋 View Detailed ATS Analysis", expanded=False):
            st.json(resp["ats"]["detail"])

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # --- Enhanced LLM Feedback Display ---
        st.markdown('<div class="section-title">🤖 AI Feedback Analysis</div>', unsafe_allow_html=True)
        st.caption("Comprehensive AI-generated feedback on your resume")
        
        # Display in an expandable code block for better readability
        with st.expander("📝 View Complete AI Analysis", expanded=True):
            st.code(resp["llm_feedback_raw"], language="json")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="section-title">ℹ️ How It Works</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <strong>1. Upload Resume</strong><br>
        Upload your PDF resume or paste the text directly
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <strong>2. Select Target Role</strong><br>
        Choose the job role you're applying for
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <strong>3. Add Job Description</strong><br>
        Optionally paste the job description for targeted analysis
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <strong>4. Get AI Feedback</strong><br>
        Receive comprehensive feedback and ATS scoring
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Enhanced privacy notice
    st.markdown("""
    <div class="privacy-notice">
        <strong>🔒 Privacy Notice</strong><br><br>
        Your files are processed in-memory and not stored permanently. 
        If cloud LLMs are enabled via API keys, model providers may receive your text for processing.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 1rem;">
    <small>Made with ❤️ using Streamlit • AI-Powered Resume Analysis Tool</small>
</div>
""", unsafe_allow_html=True)