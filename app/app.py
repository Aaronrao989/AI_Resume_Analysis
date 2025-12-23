import streamlit as st
import os
import pandas as pd

from components.resume_parser import extract_text_from_pdf
from components.llm_review import review_resume

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI Resume Reviewer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    p, div, span, label {
        font-family: 'DM Sans', sans-serif !important;
        color: #e0e0e0 !important;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em;
    }
    
    /* Caption/subtitle */
    .stApp [data-testid="stCaptionContainer"] {
        color: #a0a0b0 !important;
        font-size: 1.1rem !important;
        margin-bottom: 2rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e3f 0%, #2a2a4a 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(102, 126, 234, 0.05);
        border: 2px dashed rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.6);
        background: rgba(102, 126, 234, 0.08);
    }
    
    /* Text areas */
    .stTextArea textarea {
        background: rgba(30, 30, 62, 0.6) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-family: 'DM Sans', sans-serif !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(30, 30, 62, 0.6) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0c0 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(76, 175, 80, 0.15) !important;
        border: 1px solid rgba(76, 175, 80, 0.4) !important;
        border-radius: 10px !important;
        color: #4caf50 !important;
    }
    
    /* Error message */
    .stError {
        background: rgba(244, 67, 54, 0.15) !important;
        border: 1px solid rgba(244, 67, 54, 0.4) !important;
        border-radius: 10px !important;
        color: #f44336 !important;
    }
    
    /* Warning message */
    .stWarning {
        background: rgba(255, 152, 0, 0.15) !important;
        border: 1px solid rgba(255, 152, 0, 0.4) !important;
        border-radius: 10px !important;
        color: #ff9800 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.08) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.12) !important;
        border-color: rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(15, 15, 30, 0.8) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace !important;
        color: #a0d5ff !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(102, 126, 234, 0.2) !important;
        margin: 2rem 0 !important;
    }
    
    /* Column styling for cards */
    .card-container {
        background: rgba(30, 30, 62, 0.4);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .card-container:hover {
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Remove default padding */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Section headers */
    .section-header {
        color: #667eea !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    /* Footer */
    footer {
        color: #6b6b80 !important;
        text-align: center;
        padding: 2rem 0;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🔍 AI Resume Reviewer")
st.caption("ATS-aware resume analysis with ML role prediction, FAISS knowledge base, and LLM feedback")

# -------------------------------------------------
# LOAD JOB ROLES FROM CSV
# -------------------------------------------------

CSV_PATH = "data/job_postings_resume1(in).csv"

roles = ["(Auto-detect from resume)"]

try:
    df_roles = pd.read_csv(CSV_PATH)
    for col in ["job_position", "title", "role"]:
        if col in df_roles.columns:
            csv_roles = (
                df_roles[col]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
            roles.extend(sorted(csv_roles))
            break
except Exception as e:
    st.sidebar.warning("⚠️ Could not load job roles from CSV")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("")

    target_role = st.selectbox(
        "🎯 Target Job Role",
        options=roles,
        index=0,
        help="Select the role you are applying for, or let the system auto-detect",
    )

    jd_text = st.text_area(
        "📄 Job Description (optional)",
        height=180,
        placeholder="Paste JD here for more targeted ATS & AI feedback",
    )

    st.markdown("---")
    
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    <div style='font-size: 0.9rem; line-height: 1.8;'>
    • <b>ML prediction</b> identifies your role<br>
    • <b>Selected role</b> guides feedback<br>
    • <b>FAISS knowledge</b> base provides skills<br>
    • <b>ATS score</b> is computed<br>
    • <b>LLM generates</b> actionable feedback
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# MAIN INPUT AREA
# -------------------------------------------------

st.markdown("")
st.markdown("### 📝 Upload Your Resume")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "📄 Upload Resume (PDF)",
        type=["pdf"],
    )

with col2:
    resume_text_input = st.text_area(
        "✍️ Or paste resume text",
        height=260,
        placeholder="Paste resume text here if not uploading PDF",
    )

resume_text = ""

if uploaded_file is not None:
    tmp_path = os.path.join("/tmp", uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    text, pages = extract_text_from_pdf(tmp_path)
    if text:
        resume_text = text
        st.success(f"✅ Extracted text from PDF ({pages} pages)")
    else:
        st.error("❌ Could not extract text from PDF")

else:
    resume_text = resume_text_input

# -------------------------------------------------
# ANALYZE BUTTON
# -------------------------------------------------

st.markdown("")

if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
    if not resume_text.strip():
        st.error("⚠️ Please upload a resume or paste resume text.")
    else:
        with st.spinner("🔄 Analyzing resume…"):
            result = review_resume(
                resume_text=resume_text,
                guidance_blobs=[],
                jd_text=jd_text,
                job_role=None if target_role == "(Auto-detect from resume)" else target_role,
            )

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        st.markdown("<h2 class='section-header'>📊 Analysis Results</h2>", unsafe_allow_html=True)

        # ---- ATS SCORE ----
        ats_score = min(100.0, float(result["ats"]["score"]))
        st.metric("ATS Compatibility Score", f"{ats_score:.1f} / 100")

        with st.expander("_______________📋 View ATS Score Breakdown"):
            st.json(result["ats"]["detail"])

        st.markdown("")
        
        # ---- ROLE ANALYSIS ----
        st.markdown("<h2 class='section-header'>🎯 Role Analysis</h2>", unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)

        with colA:
            st.info(f"**Target Role**\n\n{result['target_role']}")

        with colB:
            st.info(
                f"**ML Predicted Role**\n\n{result['predicted_role'] or 'N/A'}"
            )

        with colC:
            if result.get("llm_used"):
                st.success("🤖 LLM Active (Groq)")
            else:
                st.warning("⚠️ LLM Fallback Mode")

        st.markdown("")
        
        # ---- AI FEEDBACK ----
        st.markdown("<h2 class='section-header'>🤖 AI Feedback</h2>", unsafe_allow_html=True)

        st.code(result["llm_feedback_raw"], language="json")

        # ---- DEBUG INFO (OPTIONAL) ----
        with st.expander("____________🧪 Debug Information"):
            st.write("**LLM used:**", result.get("llm_used"))
            st.write("**Predicted role:**", result.get("predicted_role"))
            st.write("**Target role:**", result.get("target_role"))

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b6b80; font-size: 0.9rem; padding: 1rem 0;'>
    Built with <span style='color: #667eea;'>Streamlit</span> • 
    <span style='color: #667eea;'>ML</span> • 
    <span style='color: #667eea;'>FAISS</span> • 
    <span style='color: #667eea;'>ATS</span> • 
    <span style='color: #667eea;'>Groq LLM</span>
</div>
""", unsafe_allow_html=True)