import re
import json
from io import BytesIO


# -------------------------------------------------
# TEXT CLEANING
# -------------------------------------------------

def clean_text(text: str) -> str:
    """
    Basic cleanup for extracted PDF text.
    IMPORTANT: preserves newlines and tech symbols.
    """
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\u0000", " ")
    text = re.sub(r"[\r\t]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# -------------------------------------------------
# TECH TOKEN NORMALIZATION (CRITICAL)
# -------------------------------------------------

def normalize_token(token: str) -> str:
    """
    Normalize technical tokens so matching is consistent across:
    resume text, JD skills, FAISS meta, ATS checks.

    Examples:
    - C#        -> csharp
    - ASP.NET   -> aspnet
    - .NET Core -> dotnetcore
    - Web API   -> webapi
    - Node.js   -> nodejs
    """
    if not token:
        return ""

    t = token.lower().strip()

    replacements = {
        "c#": "csharp",
        "c++": "cplusplus",
        ".net core": "dotnetcore",
        ".net": "dotnet",
        "asp.net core": "aspnetcore",
        "asp.net": "aspnet",
        "web api": "webapi",
        "rest api": "restapi",
        "node.js": "nodejs",
        "node js": "nodejs",
        "javascript": "javascript",
        "typescript": "typescript",
    }

    for k, v in replacements.items():
        t = t.replace(k, v)

    # remove remaining non-alphanumeric chars
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


# -------------------------------------------------
# JSON HELPERS
# -------------------------------------------------

def load_json(path: str, default=None):
    """Load a JSON file safely, return default if error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, obj) -> bool:
    """Save a JSON file safely. Return True on success."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# -------------------------------------------------
# CSV LIST UTILITY
# -------------------------------------------------

def split_csv_list(s: str):
    """Split a comma-separated string into a cleaned list of items."""
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


# -------------------------------------------------
# RESUME PDF CREATION (UNCHANGED)
# -------------------------------------------------

def create_resume_pdf(resume_text: str) -> bytes:
    """Convert resume text into a styled PDF and return bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
    except Exception:
        raise RuntimeError(
            "reportlab is required to generate PDFs. Please install reportlab."
        )

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    name_style = ParagraphStyle(
        "NameStyle",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=20,
    )

    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6,
    )

    normal_style = ParagraphStyle(
        "NormalText",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
    )

    story = []
    lines = resume_text.split("\n")

    # Detect name at top
    if lines and lines[0].strip():
        story.append(Paragraph(lines[0].strip(), name_style))
        story.append(Spacer(1, 12))
        lines = lines[1:]

    for line in lines:
        if not line.strip():
            continue

        if line.strip().endswith(":") or line.strip().isupper():
            story.append(Paragraph(line.strip(), section_style))
        else:
            story.append(Paragraph(line.strip(), normal_style))

        story.append(Spacer(1, 6))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
