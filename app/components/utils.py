import re
import json
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


def clean_text(text: str) -> str:
    """Basic cleanup for extracted PDF text."""
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\u0000", " ")
    text = re.sub(r"[\r\t]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def load_json(path: str, default=None):
    """Load a JSON file safely, return default if error."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default


def create_resume_pdf(resume_text: str) -> bytes:
    """Convert resume text into a styled PDF and return bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=40, rightMargin=40,
                            topMargin=40, bottomMargin=40)

    styles = getSampleStyleSheet()

    # Custom styles
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

    # Detect name at top (first non-empty line)
    if lines and lines[0].strip():
        story.append(Paragraph(lines[0].strip(), name_style))
        story.append(Spacer(1, 12))
        lines = lines[1:]  # remove name line

    for line in lines:
        if not line.strip():
            continue

        # Section headers like "Education:", "Skills:", etc.
        if line.strip().endswith(":") or line.strip().isupper():
            story.append(Paragraph(line.strip(), section_style))
        else:
            story.append(Paragraph(line.strip(), normal_style))
        story.append(Spacer(1, 6))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
