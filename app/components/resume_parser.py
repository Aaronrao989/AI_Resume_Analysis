from typing import Tuple
import re
from components.utils import clean_text


def _normalize_bullets(text: str) -> str:
    """
    Normalize common bullet characters to '-' so downstream
    ATS + bullet heuristics can detect them reliably.
    """
    if not text:
        return ""
    return re.sub(r"[•▪◦●‣∙]", "-", text)


def _postprocess_lines(text: str) -> str:
    """
    Clean extracted text while PRESERVING structure.
    - Keep newlines
    - Remove junk-only lines
    - Normalize spacing per line
    """
    lines = []
    for ln in text.splitlines():
        ln = ln.replace("\x00", " ").strip()
        if not ln:
            lines.append("")  # preserve blank line for structure
            continue

        # collapse excessive internal spacing (but not newlines)
        ln = re.sub(r"[ \t]{2,}", " ", ln)
        lines.append(ln)

    # collapse multiple blank lines to max 2
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_text_from_pdf(file_path: str) -> Tuple[str, int]:
    """
    Return (text, page_count).

    Strategy:
    1) Try PyMuPDF (fitz) – fast, accurate (preferred on macOS M-series)
    2) Fallback to pdfplumber

    If neither works, returns ("", 0).
    """
    text = ""
    page_count = 0

    # ---------- Try PyMuPDF ----------
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        page_count = len(doc)

        for page in doc:
            page_text = ""
            try:
                page_text = page.get_text("text") or ""
            except Exception:
                try:
                    page_text = page.get_text() or ""
                except Exception:
                    page_text = ""

            if page_text:
                text += "\n" + page_text

        doc.close()

        if text.strip():
            text = _normalize_bullets(text)
            text = _postprocess_lines(text)
            text = clean_text(text)  # final light cleanup (safe)
            return text, page_count

    except Exception:
        pass  # silently fall back

    # ---------- Fallback: pdfplumber ----------
    try:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text += "\n" + page_text

        if text.strip():
            text = _normalize_bullets(text)
            text = _postprocess_lines(text)
            text = clean_text(text)
            return text, page_count

    except Exception:
        return "", 0

    return "", 0
