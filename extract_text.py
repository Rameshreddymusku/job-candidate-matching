import pdfplumber
from docx import Document

def extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        parts = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                parts.append(p.extract_text() or "")
        return "\n".join(parts)
    if path.lower().endswith(".docx"):
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported file type. Upload PDF or DOCX.")
