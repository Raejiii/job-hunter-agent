"""Turn uploaded files into plain text."""

import io
import re

from pypdf import PdfReader


def extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = data.decode("utf-8", errors="ignore")
    return clean(text)


def clean(text: str) -> str:
    text = text.replace("•", "-").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
