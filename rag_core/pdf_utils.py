from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


def extract_pdf_pages(pdf_path: Path) -> List[Dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        normalized = " ".join(text.split())
        if normalized:
            pages.append(
                {
                    "text": normalized,
                    "source": pdf_path.name,
                    "page": i + 1,
                }
            )
    return pages
