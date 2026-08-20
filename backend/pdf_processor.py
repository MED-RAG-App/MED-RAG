from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Combined text extracted from the PDF.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The provided file is not a PDF.")

    reader = PdfReader(str(path))

    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    return "\n\n".join(pages_text).strip()