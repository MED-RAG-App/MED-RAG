import logging
import os
import unicodedata
from io import BytesIO

from PyPDF2 import PdfReader

log = logging.getLogger("medrag.pdf")

MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "300"))


def process_pdf(pdf_content):
    """
    Extract text from PDF bytes.

    Returns: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}]
    Raises ValueError (safe to show to the client) for any bad input.
    """

    if not isinstance(pdf_content, (bytes, bytearray)):
        raise TypeError("process_pdf expects PDF file content as bytes.")

    if not pdf_content:
        raise ValueError("The uploaded PDF is empty.")

    if not pdf_content.startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not appear to be a valid PDF.")

    try:
        reader = PdfReader(BytesIO(pdf_content), strict=False)

        if reader.is_encrypted:
            # decrypt("") returns 0 / NOT_DECRYPTED when a password is needed
            # (it does NOT raise), so the return value must be checked.
            try:
                decrypted = reader.decrypt("")
            except Exception as exc:
                log.warning("PDF decrypt failed: %s", exc)
                raise ValueError("Password-protected PDFs are not supported.") from exc

            if not decrypted:
                raise ValueError("Password-protected PDFs are not supported.")

        total_pages = len(reader.pages)

        if total_pages == 0:
            raise ValueError("The PDF has no pages.")

        if total_pages > MAX_PDF_PAGES:
            raise ValueError(f"The PDF has too many pages (limit {MAX_PDF_PAGES}).")

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # NFC keeps Devanagari / Kannada combining characters consistent
            text = unicodedata.normalize("NFC", text)
            text = " ".join(text.split())

            if text:
                extracted_pages.append({"page": page_number, "text": text})

        if not extracted_pages:
            raise ValueError(
                "No selectable text was extracted from this PDF. "
                "It may be scanned/image-only and require OCR."
            )

        return extracted_pages

    except ValueError:
        raise
    except Exception as exc:
        log.warning("PDF read failed: %s", exc)
        raise ValueError("The PDF is corrupted or could not be read.") from exc