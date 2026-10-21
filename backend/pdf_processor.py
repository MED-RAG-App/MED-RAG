import io
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pypdf import PdfReader


# Locate Tesseract automatically; allow an environment-variable override.
import os
import shutil

tesseract_path = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def _prepare_image(image):
    """Improve image readability before OCR."""
    image = image.convert("RGB")

    width, height = image.size
    image = image.resize(
        (width * 2, height * 2),
        Image.Resampling.LANCZOS,
    )

    image = ImageOps.grayscale(image)
    image = ImageEnhance.Contrast(image).enhance(1.8)
    image = image.filter(ImageFilter.SHARPEN)

    return image


def _ocr_image(image):
    """Extract text and estimate OCR confidence."""
    prepared = _prepare_image(image)
    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(
        prepared,
        config=config,
    ).strip()

    data = pytesseract.image_to_data(
        prepared,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    confidences = []

    for value in data["conf"]:
        try:
            value = float(value)
            if value >= 0:
                confidences.append(value)
        except (ValueError, TypeError):
            continue

    confidence = (
        round(sum(confidences) / len(confidences), 2)
        if confidences
        else 0.0
    )

    return text, confidence


def extract_text_from_image_bytes(file_content):
    """Extract text from a PNG, JPG, JPEG or WEBP image."""
    with Image.open(io.BytesIO(file_content)) as image:
        text, confidence = _ocr_image(image)

    return {
        "text": text,
        "total_pages": 1,
        "ocr_used": True,
        "ocr_pages": [1],
        "average_ocr_confidence": confidence,
        "pages": [
            {
                "page": 1,
                "text": text,
                "ocr": True,
                "confidence": confidence,
            }
        ],
    }


def extract_text_from_pdf_bytes(file_content):
    """Extract native PDF text and OCR pages that have little text."""
    pdf = pymupdf.open(
        stream=file_content,
        filetype="pdf",
    )

    pages = []
    ocr_pages = []
    ocr_confidences = []

    try:
        for page_number, page in enumerate(pdf, start=1):
            native_text = page.get_text("text").strip()

            if len(native_text) >= 30:
                text = native_text
                used_ocr = False
                confidence = None
            else:
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2.5, 2.5),
                    alpha=False,
                )

                image_bytes = pixmap.tobytes("png")

                with Image.open(io.BytesIO(image_bytes)) as image:
                    text, confidence = _ocr_image(image)

                used_ocr = True
                ocr_pages.append(page_number)
                ocr_confidences.append(confidence)

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                    "ocr": used_ocr,
                    "confidence": confidence,
                }
            )

    finally:
        pdf.close()

    extracted_text = "\n\n".join(
        page["text"]
        for page in pages
        if page["text"]
    ).strip()

    average_confidence = (
        round(
            sum(ocr_confidences) / len(ocr_confidences),
            2,
        )
        if ocr_confidences
        else 0.0
    )

    return {
        "text": extracted_text,
        "total_pages": len(pages),
        "ocr_used": bool(ocr_pages),
        "ocr_pages": ocr_pages,
        "average_ocr_confidence": average_confidence,
        "pages": pages,
    }


def extract_text_from_pdf(pdf_path: str) -> str:
    """Preserve the existing path-based PDF extraction interface."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The provided file is not a PDF.")

    extraction = extract_text_from_pdf_bytes(path.read_bytes())
    return extraction["text"]