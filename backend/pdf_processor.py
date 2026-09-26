import io
import fitz
import pytesseract

from PIL import Image, ImageOps, ImageEnhance, ImageFilter


TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def _prepare_image(image):
    image = image.convert("RGB")

    width, height = image.size
    scale = 2

    image = image.resize(
        (width * scale, height * scale),
        Image.Resampling.LANCZOS,
    )

    gray = ImageOps.grayscale(image)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = gray.filter(ImageFilter.SHARPEN)

    return gray


def _ocr_image(image):
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
            pass

    confidence = (
        round(sum(confidences) / len(confidences), 2)
        if confidences
        else 0
    )

    return text, confidence


def extract_text_from_image_bytes(file_content):
    image = Image.open(io.BytesIO(file_content))

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
    pdf = fitz.open(stream=file_content, filetype="pdf")

    pages = []
    ocr_pages = []
    confidences = []

    for page_number, page in enumerate(pdf, start=1):
        native_text = page.get_text("text").strip()

        # Scanned/handwritten pages normally have little or no native text.
        if len(native_text) >= 30:
            text = native_text
            ocr = False
            confidence = 100.0
        else:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(2.5, 2.5),
                alpha=False,
            )

            image = Image.open(
                io.BytesIO(pix.tobytes("png"))
            )

            text, confidence = _ocr_image(image)
            ocr = True
            ocr_pages.append(page_number)
            confidences.append(confidence)

        pages.append({
            "page": page_number,
            "text": text,
            "ocr": ocr,
            "confidence": confidence,
        })

    pdf.close()

    extracted_text = "\n\n".join(
        page["text"]
        for page in pages
        if page["text"]
    ).strip()

    average_confidence = (
        round(sum(confidences) / len(confidences), 2)
        if confidences
        else 0
    )

    return {
        "text": extracted_text,
        "total_pages": len(pages),
        "ocr_used": bool(ocr_pages),
        "ocr_pages": ocr_pages,
        "average_ocr_confidence": average_confidence,
        "pages": pages,
    }
