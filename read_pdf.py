import pymupdf
import pytesseract
from PIL import Image
from io import BytesIO

def extract_text_from_pdf(file_path):
    document = pymupdf.open(file_path)
    extracted_text = ""
    for page_number, page in enumerate(document, start=1):
        page_text = page.get_text()
        extracted_text += f"\n--- Page {page_number} ---\n"
        extracted_text += page_text
    document.close()
    return extracted_text


def extract_text_with_ocr(file_path):
    document = pymupdf.open(file_path)
    extracted_text = ""

    for page_number, page in enumerate(document, start=1):
        pix = page.get_pixmap(dpi=200)

        image = Image.open(
            BytesIO(pix.tobytes("png"))
        )

        page_text = pytesseract.image_to_string(image)

        extracted_text += f"\n--- Page {page_number} ---\n"
        extracted_text += page_text

    document.close()
    return extracted_text
