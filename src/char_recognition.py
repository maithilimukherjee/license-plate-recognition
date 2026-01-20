import pytesseract
import cv2

# set tesseract path once
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def recognize_characters(char_images):
    """
    input: list of character images (numpy arrays)
    output: string representing the detected plate
    """
    plate_text = ""

    for i, ch in enumerate(char_images):
        # resize to improve OCR accuracy
        ch = cv2.resize(ch, (40, 40))  # slightly larger than 28x28

        # pytesseract config for single characters and alphanumeric whitelist
        config = "--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        # OCR on the character
        text = pytesseract.image_to_string(ch, config=config).strip()

        # skip empty or junk outputs
        if text:
            plate_text += text

    return plate_text
