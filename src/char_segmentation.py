import cv2
import numpy as np

def segment_characters(plate_img):
    """
    input: plate image (BGR)
    output: list of character images, left-to-right
    """

    # convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # adaptive thresholding to handle shadows/lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # morphological closing to join broken parts
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    characters = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # filter by height relative to plate & width
        if h > 0.35 * plate_img.shape[0] and w > 10:
            char = thresh[y:y+h, x:x+w]

            # resize char to fixed size for pytesseract
            char = cv2.resize(char, (40, 40))
            characters.append((x, char))

    # sort left to right
    characters = sorted(characters, key=lambda x: x[0])
    characters = [char for _, char in characters]

    return characters
