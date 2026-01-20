from src.plate_detection import detect_plate
from src.char_segmentation import segment_characters
from src.char_recognition import recognize_characters
import cv2
import os

image_dir = "data/images"

for img_name in os.listdir(image_dir):
    if img_name.endswith((".jpg", ".png")):
        img_path = os.path.join(image_dir, img_name)

        plate, _ = detect_plate(img_path)

        if plate is None:
            continue

        chars = segment_characters(plate)

        if len(chars) == 0:
            continue

        plate_number = recognize_characters(chars)
        print("detected plate:", plate_number)