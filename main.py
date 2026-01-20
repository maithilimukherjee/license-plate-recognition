import os
from preprocessing import preprocess_image

image_dir = "data/images"

for img_name in os.listdir(image_dir):
    if img_name.endswith((".jpg", ".png")):
        img_path = os.path.join(image_dir, img_name)
        preprocess_image(img_path, visualize=False)
