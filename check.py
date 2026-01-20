import os
import cv2

image_dir = "data/images"

files = os.listdir(image_dir)
print("total images:", len(files))

img_path = os.path.join(image_dir, files[0])
img = cv2.imread(img_path)

print("image shape:", img.shape)
cv2.imshow("sample", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
