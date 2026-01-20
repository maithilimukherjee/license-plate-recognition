import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess_image(image_path, visualize=False):
    
    #read image
    img = cv2.imread(image_path)
    
    #resize image
    img = cv2.resize(img, (600, 400))

    #convert to grayscale and apply filters
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    #detect edges
    edges = cv2.Canny(gray_filtered, 30, 200)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("original")

    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("grayscale")

    plt.subplot(1, 3, 3)
    plt.imshow(edges, cmap="gray")
    plt.title("edges")
    
    if visualize:
        plt.show()
    return edges
