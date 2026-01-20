import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    
    #read the image
    img = cv2.imread(image_path)
    
    #resize for consistency
    img = cv2.resize(img, (600,400))
    
    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #apply bilateral filter to reduce noise while keeping edges sharp
    gray_filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    #detect edges using Canny
    edges = cv2.Canny(gray_filtered, 30, 200)
    
    #display results
    
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title('Original Image'); 
    plt.subplot(1,3,2); plt.imshow(gray, cmap='gray'); plt.title("grayscale")
    plt.subplot(1,3,3); plt.imshow(edges, cmap='gray'); plt.title("edges")
    plt.show()