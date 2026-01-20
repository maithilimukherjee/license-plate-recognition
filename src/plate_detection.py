import cv2
import numpy as np
from src.preprocessing import preprocess_image

'''
license plates are rectangular
they have strong edges
contours can capture closed boundaries
we filter contours by shape + size
'''

def detect_plate(image_path):
    
    #read and resize image
    
    img, gray, edges = preprocess_image(image_path)
    
    #find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    
    #sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    
    for cnt in contours:
        #approximate contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        
        #if contour has 4 points, we assume it's the plate
        if len(approx) == 4:
            plate_contour = approx
            break   
    
    if plate_contour is None:
        return None, img  #no plate found
    
    #draw contour on image
    
    cv2.drawContours(img, [plate_contour], -1, (0, 255, 0), 3)
    
    #mask and extract plate region
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    plate = cv2.bitwise_and(img, img, mask=mask)
    
    return plate, img