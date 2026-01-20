# License Plate Recognition (LPR) System

## Overview
This project implements a License Plate Recognition (LPR) system using classical computer vision techniques and OCR. It detects vehicle license plates from images, segments individual characters, and recognizes the plate number using Tesseract OCR.

The system demonstrates an end-to-end pipeline for traffic management and vehicle monitoring applications.

---

## Features
- **License Plate Detection:** Detects rectangular license plates using edge detection and contour analysis.
- **Character Segmentation:** Extracts individual characters from detected plates using thresholding and morphological operations.
- **Character Recognition:** Recognizes segmented characters using Tesseract OCR with alphanumeric whitelist.
- **End-to-End Pipeline:** Integrates detection, segmentation, and recognition for automated plate reading.
- **Visualization:** Optional visualization of detected plates and segmented characters for debugging and demonstration.

## Installation

1. Clone the repository:
```bash
git clone <repo_link>
cd lpr
```
2. Install dependencies:
```bash
pip install opencv-python numpy pytesseract

```

3. Tesseract OCR: Download and install Tesseract OCR
 and set the path in char_recognition.py:
```bash
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

4. Run the full pipeline on your dataset:
```bash
python main.py
```
Detected plate numbers will be printed in the console.

---

Dataset: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection