# Image Processing & Object Recognition Project

This project contains two main computer vision tasks:

1. **Comparative Study of Sharpening Pipelines for X-Ray Images**
2. **Transformed Object Recognition using Feature Matching (SIFT)**

The goal is to explore different image enhancement techniques and evaluate their performance, and to build a robust object recognition system that works under transformations like rotation, translation, and scaling.

---

# Requirements

Install the required dependencies before running the code:

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

## Project Structure
```bash
imageProcessingLibrary/
│
├── task2.ipynb / script   # X-ray enhancement pipelines
├── task6.ipynb / script   # Object recognition using SIFT
├── xray.jpeg              # X-ray input image
├── test1.jpg              # Object image 1
├── test3.jpg              # Object image 2
```

# Task 2: Comparative Study of X-Ray Enhancement Pipelines

## Objective

Design and compare multiple image enhancement pipelines to improve X-ray image quality and analyze their strengths and weaknesses.

---

## Implemented Pipelines

### 1. Global Enhancement Pipeline
**Code function:** `pipline_global(image)`

- Histogram Equalization
- Gamma Correction

Enhances global contrast of the image but may ignore local details.

---

### 2. Adaptive Enhancement Pipeline
**Code function:** `pipline_adaptive(image)`

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gaussian Blur
- Unsharp Masking

Improves local contrast and enhances edges while reducing noise.

---

### 3. Frequency Domain Pipeline
**Code function:** `pipeline_frequency_natural(img_gray)`

- FFT (Fast Fourier Transform)
- High-pass filtering in frequency domain
- Image reconstruction using inverse FFT
- Sharpening using frequency response

Enhances fine structures and edges using frequency-based filtering.

---

### 4. Frequency + CLAHE Pipeline
**Code function:** `pipeline_enhance_xray(img_gray)`

- FFT-based enhancement
- High-frequency emphasis mask
- Image reconstruction
- CLAHE post-processing

Combines frequency sharpening with adaptive contrast enhancement for better medical image clarity.

---

## How to Run Task 2

### Step 1: Load Image

```python
first = cv2.imread(r"D:\computer vision\xray.jpeg", 0)
#Make sure the image is loaded in grayscale mode using 0.

#choose which method you will use or see the comparison of the 4 methods

compare_results(
    first,
    pipline_global(first),
    pipline_adaptive(first),
    pipeline_frequency_natural(first),
    pipeline_enhance_xray(first)
)
```

# Task 6: Transformed Object Recognition (SIFT)

---

## Objective

Detect and match objects even under geometric transformations such as:

- Rotation  
- Translation  
- Slight scaling or distortion  

The system is designed to recognize the same object even if its appearance changes due to viewpoint or transformation.

---

## Methodology

The system is based on classical feature-based computer vision techniques:

- **SIFT (Scale-Invariant Feature Transform)**  
- **Brute Force Matcher (BFMatcher)**  
- **KNN Matching (k = 2)**  
- **Lowe’s Ratio Test** for filtering unreliable matches  

---

## Workflow

1. Detect keypoints using SIFT  
2. Compute feature descriptors  
3. Match descriptors using KNN (k=2)  
4. Apply Lowe’s Ratio Test to keep good matches  
5. Decide if the object exists based on number of matches  
6. Visualize matched keypoints on both images  

---

## How to Run Task 6

### Step 1: Load Images

```python
a = cv2.imread(path of image1)
b = cv2.imread(path of image 2)
B, x, y = detect_and_match_features(a, b)
show_mat(x, y, B)
```
if there is a common object the show_mat function will tell you and show you the common object
if there is no common object it will tell you no common object and will now show you anything
