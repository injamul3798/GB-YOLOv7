## Paper Title: Attention-driven deep object detection for improved gallbladder cancer diagnosis from ultrasound images
## 📌 Overview

This repository contains the official implementation of **GB-YOLOv7**, a novel deep learning framework designed for early detection of gallbladder cancer using advanced imaging and attention-enhanced object detection.

GB-YOLOv7 extends the baseline YOLOv7 architecture by integrating powerful attention mechanisms and image preprocessing techniques to significantly improve detection accuracy and interpretability.

## 📌 Abstraction
This study presents an advanced modification of the You Only Look Once version 7 (YOLOv7) model named
Gallbladder YOLOv7 (GB-YOLOv7). GB-YOLOv7 integrates a Normalization-based Attention Module (NAM) and a
Global Attention Mechanism (GAM) into the backbone and head architecture. Several image preprocessing
methods are also employed, including median filtering and Contrast-Limited Adaptive Histogram Equalization
(CLAHE). The framework includes three attention mechanism-based models, including Coordinate and Global
Attention Mechanism (CordGAM-YOLOv7), Dual Global Attention Mechanism (DualGAM-YOLOv7), and
Normalization-based Attention Module YOLOv7 (NAM-YOLOv7), enabling a meticulous comparative analysis
with GB-YOLOv7. Results demonstrate its superior performance across all metrics compared to both traditional
and newer YOLO versions: achieving a Recall of 91.3% (vs YOLOv8's 78.1% and YOLOv9's 84.8%), a Mean
Average Precision of 94.0%, and a Specificity of 96.2% (vs YOLOv11's 90.6%). GB-YOLOv7 also shows significant improvements in Matthews Correlation Coefficient (MCC) (72.7% vs YOLOv7's 67.3%) and F1-score (90.0%
vs You Only Look Once version 9 (YOLOv9)'s 81.4%), while maintaining greater parameter efficiency (24.34M vs
YOLOv7's 36.58M), showcasing its potential as a cutting-edge tool for more effective gallbladder cancer
detection

## 🚀 Code Run Instructions

1. First, clone the repository:
   https://github.com/injamul3798/GB-YOLOv7

2. Navigate to the main directory of the project.

3. Open the notebook file:
   **`GBCU_final_code_with_gradecam.ipynb`**

4. Follow the step-by-step instructions provided inside the notebook to run the code successfully.

---

**Author:** Md. Injamul Haque

**Role:** AI/ML Researcher & Engineer
