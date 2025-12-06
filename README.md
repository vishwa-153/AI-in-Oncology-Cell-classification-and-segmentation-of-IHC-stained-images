**AI in Oncology: Histopathology Image Analysis**
**Overview**
This repository contains a project focused on developing an AI-based system for analyzing histopathological images in oncology. The system utilizes the Swin Transformer for classifying images into 7 classes (Immune cells, Necrosis, Other, Stroma, Tumor, alveoli, background) and Cellpose for segmentation, providing accurate outlines and predictions. It includes visualization tools, evaluation metrics, and a potential Streamlit integration for interactive use.

**Features**
**Image Classification:** Uses a pre-trained Swin Transformer model to classify images into 7 pathology classes.
**Image Segmentation:** Employs Cellpose to segment cells and generate colored outlines based on predicted classes.
**Visualization:** Displays segmented images with true and predicted labels, including confidence scores.
**Evaluation Metrics:** Computes Accuracy, Precision, Recall, F1 Score, AUC-ROC, and a percentage-based Confusion Matrix heatmap.
**Dataset:** Utilizes the MIHIC dataset MIHIC_dataset

**Requirements**
Python 3.9+
**Libraries:** torch, torchvision, timm, cellpose, numpy, matplotlib, tqdm, scikit-learn, opencv-python
**Install dependencies via:** pip install torch torchvision timm cellpose numpy matplotlib tqdm scikit-learn opencv-python
**Installation**
**Install dependencies:** pip install -r requirements.txt
Download the MIHIC dataset from here MIHIC_dataset

**Usage**
Run this script to view results: python visualize_test_images.py
View the output: A plot with 4 segmented images from different classes and a second plot with evaluation metrics.
Adjust parameters (e.g., diameter in segment_and_classify) in the script for better segmentation if needed.

**File Structure**
visualize_test_images.py: Main script for segmentation, classification, visualization, and metrics.
best_swin_tiny_model.pth: Pre-trained Swin Transformer model weights.
MIHIC_dataset/: Directory containing the test dataset with 7 class subfolders.