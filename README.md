# COMP5625M Practical Assessment - Deep Learning

This repository contains the coursework for COMP5625M, focused on practical deep learning implementations. The coursework is divided into two main parts:

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Setup and Resources](#setup-and-resources)
- [Assessment Details](#assessment-details)
  - [Part I: Image Classification](#part-i-image-classification)
  - [Part II: Image Captioning](#part-ii-image-captioning)
- [Submission Guidelines](#submission-guidelines)
- [Requirements](#requirements)

---

## Overview

The coursework is divided into two parts:

### Image Classification using DNN and CNN (70 Marks)
- Train and evaluate a deep neural network (DNN) and a convolutional neural network (CNN) on the TinyImageNet30 dataset.
- Use data augmentation, dropout, and hyperparameter tuning to improve model performance.

### Image Captioning using RNN (30 Marks)
- Implement an RNN-based decoder to generate captions for images from the COCO dataset.
- Evaluate the generated captions using metrics and visualize the results.

**Note:** This assessment constitutes 50% of the final grade for the module.

---

## Motivation

The coursework aims to:
- Provide practical experience in training and evaluating deep learning models.
- Demonstrate the use of CNNs for image classification and RNNs for text generation.
- Explore techniques to mitigate overfitting and improve model generalization.
- Familiarize students with real-world datasets like TinyImageNet and COCO.

---

## Setup and Resources

### Prerequisites
- Python 3.8+
- GPU support (recommended for faster training).

### Required Libraries
Install the following libraries using `pip` or `conda`:
- `numpy`
- `matplotlib`
- `torch` (PyTorch)
- `torchvision`
- `scikit-learn`
- `pandas`

### Datasets
- **TinyImageNet30**: Subset of TinyImageNet with 30 categories.
- **COCO_5070**: Subset of the COCO dataset for image captioning.

### File Structure
- `notebooks/`: Contains Jupyter notebooks for implementation.
- `helperDL.py`: Includes utility functions for data preprocessing and evaluation.
- `data/`: Contains datasets and model checkpoints.

---

## Assessment Details

### Part I: Image Classification (70 Marks)
#### Dataset Preparation
- Use the TinyImageNet30 dataset for training and evaluation.
- Implement PyTorch `Dataset` and `DataLoader` classes.

#### Model Implementation
- Train and evaluate a DNN and a CNN.
- Visualize training and validation performance.

#### Overfitting Mitigation
- Apply data augmentation, dropout, and hyperparameter tuning.
- Generate confusion matrices and ROC curves.

#### Testing and Fine-Tuning
- Evaluate models on the test set and submit predictions to the Kaggle leaderboard.
- Fine-tune the models on the CIFAR-10 dataset.

---

### Part II: Image Captioning (30 Marks)
#### Decoder Design
- Implement an RNN-based decoder with embedding and linear layers.

#### Training
- Use precomputed ResNet50 image features and train the decoder.

#### Evaluation
- Compare generated captions with ground truth using metrics.
- Visualize text predictions for a given image.

---

## Submission Guidelines
Submit the following files:
- Completed Jupyter notebook(s) (`.ipynb`).
- HTML version of the notebook(s).
- Predictions CSV file for Kaggle leaderboard submission.

**Ensure that all outputs are displayed in the HTML files, including graphs and metrics.**

---

## Requirements

To reproduce the results:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/COMP5625M-Deep-Learning.git
