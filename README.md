# American Sign Language Recognition using CNN
The goal of this project is to help bridge communication gaps for people with hearing or speech impairments by building an image-based ASL recognition system. Using deep learning, the model can classify hand gestures representing letters of the ASL alphabet.

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Thesis Reference](#-thesis-reference)
3. [Dataset](#-dataset)
4. [Model Architecture](#-model-architecture)
5. [Implemented Notebooks](#-implemented-notebooks)
6. [Model Comparison](#-model-comparison)
7. [Results](#-results)
8. [Installation](#-installation)
9. [How to Run](#-how-to-run)
10. [Key Insights](#-key-insights)
11. [Future Improvements](#-future-improvements)
12. [About the Author](#-about-the-author)
13. [Repository Structure](#-repository-structure)


## 🧠 Overview

This project implements **Convolutional Neural Networks (CNNs)** to recognize **American Sign Language (ASL)** alphabet letters from images.  
It uses **data augmentation**, **regularization**, and **transfer learning** to build robust models capable of accurately classifying hand gestures

The project was conducted as part of my **Bachelor Thesis in Artificial Intelligence**, exploring **computer vision for accessibility** — aiming to enhance communication for individuals with hearing and speech impairments.

This repository contains two main approaches:
- A **Custom CNN** built from scratch with data augmentation and regularization.
- A **Transfer Learning** solution using **VGG-16** fine-tuning.

Both approaches and their evaluation are discussed in my bachelor thesis (link and details below).

## 📘 Thesis Reference

**Bachelor Thesis (PDF) — Full report, methodology, and results:**  
🔗 Drive link (public):  
[Thesis](https://drive.google.com/file/d/1hd7Jqv-Uwd5BXK47FnNZ4HYIxTkkrufS/view?usp=sharing)


## 📦 Dataset

**Base Dataset:** [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
The original dataset consists of labeled RGB images of hand gestures representing:
- 26 letters (A–Z)
- 3 special classes: *space*, *delete*, and *nothing*
- 
- Contains labeled images of hand gestures for **A–Z**, plus **space**, **delete**, and **nothing**
- Training images: ~87,000  
- Test images: ~29,000  
- Each image: **200x200 pixels**, RGB

**Custom Modifications:**
To enhance model compatibility and reduce preprocessing overhead, the dataset was modified as follows:
- **Resized** all images to 64×64 pixels for faster CNN training.  
- **Cleaned and standardized** folder structure for easier loading.  
- **Generated a subset** of unique images per class for faster visualization and validation.  
- **Applied ImageDataGenerator** for real-time augmentation (rotation, zoom, flipping, shifting).

Final structure:

/asl_alphabet_train
/A
/B
...
/Z
/space
/delete
/nothing

## 🧩 Models Architecture

**Frameworks & Libraries Used**
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib, Seaborn  
- OpenCV  
- scikit-learn  

**Model Design**
- Custom **Sequential CNN** with:
  - Multiple convolutional layers (ReLU activation)
  - MaxPooling for feature reduction
  - Dropout + L2 regularization to reduce overfitting
  - Batch Normalization for stable training
  - Fully connected Dense layers for classification  
- **Optimizer:** Adam  
- **Loss:** Categorical Crossentropy  
- **Metrics:** Accuracy, F1-score, Precision

**Data Augmentation:**
- Random rotation  
- Horizontal flipping  
- Zoom and width/height shift  
Implemented using `ImageDataGenerator` to improve generalization.

---
## 📓 Implemented Notebooks

### 🧩 1. Custom CNN (from scratch)
🔗 [View on Kaggle](https://www.kaggle.com/code/muhammadmagdysobhy/custom-cnn-using-data-augmentation)

A custom-designed **Sequential CNN** built from scratch using TensorFlow/Keras.  
Focused on lightweight design, regularization, and high accuracy through data augmentation.

**Layers Overview**
- Convolutional + ReLU
- MaxPooling
- Dropout + L2 Regularization
- Batch Normalization
- Dense Output Layer (Softmax, 29 classes)

**Training Setup**
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Epochs: 30  
- Image Size: 64×64×3  
- Batch Size: 32  

**Frameworks Used**
TensorFlow, Keras, NumPy, Pandas, OpenCV, Matplotlib, Seaborn, scikit-learn

- Built using Keras Sequential API  
- Includes heavy use of **data augmentation**
- Trained on resized ASL dataset  
- Reached high accuracy and low overfitting  

📊 *This notebook demonstrates understanding of CNN fundamentals and image preprocessing pipelines.*

---

### 🧠 2. CNN with VGG-16 (Transfer Learning)
🔗 [View on Kaggle](https://www.kaggle.com/code/muhammadmagdysobhy/cnn-model-vgg-16-with-data-agumentation)

- Uses a **pre-trained VGG-16** network on ImageNet  
- Top layers fine-tuned for ASL classification  
- Retains convolutional base to leverage pre-learned visual features  
- Employs same preprocessing pipeline (augmentation, resizing, normalization)

📊 *This notebook demonstrates the use of transfer learning for improved generalization and reduced training time.*

---


## ⚖️ Model Comparison

| Feature | Custom CNN | VGG-16 Transfer Learning |
| :-- | :-- | :-- |
| **Model Type** | Sequential (built from scratch) | Pre-trained (Transfer Learning) |
| **Parameters** | ~1.2M | ~15M |
| **Training Time** | Faster (due to fewer layers) | Slower (heavier model) |
| **Accuracy** | 96–98% | 97–99% |
| **F1-Score** | ~0.95 | ~0.97 |
| **Overfitting** | Slight (mitigated with dropout) | Minimal due to pre-trained base |
| **Use Case** | Lightweight, deployable model | High accuracy for research & production |
| **Complexity** | Lower | Higher (fine-tuning required) |

<!-- **Visual Comparison Ideas:**
//- 📈 *Accuracy vs Epochs* (Custom CNN vs VGG-16)
//- 🧩 *Confusion Matrices* for both models  
//- ⏱️ *Training Time Comparison Chart*
//- 🔍 *Sample Predictions* (Correct & Incorrect Cases) -->

📌 *The results showed that while both models achieved excellent accuracy, VGG-16 performed slightly better on unseen data — demonstrating the power of transfer learning.*

## 📊 Results

| Metric                  | Custom CNN | VGG-16 |
| :---------------------- | :--------: | :----: |
| **Training Accuracy**   |     98%    |   99%  |
| **Validation Accuracy** |     96%    |   98%  |
| **F1-Score**            |    0.95    |  0.97  |
| **Loss (Validation)**   |    0.18    |  0.09  |


## ⚙️ Installation

You can install the required packages manually:

### Requirements
You can install the required packages manually:
...
pip install tensorflow keras numpy pandas opencv-python matplotlib seaborn scikit-learn
...
## 🚀 How to Run

1.Clone this repository:
...
git clone https://github.com/your-username/asl-cnn.git
cd asl-cnn
jupyter notebook custom-cnn-using-data-augmentation.ipynb

...

2.Open the notebook
...
jupyter notebook custom-cnn-using-data-augmentation.ipynb
...
3.Run all cells to:

•Load and preprocess dataset

•Train the CNN model

•Evaluate results and visualize performance


## 🔬 Key Insights

•Transfer Learning (VGG-16) yields slightly better generalization.

•Custom CNN provides a balance between performance and computational efficiency.

•Augmentation and normalization were key to achieving stable training.

•Both models successfully recognize ASL gestures with near-human accuracy.

## 🧭 Future Improvements

•Extend to real-time ASL recognition (video streams).

•Experiment with other architectures (ResNet50, EfficientNet).

•Deploy via a web app or mobile app using TensorFlow Lite.

•Build a multi-language sign recognition model.

## 👤 About the Author

****Muhammad Magdy Sobhy****
- AI & Deep Learning Enthusiast | Computer Vision Researcher

📫 Links:

•[LinkedIn](https://www.linkedin.com/in/muhammad-magdy-652545238/)

•[GitHub](https://github.com/MuhammadMagdyy)

•[Kaggle](https://www.kaggle.com/muhammadmagdysobhy)

Passionate about building AI systems that enhance accessibility and human–computer interaction.

## 📁 Repository Structure
├── custom-cnn-using-data-augmentation.ipynb
├── cnn-model-vgg-16-with-data-agumentation.ipynb
└── README.md



