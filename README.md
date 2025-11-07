# American Sign Language Recognition using CNN
The goal of this project is to help bridge communication gaps for people with hearing or speech impairments by building an image-based ASL recognition system. Using deep learning, the model can classify hand gestures representing letters of the ASL alphabet.

## 📚 Table of Contents
1. [Overview](#-overview)
2. [Dataset](#-dataset)
3. [Model Architecture](#-model-architecture)
4. [Installation](#-installation)
5. [How to Run](#-how-to-run)
6. [Results](#-results)
7. [Key Insights](#-key-insights)
8. [Repository Structure](#-repository-structure)
9. [Future Improvements](#-future-improvements)
10. [About the Author](#-about-the-author)

## 🧠 Overview
This project implements a **Convolutional Neural Network (CNN)** from scratch to recognize **American Sign Language (ASL)** alphabet letters from images.  
It uses **data augmentation**, **regularization**, and deep learning best practices to achieve high accuracy in classifying hand gestures.

The model was developed as part of a research thesis exploring **computer vision for accessibility**, and serves as a portfolio project demonstrating practical AI development skills.

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

## 🧩 Model Architecture

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
...
2.Open the notebook
...
jupyter notebook custom-cnn-using-data-augmentation.ipynb
...
3.Run all cells to:

•Load and preprocess dataset

•Train the CNN model

•Evaluate results and visualize performance


## 📊 Results
| Metric                  | Score |
| :---------------------- | :---: |
| **Training Accuracy**   |  ~98% |
| **Validation Accuracy** |  ~96% |
| **F1-Score**            | 0.95+ |


Example Visualizations:

🖼️ Sample Dataset Grid – ASL Alphabet Samples (Custom Preprocessed)

📈 Accuracy & Loss Curves – Training vs Validation Accuracy

🔢 Confusion Matrix – ASL Classification Performance

✅ Model Predictions – Predicted vs Actual Signs

## 🔬 Key Insights

* Custom CNNs can reach high accuracy on domain-specific vision tasks without transfer learning

* Data augmentation and regularization were critical for strong generalization

* Model performance remained stable over long training sessions

* Learned valuable insights into CNN design, model evaluation, and data preprocessing


## 📁 Repository Structure
.
├── custom-cnn-using-data-augmentation.ipynb
└── README.md



