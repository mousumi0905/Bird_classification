# 🐦 Bird Species Classification using Deep Learning

A Convolutional Neural Network (CNN) based image classifier that identifies **10 different bird species** using **transfer learning**, deployed as an interactive web app with **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Bird Species Covered](#bird-species-covered)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Web App Demo](#web-app-demo)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## 🧠 Overview

Birds belong to the class **Aves**, with nearly 10,000 living species worldwide — yet most people lack the ornithological expertise to identify them. This project uses **deep learning** to automate the process of bird species identification from images.

A **Convolutional Neural Network (CNN)** was built using the **transfer learning** approach, leveraging a pre-trained model as a feature extractor and fine-tuning it on a custom dataset of 10 bird species. The trained model was then wrapped into a simple, user-friendly **web application** using Streamlit, allowing anyone — even without coding knowledge — to upload an image and get instant species predictions.

---

## 🐤 Bird Species Covered

| Label | Species |
|-------|---------|
| 0 | Black Footed Albatross |
| 1 | Laysan Albatross |
| 2 | Sooty Albatross |
| 3 | Groove Billed Ani |
| 4 | Least Auklet |
| 5 | Parakeet Auklet |
| 6 | Rhinoceros Auklet |
| 7 | Brewer Blackbird |
| 8 | Red Winged Blackbird |
| 9 | Crested Auklet |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Image Processing:** PIL (Pillow)
- **Numerical Computing:** NumPy
- **Web App Framework:** Streamlit
- **Deployment/Testing:** Localtunnel (for local server tunneling)

---

## 🔄 Project Workflow

1. **Dataset Preparation** – Collected and labeled images for 10 bird species (85% train / 15% test split).
2. **Transfer Learning** – Used a pre-trained CNN as a feature extractor and added a custom classification head for the 10 classes.
3. **Model Training & Evaluation** – Trained the model and evaluated it on the test set using accuracy, F1 score, and confusion matrix.
4. **Inference Pipeline** – Built a function to preprocess new images (resize, normalize) and run predictions using the trained `.h5` model.
5. **Web App Deployment** – Wrapped the trained model in a Streamlit app so users can upload an image and view the predicted species.

---

## 🏗️ Model Architecture

The final model (`sequential_4`) is built using **transfer learning** with two main components:

| Layer (type) | Output Shape | Parameters |
|--------------|--------------|------------|
| `sequential_1` (Pre-trained base) | (None, 1280) | 410,208 |
| `sequential_3` (Custom classifier head) | (None, 10) | 129,100 |

- **Total params:** 539,308
- **Trainable params:** 525,228
- **Non-trainable params:** 14,080

### Key Concepts Used
- **Convolution layers** – extract spatial features (edges, textures, patterns)
- **Pooling layers** – reduce spatial dimensions via max pooling
- **Fully connected layers** – map extracted features to class probabilities
- **Softmax activation** – produces final probability distribution over 10 classes
- **Transfer Learning** – pre-trained CNN weights used as a frozen/fine-tuned feature extractor to reduce training time and improve generalization on a small dataset

---

## 📊 Dataset

- Images for each of the 10 bird species were collected from publicly available sources.
- **Split:** 85% training / 15% testing
- Images were preprocessed by:
  - Resizing to **224 x 224**
  - Center-cropping for consistent aspect ratio
  - Normalizing pixel values to the range **[-1, 1]**

```python
# Preprocessing pipeline
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
```

---

## 💻 Web App Demo

A simple web interface was built using **Streamlit**, allowing users to upload a bird image and receive an instant species prediction along with class probabilities.

**Example Prediction:**

> Uploaded image → *Red Winged Blackbird*
> Model confidence: **99.7%**

```
[[0.00000016 0.00000051 0.00000794 0.00163547 0.00000014 0.00000089
  0.0000317  0.00129586 0.99700207 0.00002516]]
```

> ✅ Output: *"Image is from Red winged Blackbird species"*

---

## 📈 Results

### Accuracy per Class

| Species | Accuracy |
|---------|----------|
| Black Footed Albatross | 1.00 |
| Laysan Albatross | 0.75 |
| Sooty Albatross | 0.75 |
| Groove Billed Ani | 1.00 |
| Least Auklet | 0.80 |
| Parakeet Auklet | 0.60 |
| Rhinoceros Auklet | 0.60 |
| Brewer Blackbird | 0.50 |
| Red Winged Blackbird | 1.00 |
| Crested Auklet | 0.60 |

### Observations

- The model performs **very well** on visually distinct species (e.g., Black Footed Albatross, Groove Billed Ani, Red Winged Blackbird — all 100% accuracy).
- Lower accuracy on species like **Brewer Blackbird** and **Parakeet Auklet** is largely due to **inter-species visual similarity**, leading to misclassification, as confirmed by the confusion matrix.
- **Loss vs. Epochs** and **Accuracy vs. Epochs** plots indicate a degree of **overfitting** — training loss/accuracy approaches near-perfect values while test loss increases after an initial drop.

### Key Insight

> The gap between training and test performance suggests the model would benefit from a **larger, more diverse dataset** and additional regularization/architectural tuning to better distinguish visually similar species.

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/bird-species-classification.git
cd bird-species-classification
```

### 2. Install dependencies
```bash
pip install tensorflow streamlit pillow numpy
```

### 3. Run the Streamlit app
```bash
streamlit run app1.py
```

### 4. Use the app
- Open the local URL shown in the terminal.
- Upload a bird image (`.png`, `.jpg`, `.jpeg`).
- View the predicted species and class probabilities.

---

## 🚧 Limitations & Future Work

- **Dataset size:** The current dataset is relatively small, contributing to overfitting.
- **Inter-species similarity:** Some species (e.g., Auklets, Blackbirds) share visual features that the current model struggles to differentiate.

**Planned improvements:**
- Expand the dataset with more images per class
- Apply stronger data augmentation
- Add regularization (dropout, L2) and fine-tune additional layers of the pre-trained base
- Experiment with other backbone architectures (ResNet50, InceptionV3, VGG16/19)

---

## 📚 References

1. [How to Use Transfer Learning when Developing CNN Models](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/)
2. [Bird Species Identification Using Deep Learning – IJERT](https://www.ijert.org/bird-species-identification-using-deep-learning)
3. [Adventures in PyTorch Image Classification – CalTech Birds Dataset](https://towardsdatascience.com/adventures-in-pytorch-image-classification-with-caltech-birds-200-part-1-the-dataset-6e5433e9897c)
4. [Image Classification using Google's Teachable Machine – GeeksforGeeks](https://www.geeksforgeeks.org/image-classification-using-googles-teachable-machine/)
5. [Convolutional Neural Networks, Explained](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939)
6. [Deploying an Image Classification Web App with Python](https://tealfeed.com/deploying-image-classification-web-app-python-ae085)

---

## 👤 Author

**[Your Name]**
📧 [your.email@example.com] | 🔗 [LinkedIn](https://linkedin.com/in/your-profile) | 💼 [Portfolio](https://your-portfolio.com)

---

⭐ If you found this project interesting, consider giving it a star!
