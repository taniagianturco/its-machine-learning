# ITS Machine Learning Exercises

Academic exercises completed as part of the 
ITS Aerospazio Puglia curriculum, covering 
text classification and image classification 
with deep learning.

## Projects

### 1. News Article Classification with CNN and GloVe Embeddings
Text classification pipeline that categorizes news 
articles into 20 topic categories using a 1D 
Convolutional Neural Network and pre-trained GloVe 
word embeddings.

**Architecture:**
- Embedding layer (GloVe 6B, 100d, non-trainable)
- Conv1D (128 filters, kernel size 5, ReLU)
- MaxPooling1D → Conv1D → GlobalMaxPooling1D
- Dropout (0.5) → Dense (20 classes, softmax)

**Dataset:** 20 Newsgroups (CMU) — 20,000 posts, 20 categories  
**Notebook:** `CNN_text_classification.ipynb`

---

### 2. Satellite Image Classification with ResNet50
Transfer learning pipeline for satellite image 
classification using ResNet50 pre-trained on ImageNet, 
fine-tuned on a 4-class remote sensing dataset.

**Architecture:**
- Base: ResNet50 (ImageNet weights, frozen backbone)
- GlobalAveragePooling2D → Dropout (0.2)
- Dense (4 classes, softmax)

**Dataset:** Satellite Image Classification — Kaggle  
(mahmoudreda55/satellite-image-classification)  
**Notebook:** `image_classification_ResNet.ipynb`

---

## Tools & Libraries
- Python
- TensorFlow / Keras
- GloVe pre-trained embeddings (Stanford NLP)
- ResNet50 (transfer learning)
- scikit-learn
- NumPy / Seaborn / Matplotlib
- Google Colab

## Author
Tania Gianturco  
[LinkedIn](https://www.linkedin.com/in/tania-gianturco/) | 
[Tableau Public](https://public.tableau.com/app/profile/tania.gianturco/vizzes) | 
[Kaggle](https://www.kaggle.com/taniabee/code)
