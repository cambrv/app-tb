# Mobile App for Tuberculosis Detection Using Deep Learning and NLP-Based Recommendations

**Manuscript ID:** 10111  

**Authors and Affiliations:**  
- **Camily Bravo-Flores** – Universidad Técnica de Machala, Ecuador  
- **Derik Aranda-Neira** – Universidad Técnica de Machala, Ecuador  
- **Wilmer Rivas-Asanza** – Universidad Técnica de Machala, Ecuador  
- **Bertha Mazon-Olivo** – Universidad Técnica de Machala, Ecuador  

---

## 📁 Repository Contents

### Root structure

| Folder / File | Description |
|---------------|-------------|
| `dataset/` | Dataset folder structure and utilities. Public datasets are referenced in the manuscript; raw images are not redistributed. |
| `deep-learning/` | Complete deep learning pipeline: preprocessing, training, and evaluation of CNN models (DenseNet121, ResNet50, MobileNetV2). |
| `nlp/` | NLP module and resources for recommendation generation (semantic retrieval + LLM integration). |
| `app-movil/` | Mobile app and backend microservices used in the client-server architecture. |

### `app-movil/` structure

| Folder | Description |
|--------|-------------|
| `app-movil/frontend/` | Mobile application (client) used to capture/submit X-ray images and display diagnostic results and recommendations. |
| `app-movil/backend-validate-xrays/` | Image validation service (pre-inference checks / preprocessing) exposed via REST API. |
| `app-movil/backend-cnn/` | CNN inference service (server-side prediction) exposed via REST API. |
| `app-movil/backend-chat/` | Recommendation/chat service integrating NLP + DeepSeek-R1 via OpenRouter (REST API). |

## 📁 Scripts

This repository includes the scripts required to reproduce the experiments reported in the manuscript.

| Script (Path) | Related Results | Description |
|--------------|----------------|-------------|
| `deep-learning/tuberculosis-detection/densenet121/prueba1/train.py` | Table IV, Fig. 6 (a) (b) | Trains the DenseNet121 model using transfer learning and fine-tuning with data augmentation. Optimizes the decision threshold based on F1-score and saves the trained model and training history. |
| `deep-learning/tuberculosis-detection/densenet121/prueba1/test.py` | Table V | Evaluates the DenseNet121 model by selecting an optimal threshold (Recall ≥ 90%). Computes Accuracy, Precision, Recall, F1-score, Sensitivity, Specificity, and AUC-ROC, and generates confusion matrix and performance plots. |
| `deep-learning/tuberculosis-detection/resnet50/test-1/train.py` | Table IV, Fig. 6 (c) (d) | Trains the ResNet50 model using transfer learning, regularization, and early stopping. Saves the trained model and complete training history for reproducibility. |
| `deep-learning/tuberculosis-detection/resnet50/test-1/test.py` | Table V | Evaluates the ResNet50 model by optimizing the decision threshold on the validation set. Computes Accuracy, Precision, Recall, F1-score, Sensitivity, Specificity, and AUC-ROC, and generates confusion matrix visualizations. |
| `deep-learning/tuberculosis-detection/mobilenetv2/train.py` | Table IV, Fig. 6 (e) (f) | Trains the MobileNetV2 model using transfer learning with lightweight architecture optimization for mobile deployment. Saves the trained model and training history. |
| `deep-learning/tuberculosis-detection/mobilenetv2/test.py` | Table V | Evaluates the MobileNetV2 model on the test set, computing Accuracy, Precision, Recall, F1-score, Sensitivity, Specificity, and AUC-ROC, and generates confusion matrix visualizations. |
| `nlp/preprocess-pdfs/main.ipynb` | Section B | Main notebook that implements the complete NLP pipeline: PDF loading, text extraction, preprocessing, chunking, embedding generation using SentenceTransformers, and semantic indexing with FAISS. |

---

### 🔁 Reproducibility Notes

- All experiments use 224×224 RGB chest X-ray images and a batch size of 32.
- Binary cross-entropy loss is used for all CNN models.
- Threshold optimization is explicitly applied to balance false positives and false negatives.
- All evaluation metrics correspond to those reported in the manuscript.

---

## 💻 Requirements

- Python 3.10
- TensorFlow / Keras
- NumPy
- OpenCV
- scikit-learn
- FAISS
- SentenceTransformers

Dependency versions are specified in the corresponding `requirements.txt` files.

---

## ⚠️ Disclaimer

This application is intended as a **clinical decision support tool** and does not replace professional medical diagnosis.

---

## ✉️ Contact

For questions regarding replication or academic use:

**Camily Bravo-Flores**  
cbravo8@utmachala.edu.ec  
Universidad Técnica de Machala  
Ecuador
