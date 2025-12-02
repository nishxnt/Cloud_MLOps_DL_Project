# 🌩️ Intel Image Classification – End-to-End MLOps Pipeline on Google Vertex AI

*A production-ready workflow covering data annotation → training → model packaging → deployment → Streamlit UI.*

---

## 📌 Overview

This project demonstrates how to build, train, deploy, and serve a deep-learning image classification model using a modern **MLOps pipeline** on **Google Cloud Vertex AI**.

You implement the full lifecycle:

- Dataset ingestion from **GCS**
- Human-in-the-loop **data annotation** using Vertex AI Labeling
- Model training on **Vertex Workbench**
- Hyperparameter tuning using **Optuna**
- Model packaging with **TorchScript + TorchServe (.mar)**
- Deployment to a **Vertex AI Endpoint**
- Real-time inference through **Python** and a custom **Streamlit Web App**

The model used is **SqueezeNet**, chosen for its lightweight architecture and production-friendly size.

---

## 📁 Project Repository Structure

```bash
├── modules/
│   ├── dataset.py
│   ├── trainer.py
│   ├── utility.py
│   ├── optuna_monashara.py
│   ├── BufferDataset.py
│   └── ...
│
├── saved_model/
│   ├── model_scripted.pt
│   └── model_weights.pth
│
├── model_store/
│   └── model.mar              # TorchServe packaged model
│
├── plots/
│   └── training_loss_curve.png
│
├── squeezenet_handler.py      # Custom TorchServe handler
├── app.py                     # Streamlit UI
├── main_training.py           # Vertex Workbench training script
└── README.md
```
---

## 🚀 MLOps Pipeline Overview

### 1️⃣ Data Storage (GCS Bucket)

All raw and processed images are stored in a Google Cloud Storage bucket:

```bash
gs://mlops-project-intel-data/
```

Dataset structure:

```bash
seg_pred/
├── seg_train/
├── seg_test/
└── seg_pred/   # evaluation set
```

---

### 2️⃣ Data Annotation with Vertex AI

Three-step annotation process:

1. **Created a Vertex Labeling Task**  
   Used *Single-Label Image Classification* to simulate real annotation workflows.

2. **Generated image path CSV using Workbench**  
   A notebook (`data_annotation_csv.ipynb`) extracted all image paths and uploaded them to the annotation UI.

3. **Annotated images using Vertex UI**  
   Human-labelled samples improve data quality and demonstrate annotation workflows.

---

### 3️⃣ Model Architecture

The training script supports:

- **SqueezeNet 1.1**
- Optional **MobileNetV3 Small**

Key steps:

- Pretrained weights optional  
- Classifier head replaced to output **6 classes**  
- Optional layer freezing  
- Optional pruning  
- Optional Optuna hyperparameter tuning  

A simplified view of the SqueezeNet pipeline:

```java
Input Image (150×150)
        ↓
Convolution + ReLU
        ↓
Fire Modules (Squeeze + Expand)
        ↓
Global Avg Pool
        ↓
Custom Classifier (6 classes)
        ↓
Softmax Output
```

---

### 4️⃣ Training on Vertex AI Workbench

- Custom PyTorch pipeline  
- Train/validation split using `random_split`  
- Support for:  
  - Mixed CPU/GPU/MPS  
  - Multi-worker DataLoader  
  - Optuna HPO  
- Loss curves are saved locally & uploaded to GCS  

#### Training Output Includes:

- Training loss curves  
- Test accuracy  
- Saved model weights + TorchScript model

---

## 5️⃣ Packaging the Model for Deployment

Two artifacts are produced:

### **TorchScript (`model_scripted.pt`)**  
Used to run static graph inference.

### **TorchServe `.mar` Archive**  
Generated using:

```bash
torch-model-archiver \
    --model-name intel_squeezenet \
    --version 1.0 \
    --serialized-file model_scripted.pt \
    --handler squeezenet_handler.py \
    --export-path model_store \
    --force
```

Both artifacts are uploaded to GCS.

---

### 6️⃣ Deployment to Vertex AI Endpoint

- Model is **registered** in Vertex AI Model Registry  
- Deployment uses:
  - **Custom Container (TorchServe)**
  - `.mar` file for model loading  
- A dedicated endpoint is created for online prediction

---

### 7️⃣ Testing the Endpoint in Python (Workbench)

A base64-encoded image is passed to the endpoint:

```python
from google.cloud import aiplatform
endpoint = aiplatform.Endpoint(ENDPOINT_ID)

response = endpoint.predict(instances=[{
    "data": {"b64": encoded_image}
}])
```

Vertex returns the predicted class.

---

### 8️⃣ Streamlit Web Application

A simple UI for real-time inference:

- Upload an image  
- Image is encoded → sent to Vertex Endpoint  
- Predicted class displayed instantly  
- Great for demos and stakeholder presentations  

```bash
streamlit run app.py
```

---

## 🧪 Technologies Used

| Component              | Technology                    |
|-----------------------|-------------------------------|
| Framework             | PyTorch                       |
| Serving               | TorchServe                    |
| Cloud Platform        | Google Cloud Vertex AI        |
| Data Storage          | Google Cloud Storage (GCS)    |
| Programming Language  | Python                        |
| UI                    | Streamlit                     |
| HPO                   | Optuna                        |

---

## 🛠️ Key Hyperparameters

| Parameter            | Meaning                       | Example          |
|---------------------|-------------------------------|------------------|
| `choice`            | Which model to use            | 1 = SqueezeNet   |
| `freezeLayer`       | Freeze pretrained layers      | True             |
| `pretrained_Weights`| Load ImageNet weights         | True             |
| `batch_size`        | Training batch size           | 1028             |
| `lr`                | Learning rate                 | 0.9              |
| `epochs`            | Number of training epochs     | 1 (demo)         |
| `prune_model`       | Optional pruning              | False            |
| `OPTUNA_MO`         | Enable Optuna HPO             | False            |

---

## 🛠️ How to Run This Project Locally

### Clone the repo

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit app

```bash
streamlit run app.py
```

---

## 📈 Results

- Successful full MLOps pipeline on Vertex AI  
- Trained a lightweight model suitable for cloud deployment  
- Achieved strong accuracy on the Intel image dataset  
- Packaged & deployed model with TorchServe  
- Built a Streamlit inference UI  

---

## 🚀 Future Improvements

- Automate pipeline using **Vertex AI Pipelines**  
- Add CI/CD with GitHub Actions  
- Use **Vertex Hyperparameter Tuning** instead of Optuna  
- Add monitoring / logging (Prometheus, Vertex Model Monitoring)  
- Expand dataset for better generalization  

---

## 👥 Contributors

- **Nishant Gupta**  
- **Gaurangkumar Monashara**

---

## ⭐ If you found this useful, please star the repo!

