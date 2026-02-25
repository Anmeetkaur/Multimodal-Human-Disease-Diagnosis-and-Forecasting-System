# 🏥 Human Disease Analyzer and Forecasting System  
## 📱 App Name: MediFuse


**Project Team ID:** MP2025CSE87  
**Institution:** Graphic Era (Deemed to be University)  
**Guide:** Mr. Kireet Joshi  

---

# 📌 Overview

MediFuse is an AI-powered multimodal health risk assessment system that analyzes:

- 📝 Symptom Text (NLP via DistilBERT)
- 🧪 Blood Report / Lifestyle Values (MLP Encoder)
- ⌚ Wearable / Signal Data (LSTM Encoder)

The system predicts disease risk probabilities and provides:

- 📊 Risk percentages  
- 🧠 Explainable AI (SHAP-based explanations)  
- 🔬 Top contributing clinical factors  
- 🏥 Nearby hospital recommendations  
- 🖨 Printable health report  

> ⚠️ **Disclaimer:**  
> MediFuse does NOT diagnose diseases.  
> It only provides AI-based risk assessment.  
> Final medical decisions must always be made by a certified doctor.

---

# 🧠 How MediFuse Works

## 🔹 Step 1 — User Inputs

- Symptom text  
- Blood / lifestyle parameters  
- Wearable time-series data  

---

## 🔹 Step 2 — Modality-Specific Encoders

Each input type is processed by a separate neural network:

| Modality | Model Used | Output |
|----------|------------|--------|
| Text     | DistilBERT | 768-dim vector |
| Tabular  | MLP        | 128-dim vector |
| Signal   | LSTM       | 128-dim vector |

---

## 🔹 Step 3 — Cross-Attention Fusion

All embeddings are:

- Projected to 256 dimensions  
- Fused using multi-head cross-attention  
- Weighted via modality gating  

Final Output → Unified 256-dimensional representation  

---

## 🔹 Step 4 — Disease Prediction

A classification head outputs:

- 6 disease probability scores  
- Sigmoid-scaled risk percentages  

---

## 🔹 Step 5 — Explainable AI (SHAP)

SHAP identifies:

- Features increasing risk (🔴 red bars)
- Protective features (🟢 green bars)

This ensures transparency and clinical trust.

---

# 🏗 Folder Structure

```
D:\medifuse-backend\
│
├── main.py              # FastAPI backend + Full AI pipeline
├── medifuse.html        # Complete frontend UI
├── api.js               # Frontend ↔ Backend connection
└── requirements.txt     # Python dependencies
```

---

# ⚙️ Installation Guide

## 1️⃣ Prerequisites

- Python 3.9+
- pip (latest version recommended)
- Modern Browser (Chrome / Edge)
- Git (optional)
- Node.js (optional)

---

# 🚀 Backend Setup (FastAPI)

## Step 1 — Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows
```bash
venv\Scripts\activate
```

### Mac/Linux
```bash
source venv/bin/activate
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

- FastAPI  
- Uvicorn  
- PyTorch  
- Transformers (DistilBERT)  
- NumPy  
- Scikit-learn  
- Python-dotenv  

---

## Step 3 — Run Server

```bash
uvicorn main:app --reload --port 8000
```

If successful:

```
Running on http://127.0.0.1:8000
```

---

## Step 4 — Test Backend

Open in browser:

- API Docs → http://localhost:8000/docs  
- Health Check → http://localhost:8000/health  

---

# 💻 Frontend Setup

The frontend is fully contained in:

```
medifuse.html
```

## Step 1 — Open the File

Double-click:

```
medifuse.html
```

or Right Click → Open With → Chrome / Edge

## Step 2 — Backend Auto Detection

- If backend connected → Green “Connected” banner  
- If backend not running → Yellow “Demo Mode”  

No manual configuration required.

---

# 🔌 API Endpoints

| Method | Endpoint        | Description |
|--------|-----------------|-------------|
| GET    | /              | Model info |
| GET    | /health        | Health check |
| POST   | /predict       | Full AI pipeline |
| POST   | /encode        | Encoded vectors only |
| GET    | /hospitals     | Hospital finder |
| POST   | /auth/login    | Admin authentication |
| GET    | /admin/stats   | Admin dashboard stats |

---

# 👩‍💻 Team Responsibilities

### 🟢  Frontend Lead
- medifuse.html UI
- Forms & animations
- SHAP visualization
- Results dashboard
- Report generator
- Hospital finder
- Admin panel

### 🔵 Encoders
- DistilBERT (text encoder)
- MLP (tabular encoder)
- LSTM (signal encoder)
- Data normalization
- Time-series processing

### 🟣 Fusion + XAI
- CrossAttentionFusion module
- Modality gating
- Disease prediction head
- SHAP explainability

### 🟠 Backend & MLOps
- FastAPI server
- All API endpoints
- Docker setup
- Deployment configuration
- requirements.txt
- Backend–frontend integration

---

# 🏁 Running the Complete System

## Step 1
```bash
uvicorn main:app --reload
```

## Step 2
Open:

```
medifuse.html
```

## Step 3
Enter:
- Symptoms
- Blood values
- Wearable data

## Step 4
Click:

```
Run AI Diagnosis
```

You will receive:

- Disease probabilities  
- SHAP explanations  
- Hospital suggestions  
- Printable report  

---

# 🔬 Core Concepts

- Multimodal AI → text + tabular + signals  
- Separate encoders → each modality has different structure  
- Cross-attention → modalities influence each other  
- SHAP XAI → builds medical trust  
- DistilBERT → lighter & faster than BERT  
- Risk assessment ≠ Diagnosis (ethical AI guardrails)

---

# 📜 License

This project is developed for academic and research purposes.

---

# ❤️ Built With

- Python  
- FastAPI  
- PyTorch  
- Transformers  
- HTML/CSS/JavaScript  
