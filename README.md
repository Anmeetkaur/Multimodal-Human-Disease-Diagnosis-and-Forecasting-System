MediFuse — Multimodal Health Risk Assessment AI
Project Team ID: MP2025CSE87
Graphic Era (Deemed to be University)
Team Members:

Sangam Sharma — Frontend Lead
Anmeet Kaur — Data Preprocessing & Encoders
Aaradhya Gupta — Fusion Model & XAI
Mansi Agarwal — Backend & MLOps Lead
Guide: Mr. Kireet Joshi
1. Overview — What is MediFuse?
MediFuse is an AI-powered health risk analysis system that takes three inputs:

Symptom Text (NLP via DistilBERT)
Blood Report / Lifestyle Values (MLP Encoder)
Wearable/Signal Data (LSTM Encoder)
The system predicts probabilities for diseases and provides:

Risk percentages
Explainable AI (SHAP-based)
Top contributing clinical factors
Nearby hospitals
Printable health report
Disclaimer:
MediFuse does not diagnose diseases. It only provides risk assessment.
Final decision is always made by a doctor.

2. Folder Structure


D:\medifuse-backend\
│
├── main.py              # FastAPI backend + Full AI pipeline
├── medifuse.html        # Complete frontend UI
├── api.js               # Frontend ↔ Backend connection
└── requirements.txt     # Python dependencies
3. Installation Guide
3.1. Prerequisites
Install the following before starting:

Python 3.9+
Node.js (optional, for frontend improvements)
pip (latest version recommended)
A modern browser (Chrome/Edge)
Git (optional)
4. Backend Setup (FastAPI + Python)
Step 1 — Create Virtual Environment


python -m venv venv
Activate it:

Windows:


venv\Scripts\activate
Mac/Linux:


source venv/bin/activate
Step 2 — Install Dependencies


pip install -r requirements.txt
These include:

FastAPI
Uvicorn
PyTorch
Transformers (DistilBERT)
NumPy
Scikit-learn
Python-dotenv
Step 3 — Run Server


uvicorn main:app --reload --port 8000
If successful, you will see:



Running on http://127.0.0.1:8000
Step 4 — Test Backend
Open in browser:

API Docs: http://localhost:8000/docs
Health Check: http://localhost:8000/health
5. Frontend Setup
The frontend is entirely inside medifuse.html.

Step 1 — Open the File
Simply double-click:



medifuse.html
or open via:

Right Click → Open With → Chrome / Edge.

Step 2 — Backend Auto-Detection
When the frontend loads:

If backend is connected → Green “Connected” banner
If not → Yellow Demo Mode
(UI works offline using sample data)
No configuration needed.

6. How MediFuse Works (Full Pipeline)
Step 1 — User Inputs
Symptom text
Blood/lifestyle values
Wearable data
Step 2 — Encoders (Anmeet’s Work)
Three separate neural networks process the three modalities:

DistilBERT → 768‑dim text vector
MLP → 128‑dim tabular vector
LSTM → 128‑dim signal vector
Step 3 — Cross-Attention Fusion (Aaradhya’s Work)
All vectors are projected to 256 dimensions and fused using:

Multi-head cross-attention
Modality gating (dynamic weighting)
Final 256‑dim unified representation
Step 4 — Disease Prediction
Classifier outputs 6 disease probabilities (sigmoid scaled).

Step 5 — Explainability (SHAP)
For each feature:

AI temporarily removes the feature
Checks how much prediction changes
Shows red (risk increasing) / green (protective) bars
Step 6 — Results Shown on UI
Risk gauge
Disease list
SHAP bar chart
Hospital recommendation
Printable report
7. API Endpoints
GET /
Returns model info.

GET /health
Simple health check.

POST /predict
Main endpoint — runs full AI pipeline.

POST /encode
Returns only encoded vectors (for debugging).

GET /hospitals
Hospital finder.

POST /auth/login
User authentication for Admin panel.

GET /admin/stats
Admin dashboard stats.

8. Team Responsibilities
Member 1 — Sangam (Frontend Lead)
medifuse.html (UI)
Forms, charts, animations
SHAP visualizations
Results screen, report generator
Hospital finder
Admin dashboard
Member 2 — Anmeet (Encoders)
DistilBERT (text)
MLP (tabular)
LSTM (signal)
Data normalization + time-series construction
Member 3 — Aaradhya (Fusion + XAI)
CrossAttentionFusion
Modality-gated fusion
Disease prediction head
SHAP explainability
Member 4 — Mansi (Backend & MLOps)
FastAPI server
All endpoints
Docker setup
Deployment configuration
requirements.txt
Backend–frontend integration
9. Running the Complete System
Step 1
Start the backend:



uvicorn main:app --reload
Step 2
Open the frontend:



medifuse.html
Step 3
Enter:

Symptoms
Blood values
Wearable data
Step 4
Click Run AI Diagnosis.

You will get:

Disease probabilities
SHAP explanations
Hospital suggestions
Full report
10. Important Concepts for Presentation
Multimodal AI → text + tabular + signals
Why 3 encoders? → each input has different structure
Cross-attention → lets modalities influence each other
SHAP XAI → builds doctor trust
DistilBERT → lighter and faster than BERT
Risk assessment vs diagnosis → ethical guardrails
