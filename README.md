# Multimodal-Human-Disease-Diagnosis-and-Forecasting-System
##Overview

This project is an AI-based health risk assessment system that processes three modalities—textual symptoms, tabular lifestyle factors, and wearable signal data. It uses a Cross-Attention Fusion Network to generate risk scores and a Trust Factor (XAI) explaining the key contributors.
It is designed as a responsible pre-clinic assessment tool and does not provide medical diagnosis.

##Tech Stack

Frontend: React
Backend: Flask / FastAPI
AI/ML: BERT, MLP, RNN, Cross-Attention Fusion, XAI
MLOps: Docker, Cloud/Local Deployment

##Features

Multimodal data input (text, tabular, signal)

Independent encoders for each modality

Cross-attention fusion for unified feature representation

Risk scoring for potential conditions

Explainable output showing contributing features

Functional web interface via React

##System Architecture

Preprocessing of all three modalities

Individual encoders: BERT (text), MLP (tabular), RNN (signal)

Cross-Attention Fusion Network

Risk prediction module

XAI-based explanation

Frontend display of output

##Team Responsibilities

1. Frontend Development: React UI, input forms, result display, API integration
2. Data Preprocessing & Encoders: Cleaning and encoding of text, tabular, and signal data
3. Fusion & Prediction Model: Cross-attention fusion, risk scoring, XAI
4. Backend & MLOps: API development, model integration, Docker packaging, deployment
