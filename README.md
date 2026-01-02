# 🛠 PrognosAI — Predictive Maintenance System

PrognosAI is a deep learning–based predictive maintenance dashboard built using GRU networks on the NASA CMAPSS dataset.

## 🚀 Features
- Remaining Useful Life (RUL) prediction
- GRU deep learning models
- Multi-dataset support (FD001–FD004)
- Interactive Streamlit dashboard
- Fleet health monitoring (Critical / Warning / Normal)

## 📊 Dataset
NASA CMAPSS Turbofan Engine Degradation Dataset

## 🧠 Model
- GRU Neural Network
- Sequence length: 30
- Scaled sensor inputs
- Trained separately for each FD dataset

## 🖥️ Dashboard
- Dataset selection
- Engine-wise RUL prediction
- Alert classification
- Downloadable results

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app/app.py
